import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.rcParams.update({"text.usetex":True, "font.family": "serif", "figure.dpi": 150})
import os
import multiprocessing as mp
from dask.distributed import Client
import json
import pandas as pd
import matplotlib as mpl
import scipy.ndimage as ndimage
mpl.rcParams.update(mpl.rcParamsDefault)
from colossus.cosmology import cosmology
import pickle
from sparta_tools import sparta

from utils.ML_support import get_CUDA_cluster, get_combined_name, parse_ranges, create_nu_string, load_sparta_mass_prf, create_stack_mass_prf, split_calc_name, load_SPARTA_data, reform_dataset_dfs
from utils.data_and_loading_functions import create_directory, load_pickle, conv_halo_id_spid
from utils.ps_cut_support import load_ps_data
from utils.update_vis_fxns import plt_SPARTA_KE_dist, compare_split_prfs
from utils.calculation_functions import calculate_density, filter_prf, calc_mass_acc_rate


import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")

on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")

ML_dset_path = config["PATHS"]["ML_dset_path"]
path_to_models = config["PATHS"]["path_to_models"]
SPARTA_output_path = config["PATHS"]["SPARTA_output_path"]

model_sims = json.loads(config.get("XGBOOST","model_sims"))
dask_task_cpus = config.getint("XGBOOST","dask_task_cpus")
model_type = config["XGBOOST"]["model_type"]
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))
dask_task_cpus = config.getint("XGBOOST","dask_task_cpus")

sim_cosmol = config["MISC"]["sim_cosmol"]
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
    cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
else:
    cosmol = cosmology.setCosmology(sim_cosmol) 
    sim_pat = r"cbol_l(\d+)_n(\d+)"
    
plt_nu_splits = config["XGBOOST"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)

plt_macc_splits = config["XGBOOST"]["plt_macc_splits"]
plt_macc_splits = parse_ranges(plt_macc_splits)

nu_splits = config["XGBOOST"]["nu_splits"]
nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

linthrsh = config.getfloat("XGBOOST","linthrsh")
lin_nbin = config.getint("XGBOOST","lin_nbin")
log_nbin = config.getint("XGBOOST","log_nbin")
lin_rvticks = json.loads(config.get("XGBOOST","lin_rvticks"))
log_rvticks = json.loads(config.get("XGBOOST","log_rvticks"))
lin_tvticks = json.loads(config.get("XGBOOST","lin_tvticks"))
log_tvticks = json.loads(config.get("XGBOOST","log_tvticks"))
lin_rticks = json.loads(config.get("XGBOOST","lin_rticks"))
log_rticks = json.loads(config.get("XGBOOST","log_rticks"))

if on_zaratan:
    from dask_mpi import initialize
    from distributed.scheduler import logger
    import socket
elif not on_zaratan and not use_gpu:
    from dask.distributed import LocalCluster
    
    
def overlap_loss_orb(params, r_bin, lnv2_bin, sparta_labels_bin):
    m, b = params
    line_classification = (lnv2_bin <= (m * r_bin + b)).astype(int)
    return -np.sum(line_classification == sparta_labels_bin)  # Negative for maximization

def overlap_loss_inf(params, r_bin, lnv2_bin, sparta_labels_bin):
    m, b = params
    line_classification = (lnv2_bin >= (m * r_bin + b)).astype(int)
    return -np.sum(line_classification == sparta_labels_bin)  # Negative for maximization

def overlap_loss(params, lnv2_bin, sparta_labels_bin):
    decision_boundary = params
    line_classif = (lnv2_bin <= decision_boundary).astype(int)  

    # Count misclassified particles
    misclass_orb = np.sum((line_classif == 0) & (sparta_labels_bin == 1))
    misclass_inf = np.sum((line_classif == 1) & (sparta_labels_bin == 0))

    # Loss is the absolute difference between the two misclassification counts
    return abs(misclass_orb - misclass_inf)

def opt_func(bins, r, lnv2, sparta_labels, def_m, def_b, orb = True, plot_loc = "", title = ""):
    # Assign bin indices based on radius
    bin_indices = np.digitize(r, bins) - 1  
    
    magma_cmap = plt.get_cmap("magma")
    magma_cmap.set_under(color='black')
    magma_cmap.set_bad(color='black') 

    intercepts = []
    lnv2_range = (np.min(lnv2),np.max(lnv2))
    for i in range(bins.shape[0]-1):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            intercepts.append(def_b)
            continue  # Skip empty bins
        
        r_bin = r[mask]
        lnv2_bin = lnv2[mask]
        sparta_labels_bin = sparta_labels[mask]

        # Optimize
        if orb:
            # Initial guess for m and b
            initial_guess = [np.max(lnv2_bin)]
            result = minimize(overlap_loss_orb, initial_guess, args=(r_bin, lnv2_bin, sparta_labels_bin))
        elif orb is False:
            # Initial guess for m and b
            initial_guess = [np.mean(lnv2_bin)]
            result = minimize(overlap_loss_inf, initial_guess, args=(r_bin, lnv2_bin, sparta_labels_bin))
        else:
            # initial_guess = [np.min(lnv2_bin)]
            # result_min = minimize(overlap_loss, initial_guess, args=(lnv2_bin, sparta_labels_bin), method="Nelder-Mead")
            
            # initial_guess = [np.max(lnv2_bin)]
            # result_max = minimize(overlap_loss, initial_guess, args=(lnv2_bin, sparta_labels_bin), method="Nelder-Mead")
            
            initial_guess = [np.mean(lnv2_bin)]
            result_mean = minimize(overlap_loss, initial_guess, args=(lnv2_bin, sparta_labels_bin), method="Nelder-Mead")
            
            # if result_min.fun < result_max.fun:
            #     result = result_min
            # else:
            #     result = result_max
            
            result = result_mean
            
            calc_b = result.x[0]
            if calc_b < -6:
                calc_b = -6
            elif calc_b > 4:
                calc_b = 4
                
            create_directory(plot_loc + title + "bins/")
        
            fig_miss, ax_miss = plt.subplots(1,2,figsize=(14,7),share_y=True)
            fig_dist, ax_dist = plt.subplots(1,2,figsize=(14,7))
            for vel in np.arange(lnv2_range[0],lnv2_range[1],0.01):
                line_classif = (lnv2_bin <= vel).astype(int)  
                misclass_orb = np.sum((line_classif == 0) & (sparta_labels_bin == 1))
                misclass_inf = np.sum((line_classif == 1) & (sparta_labels_bin == 0))
                ax_miss[0].scatter(vel,misclass_inf)
                ax_miss[1].scatter(vel,misclass_orb)
                
                ax_dist[0].hist2d(r_bin[np.where(sparta_labels_bin == 0)[0]], lnv2_bin[np.where(sparta_labels_bin == 0)[0]], 
                        range = [[bins[i],bins[i+1]],lnv2_range], bins=200, norm="log", cmap=magma_cmap)
                ax_dist[0].hlines(calc_b,xmin=bins[i],xmax=bins[i+1])
                ax_dist[1].hist2d(r_bin[np.where(sparta_labels_bin == 1)[0]], lnv2_bin[np.where(sparta_labels_bin == 1)[0]],
                        range = [[bins[i],bins[i+1]],lnv2_range], bins=200, norm="log", cmap=magma_cmap)
                ax_dist[1].hlines(calc_b,xmin=bins[i],xmax=bins[i+1])
            
            ax_miss[0].set_xlabel(r'$\ln(v^2/v_{200m}^2)$')
            ax_miss[0].set_ylabel("Number of Misclassified Particles")
            ax_miss[0].set_title("Infalling Particles Classified as Orbiting")
            ax_miss[1].set_xlabel(r'$\ln(v^2/v_{200m}^2)$')
            ax_miss[1].set_ylabel("Number of Misclassified Particles")
            ax_miss[1].set_title("Orbiting Particles Classified as Infalling")
            create_directory(plot_loc + title + "bins/miss_class/")
            fig_miss.savefig(plot_loc + title + "bins/miss_class/miss_class_bin_" + str(i) + ".png")
            plt.close(fig_miss)
            
            
            ax_dist[0].set_title("Infalling particles")
            ax_dist[0].set_ylabel(r'$\ln(v^2/v_{200m}^2)$')
            ax_dist[0].set_ylim(lnv2_range)
            
            ax_dist[1].set_title("Orbiting particles")
            ax_dist[1].set_ylabel(r'$\ln(v^2/v_{200m}^2)$')
            ax_dist[1].set_xlabel(r'$r/R_{200m}$')
            ax_dist[1].set_ylim(lnv2_range)
            fig_dist.suptitle("Bin " + str(i) + "Radius: " + str(bins[i]) + "-" + str(bins[i+1]))
            create_directory(plot_loc + title + "bins/dist/")
            fig_dist.savefig(plot_loc + title + "bins/dist/bin_" + str(i) + ".png")          
            plt.close(fig_dist)
            
        intercepts.append(calc_b)
        
    return {"b":intercepts}
    
if __name__ == "__main__":
    if use_gpu:
        mp.set_start_method("spawn")

    if on_zaratan:            
        if use_gpu:
            initialize(local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        else:
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
            else:
                print("SLURM_CPUS_PER_TASK is not defined.")
            initialize(nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")

        print("Initialized")
        client = Client()
        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        login_node_address = "zvladimi@login.zaratan.umd.edu" # Change this to the address/domain of your login node

        logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
    else:
        if use_gpu:
            client = get_CUDA_cluster()
        else:
            tot_ncpus = mp.cpu_count()
            n_workers = int(np.floor(tot_ncpus / dask_task_cpus))
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=dask_task_cpus,
                memory_limit='5GB'  
            )
            client = Client(cluster)
    
    model_comb_name = get_combined_name(model_sims) 
    model_dir = model_type + "_" + model_comb_name + "nu" + nu_string 
    model_save_loc = path_to_models + model_comb_name + "/" + model_dir + "/"    
    
    curr_test_sims = test_sims[0]
    test_comb_name = get_combined_name(curr_test_sims) 
    dset_name = eval_datasets[0]
    plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
    create_directory(plot_loc)
    
    ps_param_dict = {
        "m_pos": -1.9973747688461672,
        "b_pos": 2.730691113802748,
        "m_neg": -1.601325049968688,
        "b_neg": 1.5101195108968333,
    }
    
    r, vr, lnv2, sparta_labels, my_data, halo_df = load_ps_data(client)
    
    r = my_data["p_Scaled_radii"].compute().to_numpy()
    vr = my_data["p_Radial_vel"].compute().to_numpy()
    vt = my_data["p_Tangential_vel"].compute().to_numpy()
    vphys = my_data["p_phys_vel"].compute().to_numpy()
    sparta_labels = my_data["Orbit_infall"].compute().to_numpy()
    lnv2 = np.log(vphys**2)
    
    c_r = my_data["c_Scaled_radii"].compute().to_numpy()
    c_vr = my_data["c_Radial_vel"].compute().to_numpy()
    c_vt = my_data["c_Tangential_vel"].compute().to_numpy()
    
    sparta_orb = np.where(sparta_labels == 1)[0]
    sparta_inf = np.where(sparta_labels == 0)[0]

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r < 1.75
    
    fltr_combs = {
    "orb_vr_neg": np.intersect1d(sparta_orb, np.where(mask_vr_neg)[0]),
    "orb_vr_pos": np.intersect1d(sparta_orb, np.where(mask_vr_pos)[0]),
    "inf_vr_neg": np.intersect1d(sparta_inf, np.where(mask_vr_neg)[0]),
    "inf_vr_pos": np.intersect1d(sparta_inf, np.where(mask_vr_pos)[0]),
    }

    split_scale_dict = {
            "linthrsh":linthrsh, 
            "lin_nbin":lin_nbin,
            "log_nbin":log_nbin,
            "lin_rvticks":lin_rvticks,
            "log_rvticks":log_rvticks,
            "lin_tvticks":lin_tvticks,
            "log_tvticks":log_tvticks,
            "lin_rticks":lin_rticks,
            "log_rticks":log_rticks,
    }

    halo_first = halo_df["Halo_first"].values
    halo_n = halo_df["Halo_n"].values
    all_idxs = halo_df["Halo_indices"].values

    all_z = []
    all_rhom = []
    # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
    sim_splits = np.where(halo_first == 0)[0]

    use_sims = curr_test_sims

    # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
    # stacked simulations such that they correspond to the larger dataset and not one specific simulation
    if len(use_sims) > 1:
        for i,sim in enumerate(use_sims):
            # The first sim remains the same
            if i == 0:
                continue
            # Else if it isn't the final sim 
            elif i < len(use_sims) - 1:
                halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
            # Else if the final sim
            else:
                halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
    
    act_mass_prf_all, act_mass_prf_orb,all_masses,bins = load_sparta_mass_prf(sim_splits,all_idxs,use_sims)
    act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb  
    
    vr_pos = opt_func(bins, r[mask_vr_pos], lnv2[mask_vr_pos], sparta_labels[mask_vr_pos], 0, ps_param_dict["b_pos"], orb = None, plot_loc = plot_loc, title = "pos")
    vr_neg = opt_func(bins, r[mask_vr_neg], lnv2[mask_vr_neg], sparta_labels[mask_vr_neg], 0, ps_param_dict["b_neg"], orb = None, plot_loc = plot_loc, title = "neg")
    
    opt_param_dict = {
        "orb_vr_pos": vr_pos,
        "orb_vr_neg": vr_neg,
        "inf_vr_neg": vr_neg,
        "inf_vr_pos": vr_pos,
    }
    
    width = 0.05
    perc = 0.99
    grad_lims = "0.2_0.5"
    r_cut = 1.75    
    
    plt_SPARTA_KE_dist(ps_param_dict, fltr_combs, bins, r, lnv2, perc = perc, width = width, r_cut = r_cut, plot_loc = plot_loc, title = "bin_fit_", cust_line_dict = opt_param_dict)

    # Get the redshifts for each simulation's primary snapshot
    for i,sim in enumerate(use_sims):
        with open(ML_dset_path + sim + "/config.pickle", "rb") as file:
            config_dict = pickle.load(file)
            curr_z = config_dict["p_snap_info"]["red_shift"][()]
            all_z.append(curr_z)
            all_rhom.append(cosmol.rho_m(curr_z))
            h = config_dict["p_snap_info"]["h"][()]

    tot_num_halos = halo_n.shape[0]
    min_disp_halos = int(np.ceil(0.3 * tot_num_halos))

    # Get SPARTA's mass profiles
    act_mass_prf_all, act_mass_prf_orb,all_masses,bins = load_sparta_mass_prf(sim_splits,all_idxs,use_sims)
    act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb


    bin_indices = np.digitize(r, bins) - 1  
    preds = np.zeros(r.shape[0])
    for i in range(bins.shape[0]-1):
        mask_pos = (bin_indices == i) & (vr > 0) & (lnv2 < ps_param_dict["b_pos"][i])
        mask_neg = (bin_indices == i) & (vr < 0) & (lnv2 < ps_param_dict["b_neg"][i])
        
        preds[mask_pos] = 1
        preds[mask_neg] = 1

    calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m = create_stack_mass_prf(sim_splits,radii=r, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds, prf_bins=bins, use_mp=True, all_z=all_z)

    # Halos that get returned with a nan R200m mean that they didn't meet the required number of ptls within R200m and so we need to filter them from our calculated profiles and SPARTA profiles 
    small_halo_fltr = np.isnan(calc_r200m)
    act_mass_prf_all[small_halo_fltr,:] = np.nan
    act_mass_prf_orb[small_halo_fltr,:] = np.nan
    act_mass_prf_inf[small_halo_fltr,:] = np.nan

    # Calculate the density by divide the mass of each bin by the volume of that bin's radius
    calc_dens_prf_all = calculate_density(calc_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    calc_dens_prf_orb = calculate_density(calc_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    calc_dens_prf_inf = calculate_density(calc_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)

    act_dens_prf_all = calculate_density(act_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    act_dens_prf_orb = calculate_density(act_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    act_dens_prf_inf = calculate_density(act_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)

    # If we want the density profiles to only consist of halos of a specific peak height (nu) bin 

    all_prf_lst = []
    orb_prf_lst = []
    inf_prf_lst = []
    cpy_plt_nu_splits = plt_nu_splits.copy()
    for i,nu_split in enumerate(cpy_plt_nu_splits):
        # Take the second element of the where to filter by the halos (?)
        fltr = np.where((calc_nus > nu_split[0]) & (calc_nus < nu_split[1]))[0]
        if fltr.shape[0] > 25:
            all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
            orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
            inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
        else:
            plt_nu_splits.remove(nu_split)
            
    curr_halos_r200m_list = []
    past_halos_r200m_list = []                
            
    for sim in use_sims:
        config_dict = load_pickle(ML_dset_path + sim + "/config.pickle")
        p_snap = config_dict["p_snap_info"]["ptl_snap"][()]
        curr_z = config_dict["p_snap_info"]["red_shift"][()]
        # TODO make this generalizable to when the snapshot separation isn't just 1 dynamical time as needed for mass accretion calculation
        # we can just use the secondary snap here because we already chose to do 1 dynamical time for that snap
        past_z = config_dict["c_snap_info"]["red_shift"][()] 
        p_sparta_snap = config_dict["p_snap_info"]["sparta_snap"][()]
        c_sparta_snap = config_dict["c_snap_info"]["sparta_snap"][()]
        
        sparta_name, sparta_search_name = split_calc_name(sim)
        
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"
                
        # Load the halo's positions and radii
        param_paths = [["halos","R200m"],["halos","id"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, p_snap)

        curr_halos_r200m = sparta_params[sparta_param_names[0]][:,p_sparta_snap]
        curr_halos_ids = sparta_params[sparta_param_names[1]][:,p_sparta_snap]
        
        halo_ddf = reform_dataset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/")
        all_idxs = halo_ddf["Halo_indices"].values
        
        use_halo_r200m = curr_halos_r200m[all_idxs]
        use_halo_ids = curr_halos_ids[all_idxs]
        
        sparta_output = sparta.load(filename=curr_sparta_HDF5_path, halo_ids=use_halo_ids, log_level=0)
        new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta re-sort the indices
        
        curr_halos_r200m_list.append(sparta_output['halos']['R200m'][:,p_sparta_snap])
        past_halos_r200m_list.append(sparta_output['halos']['R200m'][:,c_sparta_snap])
        
    curr_halos_r200m = np.concatenate(curr_halos_r200m_list)
    past_halos_r200m = np.concatenate(past_halos_r200m_list)
        
    calc_maccs = calc_mass_acc_rate(curr_halos_r200m,past_halos_r200m,curr_z,past_z)

    cpy_plt_macc_splits = plt_macc_splits.copy()
    for i,macc_split in enumerate(cpy_plt_macc_splits):
        # Take the second element of the where to filter by the halos (?)
        fltr = np.where((calc_maccs > macc_split[0]) & (calc_maccs < macc_split[1]))[0]
        if fltr.shape[0] > 25:
            all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
            orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
            inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
        else:
            plt_macc_splits.remove(macc_split)
    
    lin_rticks = json.loads(config.get("XGBOOST","lin_rticks"))
    compare_split_prfs(plt_nu_splits,len(cpy_plt_nu_splits),all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_loc,title= "perc_" + str(perc) + "_" + grad_lims + "_ps_cut_dens_",prf_name_0="Fitted Phase Space Cut", prf_name_1="SPARTA")