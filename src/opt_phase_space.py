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

from utils.ML_support import setup_client, get_combined_name, parse_ranges, load_sparta_mass_prf, create_stack_mass_prf, split_calc_name, load_SPARTA_data, reform_dataset_dfs, get_model_name
from utils.data_and_loading_functions import create_directory, load_pickle, conv_halo_id_spid, load_config, save_pickle, load_pickle, timed
from utils.ps_cut_support import load_ps_data
from src.utils.vis_fxns import plt_SPARTA_KE_dist, compare_split_prfs
from utils.calculation_functions import calculate_density, filter_prf, calc_mass_acc_rate

config_dict = load_config(os.getcwd() + "/config.ini")

ML_dset_path = config_dict["PATHS"]["ml_dset_path"]
path_to_models = config_dict["PATHS"]["path_to_models"]
SPARTA_output_path = config_dict["SPARTA_DATA"]["sparta_output_path"]

model_sims = config_dict["TRAIN_MODEL"]["model_sims"]
model_type = config_dict["TRAIN_MODEL"]["model_type"]
test_sims = config_dict["EVAL_MODEL"]["test_sims"]
eval_datasets = config_dict["EVAL_MODEL"]["eval_datasets"]

sim_cosmol = config_dict["MISC"]["sim_cosmol"]

plt_nu_splits = parse_ranges(config_dict["EVAL_MODEL"]["plt_nu_splits"])

plt_macc_splits = parse_ranges(config_dict["EVAL_MODEL"]["plt_macc_splits"])

linthrsh = config_dict["EVAL_MODEL"]["linthrsh"]
lin_nbin = config_dict["EVAL_MODEL"]["lin_nbin"]
log_nbin = config_dict["EVAL_MODEL"]["log_nbin"]
lin_rvticks = config_dict["EVAL_MODEL"]["lin_rvticks"]
log_rvticks = config_dict["EVAL_MODEL"]["log_rvticks"]
lin_tvticks = config_dict["EVAL_MODEL"]["lin_tvticks"]
log_tvticks = config_dict["EVAL_MODEL"]["log_tvticks"]
lin_rticks = config_dict["EVAL_MODEL"]["lin_rticks"]
log_rticks = config_dict["EVAL_MODEL"]["log_rticks"]
    
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
    cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
else:
    cosmol = cosmology.setCosmology(sim_cosmol) 
    sim_pat = r"cbol_l(\d+)_n(\d+)"
    
def overlap_loss(params, lnv2_bin, sparta_labels_bin):
    decision_boundary = params[0]
    line_classif = (lnv2_bin <= decision_boundary).astype(int)  

    # Count misclassified particles
    misclass_orb = np.sum((line_classif == 0) & (sparta_labels_bin == 1))
    misclass_inf = np.sum((line_classif == 1) & (sparta_labels_bin == 0))

    # Loss is the absolute difference between the two misclassification counts
    return abs(misclass_orb - misclass_inf)

def opt_func(bins, r, lnv2, sparta_labels, def_b, plot_loc = "", title = ""):
    # Assign bin indices based on radius
    bin_indices = np.digitize(r, bins) - 1  
    
    magma_cmap = plt.get_cmap("magma")
    magma_cmap.set_under(color='black')
    magma_cmap.set_bad(color='black') 

    intercepts = []
    for i in range(bins.shape[0]-1):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            intercepts.append(def_b)
            continue  # Skip empty bins
        
        lnv2_bin = lnv2[mask]
        sparta_labels_bin = sparta_labels[mask]

        # Optimize

        # initial_guess = [np.min(lnv2_bin)]
        # result_min = minimize(overlap_loss, initial_guess, args=(lnv2_bin, sparta_labels_bin), method="Nelder-Mead")
        
        initial_guess = [np.max(lnv2_bin)]
        result_max = minimize(overlap_loss, initial_guess, args=(lnv2_bin, sparta_labels_bin), method="Nelder-Mead")
        
        initial_guess = [np.mean(lnv2_bin)]
        result_mean = minimize(overlap_loss, initial_guess, args=(lnv2_bin, sparta_labels_bin), method="Nelder-Mead")
        
        if result_mean.fun < result_max.fun:
            result = result_mean
        else:
            result = result_max
        
        # result = result_mean
        
        calc_b = result.x[0]
            
        create_directory(plot_loc + title + "bins/")
    
        # fig_miss, ax_miss = plt.subplots(1,2,figsize=(14,7),share_y=True)
        # fig_dist, ax_dist = plt.subplots(1,2,figsize=(14,7))
        # for vel in np.arange(lnv2_range[0],lnv2_range[1],0.01):
        #     line_classif = (lnv2_bin <= vel).astype(int)  
        #     misclass_orb = np.sum((line_classif == 0) & (sparta_labels_bin == 1))
        #     misclass_inf = np.sum((line_classif == 1) & (sparta_labels_bin == 0))
        #     ax_miss[0].scatter(vel,misclass_inf)
        #     ax_miss[1].scatter(vel,misclass_orb)
            
        #     ax_dist[0].hist2d(r_bin[np.where(sparta_labels_bin == 0)[0]], lnv2_bin[np.where(sparta_labels_bin == 0)[0]], 
        #             range = [[bins[i],bins[i+1]],lnv2_range], bins=200, norm="log", cmap=magma_cmap)
        #     ax_dist[0].hlines(calc_b,xmin=bins[i],xmax=bins[i+1])
        #     ax_dist[1].hist2d(r_bin[np.where(sparta_labels_bin == 1)[0]], lnv2_bin[np.where(sparta_labels_bin == 1)[0]],
        #             range = [[bins[i],bins[i+1]],lnv2_range], bins=200, norm="log", cmap=magma_cmap)
        #     ax_dist[1].hlines(calc_b,xmin=bins[i],xmax=bins[i+1])
        
        # ax_miss[0].set_xlabel(r'$\ln(v^2/v_{200m}^2)$')
        # ax_miss[0].set_ylabel("Number of Misclassified Particles")
        # ax_miss[0].set_title("Infalling Particles Classified as Orbiting")
        # ax_miss[1].set_xlabel(r'$\ln(v^2/v_{200m}^2)$')
        # ax_miss[1].set_ylabel("Number of Misclassified Particles")
        # ax_miss[1].set_title("Orbiting Particles Classified as Infalling")
        # create_directory(plot_loc + title + "bins/miss_class/")
        # fig_miss.savefig(plot_loc + title + "bins/miss_class/miss_class_bin_" + str(i) + ".png")
        # plt.close(fig_miss)
        
        
        # ax_dist[0].set_title("Infalling particles")
        # ax_dist[0].set_ylabel(r'$\ln(v^2/v_{200m}^2)$')
        # ax_dist[0].set_ylim(lnv2_range)
        
        # ax_dist[1].set_title("Orbiting particles")
        # ax_dist[1].set_ylabel(r'$\ln(v^2/v_{200m}^2)$')
        # ax_dist[1].set_xlabel(r'$r/R_{200m}$')
        # ax_dist[1].set_ylim(lnv2_range)
        # fig_dist.suptitle("Bin " + str(i) + "Radius: " + str(bins[i]) + "-" + str(bins[i+1]))
        # create_directory(plot_loc + title + "bins/dist/")
        # fig_dist.savefig(plot_loc + title + "bins/dist/bin_" + str(i) + ".png")          
        # plt.close(fig_dist)
            
        intercepts.append(calc_b)
        
    return {"b":intercepts}
    
if __name__ == "__main__":
    client = setup_client()
    
    comb_model_sims = get_combined_name(model_sims) 
        
    model_name = get_model_name(model_type, model_sims, hpo_done=config_dict["OPTIMIZE"]["hpo"], opt_param_dict=config_dict["OPTIMIZE"])    
    model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"  
    
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
    
    param_path = model_fldr_loc + "ps_optparam_dict.pickle"
    if os.path.exists(param_path):
        ps_param_dict = load_pickle(param_path)
        m_pos = ps_param_dict["m_pos"]
        b_pos = ps_param_dict["b_pos"]
        m_neg = ps_param_dict["m_neg"]
        b_neg = ps_param_dict["b_neg"]
    else:
        raise FileNotFoundError(
            f"Parameter file not found at {param_path}. Please run the optimization code to generate it."
        )

    
    #TODO load this from a file/config
    width = 0.05
    perc = 0.99
    grad_lims = "0.2_0.5"
    r_cut = 1.75  
    
    #TODO make this a loop
    curr_test_sims = test_sims[0]
    test_comb_name = get_combined_name(curr_test_sims) 
    dset_name = eval_datasets[0]
    plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
    create_directory(plot_loc)
    
    if os.path.isfile(model_fldr_loc + "bin_fit_ps_cut_params.pickle"):
        print("Loading parameters from saved file")
        opt_param_dict = load_pickle(model_fldr_loc + "bin_fit_ps_cut_params.pickle")
        
        with timed("Loading Testing Data"):
            r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ps_data(client,curr_test_sims=curr_test_sims)
            r_test = my_data["p_Scaled_radii"].compute().to_numpy()
            vr_test = my_data["p_Radial_vel"].compute().to_numpy()
            vphys_test = my_data["p_phys_vel"].compute().to_numpy()
            sparta_labels_test = my_data["Orbit_infall"].compute().to_numpy()
            lnv2_test = np.log(vphys_test**2)
            
            halo_first = halo_df["Halo_first"].values
            halo_n = halo_df["Halo_n"].values
            all_idxs = halo_df["Halo_indices"].values
            # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
            sim_splits = np.where(halo_first == 0)[0]
        
            sparta_orb = np.where(sparta_labels_test == 1)[0]
            sparta_inf = np.where(sparta_labels_test == 0)[0]
    else:        
        with timed("Optimizing phase-space cut"):
            with timed("Loading Fitting Data"):
                r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ps_data(client,curr_test_sims=model_sims)
                
                # We use the full dataset since for our custom fitting it does not only specific halos (?)
                r_fit = my_data["p_Scaled_radii"].compute().to_numpy()
                vr_fit = my_data["p_Radial_vel"].compute().to_numpy()
                vphys_fit = my_data["p_phys_vel"].compute().to_numpy()
                sparta_labels_fit = my_data["Orbit_infall"].compute().to_numpy()
                lnv2_fit = np.log(vphys_fit**2)
                
                halo_first = halo_df["Halo_first"].values
                halo_n = halo_df["Halo_n"].values
                all_idxs = halo_df["Halo_indices"].values
                # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
                sim_splits = np.where(halo_first == 0)[0]
            
                sparta_orb = np.where(sparta_labels_fit == 1)[0]
                sparta_inf = np.where(sparta_labels_fit == 0)[0]

                mask_vr_neg = (vr_fit < 0)
                mask_vr_pos = ~mask_vr_neg
                mask_r = r_fit < r_cut
                
                
            sparta_name, sparta_search_name = split_calc_name(curr_test_sims[0])
            curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"      
            
            sparta_output = sparta.load(filename=curr_sparta_HDF5_path, log_level=0)
        
            bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
            bins = np.insert(bins, 0, 0)
            
            vr_pos = opt_func(bins, r_fit[mask_vr_pos], lnv2_fit[mask_vr_pos], sparta_labels_fit[mask_vr_pos], ps_param_dict["b_pos"], plot_loc = plot_loc, title = "pos")
            vr_neg = opt_func(bins, r_fit[mask_vr_neg], lnv2_fit[mask_vr_neg], sparta_labels_fit[mask_vr_neg], ps_param_dict["b_neg"], plot_loc = plot_loc, title = "neg")
        
            opt_param_dict = {
                "orb_vr_pos": vr_pos,
                "orb_vr_neg": vr_neg,
                "inf_vr_neg": vr_neg,
                "inf_vr_pos": vr_pos,
            }
        
            save_pickle(opt_param_dict,model_fldr_loc+"bin_fit_ps_cut_params.pickle")
            
            # if the testing simulations are the same as the model simulations we don't need to reload the data
            if sorted(curr_test_sims) == sorted(model_sims):
                print("Using fitting simulations for testing")
                r_test = r_fit
                vr_test = vr_fit
                vphys_test = vphys_fit
                sparta_labels_test = sparta_labels_fit
                lnv2_test = lnv2_fit
            else:
                with timed("Loading Testing Data"):
                    r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ps_data(client,curr_test_sims=curr_test_sims)
                    r_test = my_data["p_Scaled_radii"].compute().to_numpy()
                    vr_test = my_data["p_Radial_vel"].compute().to_numpy()
                    vphys_test = my_data["p_phys_vel"].compute().to_numpy()
                    sparta_labels_test = my_data["Orbit_infall"].compute().to_numpy()
                    lnv2_test = np.log(vphys_test**2)
                    
            halo_first = halo_df["Halo_first"].values
            halo_n = halo_df["Halo_n"].values
            all_idxs = halo_df["Halo_indices"].values
            # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
            sim_splits = np.where(halo_first == 0)[0]
        
            sparta_orb = np.where(sparta_labels_test == 1)[0]
            sparta_inf = np.where(sparta_labels_test == 0)[0]     
    
    mask_vr_neg = (vr_test < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r_test < r_cut
        
    fltr_combs = {
        "orb_vr_neg": np.intersect1d(sparta_orb, np.where(mask_vr_neg)[0]),
        "orb_vr_pos": np.intersect1d(sparta_orb, np.where(mask_vr_pos)[0]),
        "inf_vr_neg": np.intersect1d(sparta_inf, np.where(mask_vr_neg)[0]),
        "inf_vr_pos": np.intersect1d(sparta_inf, np.where(mask_vr_pos)[0]),
    } 
    
    act_mass_prf_all, act_mass_prf_orb, all_masses, bins = load_sparta_mass_prf(sim_splits,all_idxs,curr_test_sims)
    act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb 
    
    plt_SPARTA_KE_dist(ps_param_dict, fltr_combs, bins, r_test, lnv2_test, perc = perc, width = width, r_cut = r_cut, plot_loc = plot_loc, title = "bin_fit_", cust_line_dict = opt_param_dict)

#######################################################################################################################################    
    all_z = []
    all_rhom = []
    with timed("Density Profile Comparison"):
        # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
        # stacked simulations such that they correspond to the larger dataset and not one specific simulation
        if len(curr_test_sims) > 1:
            for i,sim in enumerate(curr_test_sims):
                # The first sim remains the same
                if i == 0:
                    continue
                # Else if it isn't the final sim 
                elif i < len(curr_test_sims) - 1:
                    halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
                # Else if the final sim
                else:
                    halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
        # Get the redshifts for each simulation's primary snapshot
        for i,sim in enumerate(curr_test_sims):
            with open(ML_dset_path + sim + "/config.pickle", "rb") as file:
                config_dict = pickle.load(file)
                curr_z = config_dict["p_snap_info"]["red_shift"][()]
                all_z.append(curr_z)
                all_rhom.append(cosmol.rho_m(curr_z))
                h = config_dict["p_snap_info"]["h"][()]

        tot_num_halos = halo_n.shape[0]
        min_disp_halos = int(np.ceil(0.3 * tot_num_halos))
        
        bin_indices = np.digitize(r_test, bins) - 1  
        preds = np.zeros(r_test.shape[0])
        for i in range(bins.shape[0]-1):
            if bins[i] <= 2.0:
                mask_pos = (bin_indices == i) & (vr_test > 0) & (lnv2_test <= opt_param_dict["inf_vr_pos"]["b"][i])
                mask_neg = (bin_indices == i) & (vr_test < 0) & (lnv2_test <= opt_param_dict["inf_vr_neg"]["b"][i])
            
                preds[mask_pos] = 1
                preds[mask_neg] = 1

        calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m = create_stack_mass_prf(sim_splits,radii=r_test, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds, prf_bins=bins, use_mp=True, all_z=all_z)

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
                
        for sim in curr_test_sims:
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
        
        
        compare_split_prfs(plt_nu_splits,len(cpy_plt_nu_splits),all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_loc,title= "fit_ps_cut_dens_",prf_name_0="Optimized Cut", prf_name_1="SPARTA")
        