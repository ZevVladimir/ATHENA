import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np
from matplotlib.cm import get_cmap
from scipy.optimize import curve_fit, minimize

plt.rcParams.update({"text.usetex":True, "font.family": "serif", "figure.dpi": 150})
cmap = get_cmap('terrain')
import os
import multiprocessing as mp
from dask.distributed import Client
import json
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import pickle
import h5py
from colossus.cosmology import cosmology
from scipy.spatial import cKDTree
import scipy.ndimage as ndimage
from sparta_tools import sparta

from utils.calculation_functions import create_stack_mass_prf, filter_prf, calculate_density, calc_mass_acc_rate
from utils.update_vis_fxns import compare_split_prfs, plot_full_ptl_dist, plot_prim_ptl_dist
from utils.ML_support import load_data, get_CUDA_cluster, get_combined_name, reform_dataset_dfs, parse_ranges, create_nu_string, load_sparta_mass_prf, split_calc_name, sim_mass_p_z
from utils.data_and_loading_functions import create_directory, timed, load_pickle, load_SPARTA_data, conv_halo_id_spid

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

####################################################################################################################################################################################################################################
    
def halo_select(sims, ptl_data):
    curr_tot_nptl = 0
    all_row_idxs = []
    # For each simulation we will search through the largest halos and determine if they dominate their region and only choose those ones
    for sim in sims:
        # Get the numnber of particles per halo and use this to sort the halos such that the largest is first
        n_ptls = load_pickle(ML_dset_path + sim + "/num_ptls.pickle")   
        match_halo_idxs = load_pickle(ML_dset_path + sim + "/match_halo_idxs.pickle")    
  
        total_num_halos = match_halo_idxs.shape[0]
        
        # Load information about the simulation
        config_dict = load_pickle(ML_dset_path + sim + "/config.pickle")
        test_halos_ratio = config_dict["test_halos_ratio"]
        curr_z = config_dict["p_snap_info"]["red_shift"][()]
        p_snap = config_dict["p_snap_info"]["ptl_snap"][()]
        p_box_size = config_dict["p_snap_info"]["box_size"][()]
        p_scale_factor = config_dict["p_snap_info"]["scale_factor"][()]
        
        # ptl_mass, use_z = sim_mass_p_z(sim,config_dict)
        
        # split all indices into train and test groups
        split_pnt = int((1-test_halos_ratio) * total_num_halos)
        test_idxs = match_halo_idxs[split_pnt:]
        test_num_ptls = n_ptls[split_pnt:]

        # need to sort indices otherwise sparta.load breaks...      
        test_idxs_inds = test_idxs.argsort()
        test_idxs = test_idxs[test_idxs_inds]
        test_num_ptls = test_num_ptls[test_idxs_inds]
        
        order_halo = np.argsort(test_num_ptls)[::-1]
        n_ptls = test_num_ptls[order_halo]
        
        # Load which particles belong to which halo and then sort them corresponding to the size of the halos again
        halo_ddf = reform_dataset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/")
        all_idxs = halo_ddf["Halo_indices"].values
        halo_n = halo_ddf["Halo_n"].values
        halo_first = halo_ddf["Halo_first"].values
        
        all_idxs = all_idxs[order_halo]
        halo_n = halo_n[order_halo]
        halo_first = halo_first[order_halo]
        
        sparta_name, sparta_search_name = split_calc_name(sim)
        
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"
        
        config_dict = load_pickle(ML_dset_path + sim + "/config.pickle")
        p_sparta_snap = config_dict["p_snap_info"]["sparta_snap"][()]

        # Load the halo's positions and radii
        param_paths = [["halos","position"],["halos","R200m"],["halos","id"],["halos","status"],["halos","last_snap"],["simulation","particle_mass"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, p_snap)

        halos_pos = sparta_params[sparta_param_names[0]][:,p_sparta_snap,:] * 10**3 * p_scale_factor # convert to kpc/h
        halos_r200m = sparta_params[sparta_param_names[1]][:,p_sparta_snap]
        
        use_halo_pos = halos_pos[all_idxs]
        use_halo_r200m = halos_r200m[all_idxs]
        
        # Construct a search tree of halo positions
        curr_halo_tree = cKDTree(data = use_halo_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size)
        
        max_nhalo = 500
        if order_halo.shape[0] < max_nhalo:
            max_nhalo = order_halo.shape[0]
        
        # For each massive halo search the area around the halo to determine if there are any halos that are more than 20% the size of this halo
        for i in range(max_nhalo):
            curr_halo_indices = curr_halo_tree.query_ball_point(use_halo_pos[i], r = 2 * use_halo_r200m[i])
            curr_halo_indices = np.array(curr_halo_indices)
            
            # The search will return the same halo that we are searching so we remove that
            surr_n_ptls = n_ptls[curr_halo_indices] 
            surr_n_ptls = surr_n_ptls[surr_n_ptls != n_ptls[i]]
            
            # If the largest halo nearby isn't large enough we will consider this halo's particles for our datasets
            if surr_n_ptls.size == 0 or np.max(surr_n_ptls) < 0.2 * n_ptls[i]:
                row_indices = list(range(
                    halo_first[i],
                    halo_first[i] + halo_n[i]
                ))
                all_row_idxs.extend(row_indices)
                curr_tot_nptl += halo_n[i]
                
            # Once we have 10,000,000 particles we are done
            # if curr_tot_nptl > 1000000:
            #     break
    subset_df = ptl_data.compute().loc[all_row_idxs]        
    return subset_df
    
def load_your_data(curr_test_sims = ["cbol_l1000_n1024_4r200m_1-5v200m_99to90"]):
    test_comb_name = get_combined_name(curr_test_sims) 
        
    print("Testing on:", curr_test_sims)
    # Loop through and/or for Train/Test/All datasets and evaluate the model
    dset_name = eval_datasets[0]

    with timed("Loading data"):             
        plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
        create_directory(plot_loc)
        
        # Load the halo information
        halo_files = []
        halo_dfs = []
        if dset_name == "Full":    
            for sim in curr_test_sims:
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + "Train" + "/halo_info/"))
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/"))
        else:
            for sim in curr_test_sims:
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + dset_name + "/halo_info/"))

        halo_df = pd.concat(halo_dfs)
        
        # Load the particle information
        data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)
        nptl = data.shape[0].compute()
        samp_data = halo_select(curr_test_sims,data)
    r = samp_data["p_Scaled_radii"]
    vr = samp_data["p_Radial_vel"]
    vphys = samp_data["p_phys_vel"]
    lnv2 = np.log(vphys**2)
    sparta_labels = samp_data["Orbit_infall"]
    
    return r, vr, lnv2, sparta_labels, data, halo_df

def gradient_minima(
    r: np.ndarray,
    lnv2: np.ndarray,
    mask_vr: np.ndarray,
    n_points: int,
    r_min: float,
    r_max: float,
):
    """Computes the r-lnv2 gradient and finds the minimum as a function of `r`
    within the interval `[r_min, r_max]`

    Parameters
    ----------
    r : np.ndarray
        Radial separation
    lnv2 : np.ndarray
        Log-kinetic energy
    mask_vr : np.ndarray
        Mask for the selection of radial velocity
    n_points : int
        Number of minima points to compute
    r_min : float
        Minimum radial distance
    r_max : float
        Maximum radial distance

    Returns
    -------
    Tuple[np.ndarray]
        Radial and minima coordinates.
    """
    r_edges_grad = np.linspace(r_min, r_max, n_points + 1)
    grad_r = 0.5 * (r_edges_grad[:-1] + r_edges_grad[1:])
    grad_min = np.zeros(n_points)
    for i in range(n_points):
        r_mask = (r > r_edges_grad[i]) * (r < r_edges_grad[i + 1])
        hist_yv, hist_edges = np.histogram(lnv2[mask_vr * r_mask], bins=200)
        hist_lnv2 = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        hist_lnv2_grad = np.gradient(hist_yv, np.mean(np.diff(hist_edges)))
        lnv2_mask = (1.0 < hist_lnv2) * (hist_lnv2 < 1.75)
        grad_min[i] = hist_lnv2[lnv2_mask][np.argmin(hist_lnv2_grad[lnv2_mask])]

    return grad_r, grad_min

def cost_percentile(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the `target` percentile of paricles is below the line.

    Parameters
    ----------
    b : float
        Fit parameter
    *data : tuple
        A tuple with `[r, lnv2, slope, target]`, where `slope` is the slope of 
        the line and is fixed, and `target` is the desired percentile

    Returns
    -------
    float
    """
    r, lnv2, slope, target = data
    below_line = (lnv2 < (slope * r + b)).sum()
    return np.log((target - below_line / r.shape[0]) ** 2)

def cost_perp_distance(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the perpendicular distance of all points to the line is maximal.

    Parameters
    ----------
    b : float
        Fit parameter
    *data: tuple
        A tuple with `[r, lnv2, slope, width]`, where `slope` is the slope of 
        the line and is fixed, and `width` is the width of a band around the 
        line within which the distance is computed

    Returns
    -------
    float
        _description_
    """
    r, lnv2, slope, width = data
    d = np.abs(lnv2 - slope * r - b) / np.sqrt(1 + slope**2)
    return -np.log(np.mean(d[(d < width)] ** 2))

def calibrate_finder(
    n_points: int = 20,
    perc: float = 0.99,
    width: float = 0.05,
    grad_lims: tuple = (0.2, 0.5),
):
    """_summary_

    Parameters
    ----------
    n_points : int, optional
        Number of minima points to compute, by default 20
    perc : float, optional
        Target percentile for the positive radial velocity calibration, 
        by default 0.98
    width : float, optional
        Band width for the negattive radial velocity calibration, 
        by default 0.05
    grad_lims : tuple
        Radial interval in which the gradient is to be computed, by default
        [0.2, 0.5]
    """
    # MODIFY this line if needed ======================
    r, vr, lnv2, sparta_labels, my_data, halo_df = load_your_data()

    # =================================================

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r < 1.75

    # For vr > 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_pos, n_points, *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * x + b, r_grad, min_grad, p0=[-1, 2])
    m_pos, b01 = popt

    # Find intercept by finding the value that contains 'perc' percent of 
    # particles below the line at fixed slope 'm_pos'
    res = minimize(
        cost_percentile,
        1.1 * b01,
        bounds=((0.8 * b01, 3.0),),
        args=(r[mask_vr_pos * mask_r], lnv2[mask_vr_pos * mask_r], m_pos, perc),
        method='Nelder-Mead',
    )
    b_pos = res.x[0]

    # For vr < 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_neg, n_points, *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * x + b, r_grad, min_grad, p0=[-1, 2])
    m_neg, b02 = popt

    # Find intercept by finding the value that maximizes the perpendicular
    # distance to the line at fixed slope of all points within a perpendicular
    # 'width' distance from the line (ignoring all others).
    res = minimize(
        cost_perp_distance,
        0.75 * b02,
        bounds=((1.2, b02),),
        args=(r[mask_vr_neg], lnv2[mask_vr_neg], m_neg, width),
        method='Nelder-Mead',
    )
    b_neg = res.x[0]

    return (m_pos, b_pos), (m_neg, b_neg)
    
    
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
    
    ####################################################################################################################################################################################################################################
    
    (m_pos, b_pos), (m_neg, b_neg) = calibrate_finder()
          
    # m_pos = -1.9973747688461672
    # b_pos = 2.730691113802748
    # m_neg = -1.601325049968688
    # b_neg = 1.5101195108968333
    
    print("\nCalibration Params")
    print(m_pos,b_pos,m_neg,b_neg)
    print("\n")
    # These are the values I used for my dataset. Feel free to play around with them
    # to improve results. 'perc' is the percent of particles expected below the line
    # for vr > 0. 'width' is used for 
    width = 0.05
    perc = 0.99
    grad_lims = "0.2_0.5"
    r_cut = 1.75

    r, vr, lnv2, sparta_labels, my_data, halo_df = load_your_data(test_sims[0])
    
    # r = r.to_numpy()
    # vr = vr.to_numpy()
    # lnv2 = lnv2.to_numpy()
    # sparta_labels = sparta_labels.to_numpy()
    
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

    plot_full_ptl_dist(sparta_labels,r,vr,vt,c_r,c_vr,split_scale_dict,num_bins=30,save_loc=plot_loc,save_title="_ps_data")
    plot_prim_ptl_dist(sparta_labels[fltr_combs["inf_vr_neg"]],r[fltr_combs["inf_vr_neg"]],vr[fltr_combs["inf_vr_neg"]],vt[fltr_combs["inf_vr_neg"]],split_scale_dict,num_bins=30,save_loc=plot_loc,save_title="negvr_inf")
    
    # y_shift = w / np.cos(np.arctan(m_neg))
    x = np.linspace(0, 3, 1000)
    y12 = m_pos * x + b_pos
    y22 = m_neg * x + b_neg

    nbins = 200   
    
    x_range = (0, 3)
    y_range = (-2, 2.5)

    hist1, xedges, yedges = np.histogram2d(r[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], bins=nbins, range=(x_range, y_range))
    hist2, _, _ = np.histogram2d(r[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], bins=nbins, range=(x_range, y_range))
    hist3, _, _ = np.histogram2d(r[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], bins=nbins, range=(x_range, y_range))
    hist4, _, _ = np.histogram2d(r[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], bins=nbins, range=(x_range, y_range))

    # Combine the histograms to determine the maximum density for consistent color scaling
    combined_hist = np.maximum.reduce([hist1, hist2, hist3, hist4])
    vmax=combined_hist.max()
    
    lin_vmin = 0
    log_vmin = 1

    title_fntsize = 18
    legend_fntsize = 14

    with timed("SPARTA KE Dist plot"):
        magma_cmap = plt.get_cmap("magma")
        magma_cmap.set_under(color='black')
        magma_cmap.set_bad(color='black') 
        
        widths = [4,4,4,4,.5]
        heights = [0.15,4,4]
        fig = plt.figure(constrained_layout=True, figsize=(28,12))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

        
        fig.suptitle(
            r"Kinetic Energy Distribution of Particles Around Largest Halos at $z=0.03$""\nSimulation: Bolshoi 1000Mpc",fontsize=16)
        
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        ax4 = fig.add_subplot(gs[0,3])
        ax5 = fig.add_subplot(gs[1,0])
        ax6 = fig.add_subplot(gs[1,1])
        ax7 = fig.add_subplot(gs[1,2])
        ax8 = fig.add_subplot(gs[1,3])
        
        axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
        
        ax1.set_ylabel(r'$\ln(v^2/v_{200m}^2)$',fontsize=16)
        ax5.set_ylabel(r'$\ln(v^2/v_{200m}^2)$',fontsize=16)
        ax5.set_xlabel(r'$r/R_{200m}$',fontsize=16)
        ax6.set_xlabel(r'$r/R_{200m}$',fontsize=16)
        ax7.set_xlabel(r'$r/R_{200m}$',fontsize=16)
        ax8.set_xlabel(r'$r/R_{200m}$',fontsize=16)
        
        ax1.tick_params('x', labelbottom=False,colors="white",direction="in")
        ax2.tick_params('x', labelbottom=False,colors="white",direction="in")
        ax2.tick_params('y', labelleft=False,colors="white",direction="in")
        ax3.tick_params('x', labelbottom=False,colors="white",direction="in")
        ax3.tick_params('y', labelleft=False,colors="white",direction="in")
        ax4.tick_params('x', labelbottom=False,colors="white",direction="in")
        ax4.tick_params('y', labelleft=False,colors="white",direction="in")
        ax6.tick_params('y', labelleft=False,colors="white",direction="in")
        ax7.tick_params('y', labelleft=False,colors="white",direction="in")
        ax8.tick_params('y', labelleft=False,colors="white",direction="in")

        
        for ax in axes:
            ax.set_xlim(0, 2)
            ax.set_ylim(-2, 2.5)
            ax.text(0.25, -1.4, "Orbiting", fontsize=16, color="r",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.text(1.5, 0.7, "Infalling", fontsize=16, color="b",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.tick_params(axis='both',which='both',direction="in",labelsize=12,length=8,width=2)

        plt.sca(axes[0])
        plt.title(r'$Orbiting Particles: v_r > 0$',fontsize=title_fntsize)
        plt.hist2d(r[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], bins=nbins, vmin=lin_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="g",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[1])
        plt.title(r'$Orbiting Particles: v_r < 0$',fontsize=title_fntsize)
        plt.hist2d(r[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], bins=nbins, vmin=lin_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="g",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        plt.sca(axes[2])
        plt.title(r'Infalling Particles: $v_r > 0$',fontsize=title_fntsize)
        plt.hist2d(r[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], bins=nbins, vmin=lin_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="g",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[3])
        plt.title(r'$Infalling Particles: v_r < 0$',fontsize=title_fntsize)
        plt.hist2d(r[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], bins=nbins, vmin=lin_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="g",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        plt.sca(axes[4])
        plt.hist2d(r[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], bins=nbins, norm="log", vmin=log_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="g",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[5])
        plt.hist2d(r[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], bins=nbins, norm="log", vmin=log_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="g",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[6])
        plt.hist2d(r[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], bins=nbins, norm="log", vmin=log_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="g",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[7])
        plt.hist2d(r[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], bins=nbins, norm="log", vmin=log_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="g",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        plt.tight_layout();
        plt.savefig(plot_loc + "sparta_KE_dist_cut.png")
    
    with timed("PS KE Dist plot"):
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        axes = axes.flatten()
        fig.suptitle(
            r"Kinetic energy distribution of particles around halos at $z=0$""\nSimulation: Bolshoi 1000Mpc",fontsize=16)

        for ax in axes:
            ax.set_xlabel(r'$r/R_{200m}$',fontsize=16)
            ax.set_ylabel(r'$\ln(v^2/v_{200m}^2)$',fontsize=16)
            ax.set_xlim(0, 2)
            ax.set_ylim(-2, 2.5)
            ax.text(0.25, -1.4, "Orbiting", fontsize=16, color="r",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.text(1.5, 0.7, "Infalling", fontsize=16, color="b",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.tick_params(axis='both',which='both',direction="in",labelsize=12,length=8,width=2)

        plt.sca(axes[0])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        plt.hist2d(r[mask_vr_pos], lnv2[mask_vr_pos], bins=nbins,
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="k",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[1])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        plt.hist2d(r[mask_vr_neg], lnv2[mask_vr_neg], bins=nbins,
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="k",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        plt.sca(axes[2])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        h3 = plt.hist2d(r[mask_vr_pos], lnv2[mask_vr_pos], bins=nbins, norm="log",
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="k",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(h3[3], label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[3])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        h4 = plt.hist2d(r[mask_vr_neg], lnv2[mask_vr_neg], bins=nbins, norm="log",
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="k",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(h4[3], label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        mask_vrn = (vr < 0)
        mask_vrp = ~mask_vrn

        # Compute density and gradient.
        # For vr > 0
        hist_zp, hist_xp, hist_yp = np.histogram2d(r[mask_vrp], lnv2[mask_vrp], 
                                                    bins=nbins, 
                                                    range=((0, 3.), (-2, 2.5)),
                                                    density=True)
        # Bin centres
        hist_xp = 0.5 * (hist_xp[:-1] + hist_xp[1:])
        hist_yp = 0.5 * (hist_yp[:-1] + hist_yp[1:])
        # Bin spacing
        dx = np.mean(np.diff(hist_xp))
        dy = np.mean(np.diff(hist_yp))
        # Generate a 2D grid corresponding to the histogram
        hist_xp, hist_yp = np.meshgrid(hist_xp, hist_yp)
        # Evaluate the gradient at each radial bin
        hist_z_grad = np.zeros_like(hist_zp)
        for i in range(hist_xp.shape[0]):
            hist_z_grad[i, :] = np.gradient(hist_zp[i, :], dy)
        # Apply a gaussian filter to smooth the gradient.
        hist_zp = ndimage.gaussian_filter(hist_z_grad, 2.0)

        # Same for vr < 0
        hist_zn, hist_xn, hist_yn = np.histogram2d(r[mask_vrn], lnv2[mask_vrn],
                                                    bins=nbins,
                                                    range=((0, 3.), (-2, 2.5)),
                                                    density=True)
        hist_xn = 0.5 * (hist_xn[:-1] + hist_xn[1:])
        hist_yn = 0.5 * (hist_yn[:-1] + hist_yn[1:])
        dy = np.mean(np.diff(hist_yn))
        hist_xn, hist_yn = np.meshgrid(hist_xn, hist_yn)
        hist_z_grad = np.zeros_like(hist_zn)
        for i in range(hist_xn.shape[0]):
            hist_z_grad[i, :] = np.gradient(hist_zn[i, :], dy)
        hist_zn = ndimage.gaussian_filter(hist_z_grad, 2.0)

        #Plot the smoothed gradient
        plt.sca(axes[4])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        plt.contourf(hist_xp, hist_yp, hist_zp.T, levels=80, cmap='terrain')
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label="Smoothed Gradient Magnitude")
        plt.xlim(0, 2)
        
        # Plot the smoothed gradient
        plt.sca(axes[5])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        plt.contourf(hist_xn, hist_yn, hist_zn.T, levels=80, cmap='terrain')
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label="Smoothed Gradient Magnitude")
        plt.xlim(0, 2)
        
        

        plt.tight_layout();
        plt.savefig(plot_loc + "perc_" + str(perc) + "_" + grad_lims + "_KE_dist_cut.png")
        
    # Orbiting classification for vr > 0
    mask_cut_pos = (lnv2 < (m_pos * r + b_pos)) & (r < 3.0)

    # Orbiting classification for vr < 0
    mask_cut_neg = (lnv2 < (m_neg * r + b_neg)) & (r < 3.0)

    # Particle is infalling if it is below both lines and 3*R00
    mask_orb = \
    (mask_cut_pos & mask_vr_pos) ^ \
    (mask_cut_neg & mask_vr_neg)
    
    with timed("Phase space dist of ptls plot"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes = axes.flatten()
        fig.suptitle(f'Phase-space distribution of particles around haloes at $z=0$')

        x_range = (0, 3)
        y_range = (-2, 2)

        for ax in axes:
            ax.set_xlabel(r'$r/R_{200}$')
            ax.set_ylabel(r'$v_r/v_{200}$')

        plt.sca(axes[0])
        plt.title('Orbiting')
        plt.hist2d(r[mask_orb], vr[mask_orb], bins=nbins, range=(x_range, y_range), cmap='terrain')
        plt.colorbar(label=r'$N$ (Counts)')

        plt.sca(axes[1])
        plt.title('Infalling')
        plt.hist2d(r[~mask_orb], vr[~mask_orb], bins=nbins, range=(x_range, y_range), cmap='terrain')
        plt.colorbar(label=r'$N$ (Counts)')

        plt.tight_layout()
        plt.savefig(plot_loc + "perc_" + str(perc) + "_" + grad_lims + "_ps_dist.png")
    
    
####################################################################################################################################################################################################################################

    with timed("Density profile plot"):
        feature_columns = ["p_Scaled_radii","p_Radial_vel","p_Tangential_vel","c_Scaled_radii","c_Radial_vel","c_Tangential_vel"]
        target_column = ["Orbit_infall"]

        print("Testing on:", curr_test_sims)
        # Loop through and/or for Train/Test/All datasets and evaluate the model
        
        r = my_data["p_Scaled_radii"]
        vr = my_data["p_Radial_vel"]
        vphys = my_data["p_phys_vel"]
        lnv2 = np.log(vphys**2)
        
        mask_vr_neg = (vr < 0)
        mask_vr_pos = ~mask_vr_neg

        mask_cut_pos = (lnv2 < (m_pos * r + b_pos)) & (r < 3.0)

        # Orbiting classification for vr < 0
        mask_cut_neg = (lnv2 < (m_neg * r + b_neg)) & (r < 3.0)

        # Particle is infalling if it is below both lines and 2*R00
        mask_orb = \
        (mask_cut_pos & mask_vr_pos) ^ \
        (mask_cut_neg & mask_vr_neg)
        
        X = my_data[feature_columns]
        y = my_data[target_column]
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

        # Create mass profiles from the model's predictions
        preds = np.zeros(X.shape[0].compute())
        preds[mask_orb] = 1

        calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m = create_stack_mass_prf(sim_splits,radii=X["p_Scaled_radii"].values.compute(), halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds, prf_bins=bins, use_mp=True, all_z=all_z)

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
        compare_split_prfs(plt_nu_splits,len(cpy_plt_nu_splits),all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_loc,title= "perc_" + str(perc) + "_" + grad_lims + "_ps_cut_dens_",prf_name_0="Phase Space Cut", prf_name_1="SPARTA")
        compare_split_prfs(plt_macc_splits,len(cpy_plt_macc_splits),all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_loc,title= "perc_" + str(perc) + "_" + grad_lims + "_ps_cut_macc_dens_", split_name="\Gamma", prf_name_0="Phase Space Cut", prf_name_1="SPARTA")