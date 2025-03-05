import matplotlib.pyplot as plt
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
from colossus.cosmology import cosmology
from scipy.spatial import cKDTree
import scipy.ndimage as ndimage
from sparta_tools import sparta

from utils.calculation_functions import create_stack_mass_prf, filter_prf, calculate_density, calc_mass_acc_rate
from utils.update_vis_fxns import compare_split_prfs, plot_full_ptl_dist, plot_prim_ptl_dist
from utils.ML_support import load_data, get_CUDA_cluster, get_combined_name, reform_dataset_dfs, parse_ranges, create_nu_string, load_sparta_mass_prf, split_calc_name, sim_mass_p_z
from utils.data_and_loading_functions import create_directory, timed, load_pickle, load_SPARTA_data, conv_halo_id_spid, load_ps_data

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

def load_ps_data(client, curr_test_sims = ["cbol_l1000_n1024_4r200m_1-5v200m_99to90"]):
    test_comb_name = get_combined_name(curr_test_sims) 

    # Loop through and/or for Train/Test/All datasets and evaluate the model
    dset_name = eval_datasets[0]

    with timed("Loading data"):             
        
        
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

def ps_predictor(feat_dict, r, vr, lnv2):
    m_pos = feat_dict["m_pos"]
    b_pos = feat_dict["b_pos"]
    m_neg = feat_dict["m_neg"]
    b_neg = feat_dict["b_neg"]
    
    preds = np.zeros(r.shape[0].compute())
    
    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg

    mask_cut_pos = (lnv2 < (m_pos * r + b_pos)) & (r < 3.0)

    # Orbiting classification for vr < 0
    mask_cut_neg = (lnv2 < (m_neg * r + b_neg)) & (r < 3.0)

    # Particle is infalling if it is below both lines and 2*R00
    mask_orb = \
    (mask_cut_pos & mask_vr_pos) ^ \
    (mask_cut_neg & mask_vr_neg)

    preds[mask_orb] = 1

    return preds
