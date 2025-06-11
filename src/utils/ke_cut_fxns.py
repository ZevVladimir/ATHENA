import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

from .ML_fxns import split_sparta_hdf5_name
from .util_fxns import load_pickle, load_SPARTA_data, load_config, load_ML_dsets
from .util_fxns import reform_dset_dfs, parse_ranges, depair_np, timed

config_dict = load_config(os.getcwd() + "/config.ini")

ML_dset_path = config_dict["PATHS"]["ml_dset_path"]
SPARTA_output_path = config_dict["SPARTA_DATA"]["sparta_output_path"]
rockstar_ctlgs_path = config_dict["PATHS"]["rockstar_ctlgs_path"]

test_sims = config_dict["EVAL_MODEL"]["test_sims"]
eval_datasets = config_dict["EVAL_MODEL"]["eval_datasets"]

plt_nu_splits = config_dict["EVAL_MODEL"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)

plt_macc_splits = config_dict["EVAL_MODEL"]["plt_macc_splits"]
plt_macc_splits = parse_ranges(plt_macc_splits)

linthrsh = config_dict["EVAL_MODEL"]["linthrsh"]
lin_nbin = config_dict["EVAL_MODEL"]["lin_nbin"]
log_nbin = config_dict["EVAL_MODEL"]["log_nbin"]
lin_rvticks = config_dict["EVAL_MODEL"]["lin_rvticks"]
log_rvticks = config_dict["EVAL_MODEL"]["log_rvticks"]
lin_tvticks = config_dict["EVAL_MODEL"]["lin_tvticks"]
log_tvticks = config_dict["EVAL_MODEL"]["log_tvticks"]
lin_rticks = config_dict["EVAL_MODEL"]["lin_rticks"]
log_rticks = config_dict["EVAL_MODEL"]["log_rticks"]

    
def halo_select(sims, ptl_data):
    curr_tot_nptl = 0
    curr_halo_start = 0
    all_row_idxs = []
    hipids = ptl_data["HIPIDS"].compute().to_numpy()
    all_pids, ptl_halo_idxs = depair_np(hipids)
    # For each simulation we will search through the largest halos and determine if they dominate their region and only choose those ones
    for i,sim in enumerate(sims):
        # Get the numnber of particles per halo and use this to sort the halos such that the largest is first
        n_ptls = load_pickle(ML_dset_path + sim + "/num_ptls.pickle")   
        match_halo_idxs = load_pickle(ML_dset_path + sim + "/match_halo_idxs.pickle")    
  
        total_num_halos = match_halo_idxs.shape[0]
        
        # Load information about the simulation
        dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
        test_halos_ratio = dset_params["test_halos_ratio"]
        p_box_size = dset_params["all_snap_info"]["prime_snap_info"]["box_size"][()]
        p_scale_factor = dset_params["all_snap_info"]["prime_snap_info"]["scale_factor"][()]
        
        # split all indices into train and test groups
        split_pnt = int((1-test_halos_ratio) * total_num_halos)
        test_idxs = match_halo_idxs[split_pnt:]
        test_num_ptls = n_ptls[split_pnt:]

        # need to sort indices otherwise sparta.load breaks...      
        test_idxs_inds = test_idxs.argsort()
        test_idxs = test_idxs[test_idxs_inds]
        test_num_ptls = test_num_ptls[test_idxs_inds]
        
        sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
        
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"
        
        dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
        p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"][()]

        # Load the halo's positions and radii
        param_paths = [["halos","position"],["halos","R200m"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name)

        halos_pos = sparta_params[sparta_param_names[0]][:,p_sparta_snap,:] * 10**3 * p_scale_factor # convert to kpc/h physical
        halos_r200m = sparta_params[sparta_param_names[1]][:,p_sparta_snap]
        
        
        halo_ddf = reform_dset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/")
        curr_idxs = halo_ddf["Halo_indices"].values
        halo_first = halo_ddf["Halo_first"].values
        halo_n = halo_ddf["Halo_n"].values

        curr_halo_start = curr_halo_start + np.sum(halo_n) # always increment even if this halo isn't going to be counted
        
        max_nhalo = 500
        
        order_halo = np.argsort(test_num_ptls)[::-1]
        n_ptls = test_num_ptls[order_halo]
        if order_halo.shape[0] < max_nhalo:
            max_nhalo = order_halo.shape[0]
                
        # Load which particles belong to which halo and then sort them corresponding to the size of the halos again
        curr_idxs = curr_idxs[order_halo]
        halo_n = halo_n[order_halo]
        halo_first = halo_first[order_halo]
        
        use_halo_pos = halos_pos[curr_idxs]
        use_halo_r200m = halos_r200m[curr_idxs]
        # Construct a search tree of halo positions
        curr_halo_tree = KDTree(data = use_halo_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size)
        
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
                curr_tot_nptl += np.sum(halo_n)
                
            # Once we have 10,000,000 particles we are done
            # if curr_tot_nptl > 1000000:
            #     break
    subset_df = ptl_data.compute().loc[all_row_idxs]        
    return subset_df

def load_ke_data(client, curr_test_sims, sim_cosmol_list, snap_list):
    # Loop through and/or for Train/Test/All datasets and evaluate the model
    dset_name = eval_datasets[0]

    with timed("Loading data"):             
        # Load the halo information
        halo_dfs = []
        if dset_name == "Full":    
            for sim in curr_test_sims:
                halo_dfs.append(reform_dset_dfs(ML_dset_path + sim + "/" + "Train" + "/halo_info/"))
                halo_dfs.append(reform_dset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/"))
        else:
            for sim in curr_test_sims:
                halo_dfs.append(reform_dset_dfs(ML_dset_path + sim + "/" + dset_name + "/halo_info/"))

        halo_df = pd.concat(halo_dfs)
        
        # Load the particle information
        data,scale_pos_weight = load_ML_dsets(client,curr_test_sims,dset_name,sim_cosmol_list,prime_snap=snap_list[0])
        samp_data = halo_select(curr_test_sims,data)
    r = samp_data["p_Scaled_radii"]
    vr = samp_data["p_Radial_vel"]
    vphys = samp_data["p_phys_vel"]
    lnv2 = np.log(vphys**2)
    sparta_labels = samp_data["Orbit_infall"]
    
    return r, vr, lnv2, sparta_labels, samp_data, data, halo_df

# r200m_cut is the farthest radius (r/r200m) that a particle can be orbiting
def fast_ke_predictor(fast_param_dict, r_r200m, vr, lnv2, r200m_cut):
    m_pos = fast_param_dict["m_pos"]
    b_pos = fast_param_dict["b_pos"]
    m_neg = fast_param_dict["m_neg"]
    b_neg = fast_param_dict["b_neg"]
    
    preds = np.zeros(r_r200m.shape[0],dtype=np.int8)
    
    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg

    mask_cut_pos = (lnv2 < (m_pos * r_r200m + b_pos)) & (r_r200m < r200m_cut)

    # Orbiting classification for vr < 0
    mask_cut_neg = (lnv2 < (m_neg * r_r200m + b_neg)) & (r_r200m < r200m_cut)

    # Particle is infalling if it is below both lines and 2*R00b
    mask_orb = \
    (mask_cut_pos & mask_vr_pos) ^ \
    (mask_cut_neg & mask_vr_neg)

    preds[mask_orb] = 1

    return mask_orb, preds

def opt_ke_predictor(opt_param_dict, bins, r_r200m, vr, lnv2, r200m_cut):
    bin_indices = np.digitize(r_r200m, bins) - 1  
    preds_fit_ke = np.zeros(r_r200m.shape[0],dtype=np.int8)
    for i in range(bins.shape[0]-1):
        if bins[i] <= r200m_cut:
            mask_pos = (bin_indices == i) & (vr > 0) & (lnv2 <= opt_param_dict["inf_vr_pos"]["b"][i])
            mask_neg = (bin_indices == i) & (vr < 0) & (lnv2 <= opt_param_dict["inf_vr_neg"]["b"][i])
        
            preds_fit_ke[mask_pos] = 1
            preds_fit_ke[mask_neg] = 1
    return preds_fit_ke
