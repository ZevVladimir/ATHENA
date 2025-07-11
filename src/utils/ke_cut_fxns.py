import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import argparse
import dask.array as da

from .ML_fxns import split_sparta_hdf5_name
from .util_fxns import load_pickle, load_SPARTA_data, load_config, load_ML_dsets
from .util_fxns import reform_dset_dfs, parse_ranges, depair_np, timed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default=os.getcwd() + "/config.ini", 
    help='Path to config file (default: config.ini)'
)

args = parser.parse_args()
config_params = load_config(args.config)

ML_dset_path = config_params["PATHS"]["ml_dset_path"]
SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]
rockstar_ctlgs_path = config_params["PATHS"]["rockstar_ctlgs_path"]

test_sims = config_params["EVAL_MODEL"]["test_sims"]

plt_nu_splits = config_params["EVAL_MODEL"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)

plt_macc_splits = config_params["EVAL_MODEL"]["plt_macc_splits"]
plt_macc_splits = parse_ranges(plt_macc_splits)

linthrsh = config_params["EVAL_MODEL"]["linthrsh"]
lin_nbin = config_params["EVAL_MODEL"]["lin_nbin"]
log_nbin = config_params["EVAL_MODEL"]["log_nbin"]
lin_rvticks = config_params["EVAL_MODEL"]["lin_rvticks"]
log_rvticks = config_params["EVAL_MODEL"]["log_rvticks"]
lin_tvticks = config_params["EVAL_MODEL"]["lin_tvticks"]
log_tvticks = config_params["EVAL_MODEL"]["log_tvticks"]
lin_rticks = config_params["EVAL_MODEL"]["lin_rticks"]
log_rticks = config_params["EVAL_MODEL"]["log_rticks"]

#TODO do some extra checking that this is right especially with different datasets and loading the right number of particles
def halo_select(sims, ptl_data, halo_df, dset_name):
    curr_halo_start = 0
    all_row_idxs = []
    
    sim_splits = np.where(halo_df["Halo_first"].values == 0)[0]
    
    # For each simulation we will search through the largest halos and determine if they dominate their region and only choose those ones
    for i,sim in enumerate(sims):
        curr_halo_df = halo_df.iloc[sim_splits[i]:sim_splits[i+1]] if i < len(sim_splits)-1 else halo_df[sim_splits[i]:]
        
        # Get the numnber of particles per halo and use this to sort the halos such that the largest is first
        n_ptls = load_pickle(ML_dset_path + sim + "/num_ptls.pickle")   
        
        # Load information about the simulation
        dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
        p_box_size = dset_params["all_snap_info"]["prime_snap_info"]["box_size"][()]
        p_scale_factor = dset_params["all_snap_info"]["prime_snap_info"]["scale_factor"][()]
        
        sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
        
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"
        
        dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
        p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"][()]

        # Load the halo's positions and radii
        param_paths = [["halos","position"],["halos","R200m"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name)

        halos_pos = sparta_params[sparta_param_names[0]][:,p_sparta_snap,:] * 10**3 * p_scale_factor # convert to kpc/h physical
        halos_r200m = sparta_params[sparta_param_names[1]][:,p_sparta_snap]

        
        curr_idxs = curr_halo_df["Halo_indices"].values
        halo_first = curr_halo_df["Halo_first"].values
        halo_n = curr_halo_df["Halo_n"].values
        
        curr_halo_start = curr_halo_start + np.sum(halo_n) # always increment even if this halo isn't going to be counted
        
        max_nhalo = 500
        
        order_halo = np.argsort(halo_n)[::-1]

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
    
    print(ptl_data.shape,halo_df["Halo_n"].sum(),np.max(halo_df["Halo_first"].values),np.max(all_row_idxs))
    subset_df = ptl_data.iloc[all_row_idxs]        
    return subset_df

def load_ke_data(curr_sims, sim_cosmol_list, snap_list, dset_name):
    with timed("Loading data"):                     
        # Load the particle information
        data,scale_pos_weight,halo_df = load_ML_dsets(curr_sims,dset_name,sim_cosmol_list)

        samp_data = halo_select(curr_sims,data,halo_df,dset_name)
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

def dask_opt_ke_predictor(opt_param_dict, bins, r_r200m, vr, lnv2, r200m_cut):
    r_r200m = r_r200m.to_dask_array(lengths=True)
    vr = vr.to_dask_array(lengths=True)
    lnv2 = lnv2.to_dask_array(lengths=True)
    
    
    bin_indices = da.digitize(r_r200m, bins) - 1  
    preds_fit_ke = da.zeros_like(r_r200m,dtype=da.int8)
    for i in range(bins.shape[0]-1):
        if bins[i] <= r200m_cut:
            b_pos = opt_param_dict["inf_vr_pos"]["b"][i]
            b_neg = opt_param_dict["inf_vr_neg"]["b"][i]

            mask_bin = bin_indices == i
            mask_pos = mask_bin & (vr > 0) & (lnv2 <= b_pos)
            mask_neg = mask_bin & (vr < 0) & (lnv2 <= b_neg)

            # da.where to lazily update
            preds_fit_ke = da.where(mask_pos | mask_neg, 1, preds_fit_ke)

    return preds_fit_ke