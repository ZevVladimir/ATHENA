import numpy as np
import os
import sys
import pickle
from dask import array as da
import xgboost as xgb
from xgboost import dask as dxgb
import json
import time
import h5py
import re
from contextlib import contextmanager
from utils.data_and_loading_functions import load_or_pickle_SPARTA_data, find_closest_z, conv_halo_id_spid, create_directory
from utils.visualization_functions import compare_density_prf, plot_r_rv_tv_graph, plot_misclassified
from sparta_tools import sparta
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
rand_seed = config.getint("MISC","random_seed")
on_zaratan = config.getboolean("MISC","on_zaratan")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
path_to_hdf5_file = path_to_SPARTA_data + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
path_to_xgboost = config["PATHS"]["path_to_xgboost"]
model_sims = json.loads(config.get("DATASET","model_sims"))

global p_red_shift
p_red_shift = config.getfloat("SEARCH","p_red_shift")
search_rad = config.getfloat("SEARCH","search_rad")
model_type = config["XGBOOST"]["model_type"]
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
# size float32 is 4 bytes
chunk_size = int(np.floor(1e9 / (num_save_ptl_params * 4)))

@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))

def make_preds(client, bst, X_np, y_np, report_name="Classification Report", print_report=False):
    X = da.from_array(X_np,chunks=(chunk_size,X_np.shape[1]))
    
    preds = dxgb.inplace_predict(client, bst, X).compute()
    preds = np.round(preds)
    preds = preds.astype(np.int8)
    
    return preds

def print_model_prop(model_dict, indent=''):
    # If using from command line and passing path to the pickled dictionary instead of dict load the dict from the file path
    if isinstance(model_dict, str):
        with open(model_dict, "rb") as file:
            model_dict = pickle.load(file)
        
    for key, value in model_dict.items():
        # use recursion for dictionaries within the dictionary
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            print_model_prop(value, indent + '  ')
        # if the value is instead a list join them all together with commas
        elif isinstance(value, list):
            print(f"{indent}{key}: {', '.join(map(str, value))}")
        else:
            print(f"{indent}{key}: {value}")
            
def create_dmatrix(client, X_cpu, y_cpu, features, chunk_size, frac_use_data = 1, calc_scale_pos_weight = False):    
    scale_pos_weight = np.where(y_cpu == 0)[0].size / np.where(y_cpu == 1)[0].size
    
    num_features = X_cpu.shape[1]
    
    num_use_data = int(np.floor(X_cpu.shape[0] * frac_use_data))
    print("Tot num of particles:", X_cpu.shape[0], "Num use particles:", num_use_data)
    X = da.from_array(X_cpu,chunks=(chunk_size,num_features))
    y = da.from_array(y_cpu,chunks=(chunk_size))
        
    print("X Number of total bytes:", X.nbytes, "X Number of Gigabytes:", (X.nbytes)/(10**9))
    print("y Number of total bytes:", y.nbytes, "y Number of Gigabytes:", (y.nbytes)/(10**9))
    
    dqmatrix = xgb.dask.DaskDMatrix(client, X, y, feature_names=features)
    
    if calc_scale_pos_weight:
        return dqmatrix, X, y_cpu, scale_pos_weight 
    return dqmatrix, X, y_cpu

#TODO have the locations all be in a dictionary or some more general way
def eval_model(model_info, client, model, use_sims, dst_type, dst_loc, combined_name, plot_save_loc, p_red_shift, dens_prf = False, r_rv_tv = False, misclass=False): 
    with timed(dst_type + " Predictions"):
        with open(dst_loc + dst_type.lower() + "_dataset/dataset.pickle", "rb") as file:
            X = pickle.load(file)
        with open(dst_loc + dst_type.lower() + "_dataset/labels.pickle", "rb") as file:
            y = pickle.load(file)
        preds = make_preds(client, model, X, y, report_name=dst_type + " Report", print_report=False)
        
    num_bins = 30
    with open(dst_loc + "full_dataset/keys.pickle", "rb") as file:
        all_keys = pickle.load(file)
    p_r_loc = np.where(all_keys == "p_Scaled_radii_")[0][0]
    c_r_loc = np.where(all_keys == "c_Scaled_radii_")[0][0]
    p_rv_loc = np.where(all_keys == "p_Radial_vel_")[0][0]
    c_rv_loc = np.where(all_keys == "c_Radial_vel_")[0][0]
    p_tv_loc = np.where(all_keys == "p_Tangential_vel_")[0][0]
    c_tv_loc = np.where(all_keys == "c_Tangential_vel_")[0][0]
    
    if dens_prf:
        with open(dst_loc + dst_type.lower() + "_dataset/halo_first.pickle", "rb") as file:
            halo_first = pickle.load(file) 
        with open(dst_loc + dst_type.lower() + "_dataset/halo_n.pickle", "rb") as file:
            halo_n = pickle.load(file)   
        with open(dst_loc + dst_type.lower() + "_dataset/halo_indices.pickle", "rb") as file:
            all_idxs = pickle.load(file)
        
        # Know where each simulation's data starts in the stacekd dataset based on when the indexing starts from 0 again
        sim_splits = np.where(halo_first == 0)[0]

        for i,sim in enumerate(use_sims):
            # Get the halo indices corresponding to this simulation
            if i < len(use_sims) - 1:
                use_idxs = all_idxs[sim_splits[i]:sim_splits[i+1]]
            else:
                use_idxs = all_idxs[sim_splits[i]:]
        
             # find the snapshots for this simulation
            snap_pat = r"(\d+)to(\d+)"
            match = re.search(snap_pat, sim)
            if match:
                curr_snap_list = [match.group(1), match.group(2)] 
            sim_pat = r"cbol_l(\d+)_n(\d+)"
            match = re.search(sim_pat, sim)
            if match:
                sparta_name = match.group(0)
            sim_search_pat = sim_pat + r"_(\d+)r200m"
            match = re.search(sim_search_pat, sim)
            if match:
                sparta_search_name = match.group(0)
            
            #TODO make this simulation dependent
            new_p_snap, p_red_shift = find_closest_z(p_red_shift)
            p_scale_factor = 1/(1+p_red_shift)
            with h5py.File(path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5","r") as f:
                dic_sim = {}
                grp_sim = f['simulation']
                for f in grp_sim.attrs:
                    dic_sim[f] = grp_sim.attrs[f]
            
            all_red_shifts = dic_sim['snap_z']
            p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
            halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, parent_id, ptl_mass = load_or_pickle_SPARTA_data(sparta_name, p_scale_factor, curr_snap_list[0], p_sparta_snap)

            use_halo_ids = halos_id[use_idxs]
            sparta_output = sparta.load(filename=path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5", halo_ids=use_halo_ids, log_level=0)
        
            new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta resort the indices
            dens_prf_all = sparta_output['anl_prf']['M_all'][new_idxs,p_sparta_snap,:]
            dens_prf_1halo = sparta_output['anl_prf']['M_1halo'][new_idxs,p_sparta_snap,:]
        
        bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
        bins = np.insert(bins, 0, 0)
         
        use_halo_first = halo_first
        # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
        # stacked simulations such that they correspond to the larger dataset and not one specific simulation
        if sim_splits.size != 1:
            for i in range(sim_splits.size):
                if i < sim_splits.size - 1:
                    use_halo_first[sim_splits[i]:sim_splits[i+1]] += use_halo_first[sim_splits[i]-1]
                else:
                    use_halo_first[sim_splits[i]:] += use_halo_first[sim_splits[i]-1]
                
        compare_density_prf(radii=X[:,p_r_loc], halo_first=use_halo_first, halo_n=halo_n, act_mass_prf_all=dens_prf_all, act_mass_prf_orb=dens_prf_1halo, mass=ptl_mass, orbit_assn=preds, prf_bins=bins, title="", save_location=plot_save_loc, use_mp=True, save_graph=True)
    
    if r_rv_tv:
        plot_r_rv_tv_graph(preds, X[:,p_r_loc], X[:,p_rv_loc], X[:,p_tv_loc], y, title="", num_bins=num_bins, save_location=plot_save_loc)
    
    if misclass:
        plot_misclassified(p_corr_labels=y, p_ml_labels=preds, p_r=X[:,p_r_loc], p_rv=X[:,p_rv_loc], p_tv=X[:,p_tv_loc], c_r=X[:,c_r_loc], c_rv=X[:,c_rv_loc], c_tv=X[:,c_tv_loc],title="",num_bins=num_bins,save_location=plot_save_loc,model_info=model_info,dataset_name=dst_type + "_" + combined_name)
    

