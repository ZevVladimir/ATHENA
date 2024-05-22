import numpy as np
import os
import sys
import pickle
from dask import array as da
import xgboost as xgb
import json
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

model_snapshot_list = json.loads(config.get("XGBOOST","model_snaps"))
test_snapshot_list = json.loads(config.get("XGBOOST","test_snaps"))
global p_red_shift
p_red_shift = config.getfloat("SEARCH","p_red_shift")
search_rad = config.getfloat("SEARCH","search_rad")
model_type = config["XGBOOST"]["model_type"]
model_sparta_file = config["XGBOOST"]["model_sparta_file"]
model_name = model_type + "_" + model_sparta_file

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
def eval_model(model_info, sparta_file, X, y, preds, dataset_type, dataset_location, model_save_location, p_red_shift, dens_prf = False, r_rv_tv = False, misclass=False):
    plot_save_location = model_save_location + dataset_type + "_" + sparta_file + "/"
    create_directory(plot_save_location)
    
    if len(test_snapshot_list) > 1:
        specific_save = curr_sparta_file + "_" + str(test_snapshot_list[0]) + "to" + str(test_snapshot_list[-1]) + "_" + str(search_rad) + "search/"
    else:
        specific_save = curr_sparta_file + "_" + str(test_snapshot_list[0]) + "_" + str(search_rad) + "search/"

    num_bins = 30
    with open(dataset_location + dataset_type.lower() + "_dataset_all_keys.pickle", "rb") as file:
        all_keys = pickle.load(file)
    p_r_loc = np.where(all_keys == "Scaled_radii_" + str(test_p_snap))[0][0]
    c_r_loc = np.where(all_keys == "Scaled_radii_" + str(test_c_snap))[0][0]
    p_rv_loc = np.where(all_keys == "Radial_vel_" + str(test_p_snap))[0][0]
    c_rv_loc = np.where(all_keys == "Radial_vel_" + str(test_c_snap))[0][0]
    p_tv_loc = np.where(all_keys == "Tangential_vel_" + str(test_p_snap))[0][0]
    c_tv_loc = np.where(all_keys == "Tangential_vel_" + str(test_c_snap))[0][0]
    
    if dens_prf:
        with open(dataset_location + dataset_type.lower() + "_all_rad_halo_first.pickle", "rb") as file:
            halo_first = pickle.load(file) 
        with open(dataset_location + dataset_type.lower() + "_all_rad_halo_n.pickle", "rb") as file:
            halo_n = pickle.load(file)   

        with open(path_to_calc_info + specific_save + "test_indices.pickle", "rb") as pickle_file:
            test_indices = pickle.load(pickle_file)
        
        new_p_snap, p_red_shift = find_closest_z(p_red_shift)
        p_scale_factor = 1/(1+p_red_shift)
        sparta_output = sparta.load(filename=path_to_hdf5_file, load_halo_data=False, log_level= 0)
        all_red_shifts = sparta_output["simulation"]["snap_z"][:]
        p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
        halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, parent_id, ptl_mass = load_or_pickle_SPARTA_data(curr_sparta_file, p_scale_factor, test_p_snap, p_sparta_snap)

        use_halo_ids = halos_id[test_indices]
        sparta_output = sparta.load(filename=path_to_hdf5_file, halo_ids=use_halo_ids, log_level=0)
        all_red_shifts = sparta_output["simulation"]["snap_z"][:]
        p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
        new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta resort the indices
        dens_prf_all = sparta_output['anl_prf']['M_all'][new_idxs,p_sparta_snap,:]
        dens_prf_1halo = sparta_output['anl_prf']['M_1halo'][new_idxs,p_sparta_snap,:]
        
        bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
        bins = np.insert(bins, 0, 0) 

        compare_density_prf(radii=X[:,p_r_loc], halo_first=halo_first, halo_n=halo_n, act_mass_prf_all=dens_prf_all, act_mass_prf_orb=dens_prf_1halo, mass=ptl_mass, orbit_assn=preds, prf_bins=bins, title="", save_location=plot_save_location, use_mp=True, save_graph=True)
    
    if r_rv_tv:
        plot_r_rv_tv_graph(preds, X[:,p_r_loc], X[:,p_rv_loc], X[:,p_tv_loc], y, title="", num_bins=num_bins, save_location=plot_save_location)
    
    if misclass:
        plot_misclassified(p_corr_labels=y, p_ml_labels=preds, p_r=X[:,p_r_loc], p_rv=X[:,p_rv_loc], p_tv=X[:,p_tv_loc], c_r=X[:,c_r_loc], c_rv=X[:,c_rv_loc], c_tv=X[:,c_tv_loc],title="",num_bins=num_bins,save_location=plot_save_location,model_info=model_info,dataset_name=dataset_type + "_" + sparta_file)
    

