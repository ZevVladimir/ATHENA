import xgboost as xgb
from xgboost import dask as dxgb
from dask import array as da

from colossus.cosmology import cosmology    
from sklearn.metrics import classification_report
import pickle
import time
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pairing import depair

from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from data_and_loading_functions import create_directory, load_or_pickle_SPARTA_data, conv_halo_id_spid, find_closest_z
from visualization_functions import *
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("/home/zvladimi/MLOIS/config.ini")
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
create_directory(path_to_MLOIS)
create_directory(path_to_snaps)
create_directory(path_to_SPARTA_data)
create_directory(path_to_hdf5_file)
create_directory(path_to_pickle)
create_directory(path_to_calc_info)
create_directory(path_to_xgboost)
snap_format = config["MISC"]["snap_format"]
global prim_only
prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
p_red_shift = config.getfloat("SEARCH","p_red_shift")
global p_snap
p_snap = config.getint("XGBOOST","p_snap")
c_snap = config.getint("XGBOOST","c_snap")
model_name = config["XGBOOST"]["model_name"]
model_sparta_file = config["XGBOOST"]["model_sparta_file"]
model_name = model_name + "_" + model_sparta_file
radii_splits = config.get("XGBOOST","rad_splits").split(',')
snapshot_list = [p_snap, c_snap]
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
do_hpo = config.getboolean("XGBOOST","hpo")
frac_training_data = config.getfloat("XGBOOST","frac_train_data")
# size float32 is 4 bytes
chunk_size = int(np.floor(1e9 / (num_save_ptl_params * 4)))

###############################################################################################################
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta

def get_CUDA_cluster():
    cluster = LocalCUDACluster(
                               device_memory_limit='10GB',
                               jit_unspill=True)
    client = Client(cluster)
    return client

if __name__ == "__main__":
    if len(snapshot_list) > 1:
            specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
    else:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"
        
    save_location = path_to_xgboost + specific_save
    data_location = path_to_calc_info + specific_save
    dataset_name = "test"

    model_save_location = save_location + model_name + "/"
    bst = xgb.Booster()
    bst.load_model("/home/zvladimi/MLOIS/xgboost_datasets_plots/sparta_cbol_l0063_n0256_10r200m_190to166_10.0r200msearch/base_model_sparta_cbol_l0063_n0256_10r200m_frac_1.0_gpu_model/base_model_sparta_cbol_l0063_n0256_10r200m_frac_1.0_gpu_model.json")

    

    with h5py.File((data_location + dataset_name + "_all_particle_properties_" + curr_sparta_file + ".hdf5"), 'r') as file:
        halo_n = file["Halo_n"][:]
        sorted_indices = np.argsort(halo_n)
        median_index = sorted_indices[(halo_n.size // 2) + 192] 

        test_halo_first = file["Halo_first"][median_index]
        test_halo_n = file["Halo_n"][median_index]
        test_halo_r = file["Scaled_radii_"][test_halo_first:test_halo_first+test_halo_n]
        test_halo_rv = file["Radial_vel_"][test_halo_first:test_halo_first+test_halo_n]
        test_halo_tv = file["Tangential_vel_"][test_halo_first:test_halo_first+test_halo_n]
        test_halo_HIPIDS = file["HIPIDS"][test_halo_first:test_halo_first+test_halo_n]
        test_halo_real_labels = file["Orbit_Infall"][test_halo_first:test_halo_first+test_halo_n]

    test_halo_pid = np.zeros(test_halo_HIPIDS.shape[0])
    for i,hipid in enumerate(test_halo_HIPIDS):
        test_halo_pid[i] = depair(hipid)[0]
        
    test_halo_idx = depair(test_halo_HIPIDS[1])[1]
    
    with open("/home/zvladimi/MLOIS/pickle_data/190_sparta_cbol_l0063_n0256_10r200m/halos_pos.pickle", "rb") as file:
        all_halo_pos = pickle.load(file)
    
    p_snap, p_red_shift = find_closest_z(p_red_shift)
    print("Snapshot number found:", p_snap, "Closest redshift found:", p_red_shift)
    sparta_output = sparta.load(filename=path_to_hdf5_file, load_halo_data=False, log_level= 0)
    all_red_shifts = sparta_output["simulation"]["snap_z"][:]
    p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
    print("corresponding SPARTA snap num:", p_sparta_snap)
    print("check sparta redshift:",all_red_shifts[p_sparta_snap])

    p_scale_factor = 1/(1+p_red_shift)
    
    test_halo_pos = all_halo_pos[test_halo_idx] 
 
    with open("/home/zvladimi/MLOIS/pickle_data/190_sparta_cbol_l0063_n0256_10r200m/pid_190.pickle", "rb") as file:
        all_ptl_pid = pickle.load(file)   
    with open("/home/zvladimi/MLOIS/pickle_data/190_sparta_cbol_l0063_n0256_10r200m/pos_190.pickle", "rb") as file:
        all_ptl_pos = pickle.load(file)    

    test_halo_ptl_pos = all_ptl_pos[np.where(np.isin(all_ptl_pid, test_halo_pid))[0]] * 10**3 * p_scale_factor

    X_np = np.column_stack((test_halo_rv[:,0],test_halo_r[:,0],test_halo_tv[:,0],test_halo_rv[:,1],test_halo_r[:,1],test_halo_tv[:,1]))        
        
    X = da.from_array(X_np,chunks=(chunk_size,X_np.shape[1]))

    client = get_CUDA_cluster()
    test_halo_preds = dxgb.inplace_predict(client, bst, X).compute()
    test_halo_preds = np.round(test_halo_preds)
    test_halo_preds = test_halo_preds.astype(np.int8)

    halo_plot_3d(test_halo_ptl_pos, test_halo_pos, test_halo_real_labels, test_halo_preds)