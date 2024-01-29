from dask import array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from contextlib import contextmanager

import xgboost as xgb
from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix
from sklearn.metrics import classification_report
import pickle
import time
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pairing import depair
from colossus.cosmology import cosmology
from data_and_loading_functions import create_directory, load_or_pickle_SPARTA_data, conv_halo_id_spid
from visualization_functions import *
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
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
global p_snap
p_snap = config.getint("SEARCH","p_snap")
c_snap = config.getint("XGBOOST","c_snap")
model_name = config["XGBOOST"]["model_name"]
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
# size float32 is 4 bytes
chunk_size = int(np.floor(1e9 / (num_save_ptl_params * 4)))
frac_training_data = 1

import subprocess

try:
    subprocess.check_output('nvidia-smi')
    gpu_use = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    gpu_use = False
###############################################################################################################
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta import sparta
@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))

def get_cluster():
    cluster = LocalCUDACluster(
                               device_memory_limit='10GB',
                               jit_unspill=True)
    client = Client(cluster)
    return client

def make_preds(client, bst, dataset_loc, labels_loc, report_name="Classification Report", print_report=False):
    with open(dataset_loc, "rb") as file:
        X_np = pickle.load(file)
    with open(labels_loc, "rb") as file:
        y_np = pickle.load(file)
    X = da.from_array(X_np,chunks=(chunk_size,X_np.shape[1]))
    
    preds = dxgb.inplace_predict(client, bst, X).compute()
    preds = np.round(preds)
    
    if print_report:
        print(report_name)    
        report = classification_report(y_np, preds)
        print(report)
        file = open(model_save_location + "model_info.txt", 'a')
        file.write(report_name+"\n")
        file.write(report)
        file.close()
    
    return preds

if __name__ == "__main__":
    client = get_cluster()
    if len(snapshot_list) > 1:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
    else:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"

    if gpu_use:
        model_name = model_name + "_frac_" + str(frac_training_data) + "_gpu_model"
    else:
        model_name = model_name + "_frac_" + str(frac_training_data) + "_cpu_model"

    save_location = path_to_xgboost + specific_save
    model_save_location = save_location + model_name + "/"  
    plot_save_location = model_save_location + "plots/"
    create_directory(model_save_location)
    create_directory(plot_save_location)
    train_dataset_loc = save_location + "datasets/" + "train_dataset.pickle"
    train_labels_loc = save_location + "datasets/" + "train_labels.pickle"
    test_dataset_loc = save_location + "datasets/" + "test_dataset.pickle"
    test_labels_loc = save_location + "datasets/" + "test_labels.pickle"
    
    file = open(model_save_location + "model_info.txt", 'w')
    file.write("SPARTA File: " +curr_sparta_file+ "\n")
    snap_str = "Snapshots used: "
    for snapshot in snapshot_list:
        snap_str += (str(snapshot) + "_")
    file.write(snap_str)
    file.write("Search Radius: " + str(search_rad) + "\n")
    file.write("Fraction of training data used: "+str(frac_training_data)+"\n")
    file.close()
    
    with open(save_location + "datasets/" + "test_dataset_all_keys.pickle", "rb") as file:
        test_all_keys = pickle.load(file)

    if os.path.isfile(model_save_location + model_name + ".json"):
        bst = xgb.Booster()
        bst.load_model(model_save_location + model_name + ".json")
    
        print(bst.get_score())
        fig, ax = plt.subplots(figsize=(400, 10))
        xgb.plot_tree(bst, num_trees=4, ax=ax)
        plt.savefig("/home/zvladimi/MLOIS/Random_figures/temp.png")
        print("Loaded Booster")
    else:
        print("Couldn't load Booster Located at: " + model_save_location + model_name + ".json")
        
    train_preds = make_preds(client, bst, train_dataset_loc, train_labels_loc, report_name="Train Report", print_report=False)
    test_preds = make_preds(client, bst, test_dataset_loc, test_labels_loc, report_name="Test Report", print_report=False)
    
    with open(test_dataset_loc, "rb") as file:
        X_np = pickle.load(file)
    with open(test_labels_loc, "rb") as file:
        y_np = pickle.load(file)
    
    for i,key in enumerate(test_all_keys):
        if key == "Scaled_radii_" + str(p_snap):
            scaled_radii_loc = i
        elif key == "Radial_vel_" + str(p_snap):
            rad_vel_loc = i
        elif key == "Tangential_vel_" + str(p_snap):
            tang_vel_loc = i
    with open(path_to_calc_info + specific_save + "test_indices.pickle", "rb") as pickle_file:
        test_indices = pickle.load(pickle_file)
    
    p_snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)
    p_red_shift = readheader(p_snapshot_path, 'redshift')
    p_scale_factor = 1/(1+p_red_shift)
    halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, ptl_mass = load_or_pickle_SPARTA_data(curr_sparta_file, p_scale_factor, p_snap)
    cosmol = cosmology.setCosmology("bolshoi")
    use_halo_ids = halos_id[test_indices]
    sparta_output = sparta.load(filename=path_to_hdf5_file, halo_ids=use_halo_ids, log_level=0)
    new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_snap) # If the order changed by sparta resort the indices
    dens_prf_all = sparta_output['anl_prf']['M_all'][new_idxs,p_snap,:]
    dens_prf_1halo = sparta_output['anl_prf']['M_1halo'][new_idxs,p_snap,:]
    # test indices are the indices of the match halo idxs used (see find_particle_properties_ML.py to see how test_indices are created)
    num_test_halos = test_indices.shape[0]
    density_prf_all_within = np.sum(dens_prf_all, axis=0)
    density_prf_1halo_within = np.sum(dens_prf_1halo, axis=0)
    num_bins = 30
    bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
    bins = np.insert(bins, 0, 0)
    compare_density_prf(radii=X_np[:,scaled_radii_loc], act_mass_prf_all=density_prf_all_within, act_mass_prf_1halo=density_prf_1halo_within, mass=ptl_mass, orbit_assn=test_preds, prf_bins=bins, title = model_name, show_graph = False, save_graph = True, save_location = plot_save_location)
    plot_r_rv_tv_graph(test_preds, X_np[:,scaled_radii_loc], X_np[:,rad_vel_loc], X_np[:,tang_vel_loc], y_np, model_name, num_bins, show = False, save = True, save_location=plot_save_location)
    #ssgraph_acc_by_bin(test_prediction, y_np, X_np[:,scaled_radii_loc], num_bins, model_name + " Predicts", plot = False, save = True, save_location = plot_save_location)
