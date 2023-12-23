from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import cudf
import cuml

import sklearn.model_selection as sk
import dask_ml.model_selection as dcv

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score as sk_acc

from cuml.metrics.accuracy import accuracy_score

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
import time 
import pickle
import os
from imblearn import under_sampling, over_sampling
from data_and_loading_functions import build_ml_dataset, check_pickle_exist_gadget, choose_halo_split, create_directory
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
for split in radii_splits:
    model_name = model_name + "_" + str(split)

snapshot_list = [p_snap, c_snap]
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")

frac_use_data = 1
##################################################################################################################
def get_cluster():
    cluster = LocalCUDACluster(
                               device_memory_limit='10GB',
                               jit_unspill=True)
    client = Client(cluster)
    return client

if __name__ == "__main__":
    client = get_cluster()
    if len(snapshot_list) > 1:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
    else:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"

    save_location = path_to_xgboost + specific_save
    model_save_location = save_location + model_name + "/"  
    plot_save_location = save_location + "plots/"
    create_directory(model_save_location)
    create_directory(plot_save_location)
    train_dataset_loc = save_location + "datasets/" + "train_dataset.pickle"
    train_labels_loc = save_location + "datasets/" + "train_labels.pickle"
    test_dataset_loc = save_location + "datasets/" + "test_dataset.pickle"
    test_labels_loc = save_location + "datasets/" + "test_labels.pickle"
    
    with open(train_dataset_loc, "rb") as file:
        X = pickle.load(file) 
    with open(train_labels_loc, "rb") as file:
        y = pickle.load(file)
    
    scale_pos_weight = np.where(y == 0)[0].size / np.where(y == 1)[0].size
    
    num_features = X.shape[1]
    
    num_use_data = int(np.floor(X.shape[0] * frac_use_data))
    print("Tot num of train particles:", X.shape[0])
    print("Num use train particles:", num_use_data)
    X = da.from_array(X,chunks=(chunk_size,num_features))
    y = da.from_array(y,chunks=(chunk_size))
    print("converted to array")