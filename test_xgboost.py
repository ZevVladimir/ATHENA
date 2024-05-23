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
from utils.data_and_loading_functions import create_directory, load_or_pickle_SPARTA_data, conv_halo_id_spid
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
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
sys.path.insert(1, path_to_pygadgetreader)
sys.path.insert(1, path_to_sparta)
from utils.visualization_functions import *
from pygadgetreader import readsnap, readheader
from sparta_dev import sparta
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

if __name__ == "__main__":
    client = get_cluster()
    

    save_location = path_to_xgboost + specific_save
    model_save_location = save_location + model_name + "/"  
    plot_save_location = model_save_location + "plots/"
    create_directory(model_save_location)
    create_directory(plot_save_location)
    train_dataset_loc = save_location + "datasets/" + "train_dataset.pickle"
    train_labels_loc = save_location + "datasets/" + "train_labels.pickle"
    test_dataset_loc = save_location + "datasets/" + "test_dataset.pickle"
    test_labels_loc = save_location + "datasets/" + "test_labels.pickle"
    
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
