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
import json
import re
from utils.ML_support import *
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
p_red_shift = config.getfloat("SEARCH","p_red_shift")
radii_splits = config.get("XGBOOST","rad_splits").split(',')
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
test_halos_ratio = config.getfloat("DATASET","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
do_hpo = config.getboolean("XGBOOST","hpo")
# size float32 is 4 bytes
chunk_size = int(np.floor(1e9 / (num_save_ptl_params * 4)))
frac_training_data = 1
model_sims = json.loads(config.get("DATASET","model_sims"))
model_type = config["XGBOOST"]["model_type"]
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))

import subprocess

try:
    subprocess.check_output('nvidia-smi')
    gpu_use = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    gpu_use = False
###############################################################################################################

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
    
    combined_model_name = ""
    for i,sim in enumerate(model_sims):
        pattern = r"(\d+)to(\d+)"
        match = re.search(pattern, sim)

        if match:
            curr_snap_list = [match.group(1), match.group(2)] 
        else:
            print("Pattern not found in the string.")
        parts = sim.split("_")
        combined_model_name += parts[1] + parts[2] + "s" + parts[4] 
        if i != len(model_sims)-1:
            combined_model_name += "_"
            
    combined_test_name = ""
    for i,sim in enumerate(test_sims):
        pattern = r"(\d+)to(\d+)"
        match = re.search(pattern, sim)

        if match:
            curr_snap_list = [match.group(1), match.group(2)] 
            print(f"First number: {match.group(1)}")
            print(f"Second number: {match.group(2)}")
        else:
            print("Pattern not found in the string.")
        parts = sim.split("_")
        combined_test_name += parts[1] + parts[2] + "s" + parts[4] 
        if i != len(test_sims)-1:
            combined_test_name += "_"
        
    model_name = model_type + "_" + combined_model_name

    test_dataset_loc = path_to_xgboost + combined_test_name + "/datasets/"
    model_save_loc = path_to_xgboost + combined_model_name + "/" + model_name + "/"

    try:
        bst = xgb.Booster()
        bst.load_model(model_save_loc + model_name + ".json")
        print("Loaded Booster")
    except:
        print("Couldn't load Booster Located at: " + model_save_loc + model_name + ".json")
        
    try:
        with open(model_save_loc + "model_info.pickle", "rb") as pickle_file:
            model_info = pickle.load(pickle_file)
    except FileNotFoundError:
        print("Model info could not be loaded please ensure the path is correct or rerun train_xgboost.py")
        
    for dst_type in eval_datasets:
        with timed(dst_type + " Plots"):
            plot_loc = model_save_loc + dst_type + "_" + combined_test_name + "/plots/"
            create_directory(model_save_loc + dst_type + "_" + combined_test_name)
            create_directory(plot_loc)
            eval_model(model_info, client, bst, use_sims=model_sims, dst_type=dst_type, dst_loc=test_dataset_loc, combined_name=combined_test_name, plot_save_loc=plot_loc, p_red_shift=p_red_shift, dens_prf = True, r_rv_tv = True, misclass=True)
    
    with open(model_save_loc + "model_info.pickle", "wb") as pickle_file:
        pickle.dump(model_info, pickle_file) 
