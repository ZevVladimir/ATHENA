from dask import array as da
from dask.distributed import Client
#from dask_jobqueue import SLURMCluster

from dask_mpi import initialize
from mpi4py import MPI
from distributed.scheduler import logger
import socket

import xgboost as xgb
from xgboost import dask as dxgb

from sklearn.metrics import classification_report
import pickle
import time
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from pairing import depair
from colossus.cosmology import cosmology

from data_and_loading_functions import create_directory, load_or_pickle_SPARTA_data, conv_halo_id_spid
from visualization_functions import *

from guppy import hpy

##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("/home/zvladimi/scratch/MLOIS/config.ini")
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

frac_training_data = 1

# size float32 is 4 bytes
chunk_size = int(np.floor(1e9 / (num_save_ptl_params * 4)))
###############################################################################################################
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta import sparta


import subprocess

try:
    subprocess.check_output('nvidia-smi')
    gpu_use = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    gpu_use = False

###############################################################################################################

def get_cluster():
    #cluster = LocalCUDACluster(n_workers = 8,
     #                          device_memory_limit='10GB',

    #                         jit_unspill=True)
    
    num_gpu_per_job = 4
    num_jobs = 8
    cluster = SLURMCluster(queue="gpu",
            cores = 4, 
            memory = "300GB", 
            walltime = "01:00:00",
            interface = "ib0",
            local_directory = "/home/zvladimi/scratch/MLOIS/",
            log_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/",
            job_extra_directives = ["--gpus-per-task=a100:1", "--nodes=1"],
            )
    print(cluster.job_script())
    cluster.scale(jobs = num_jobs)
    client = Client(cluster)
    print(client)
    client.wait_for_workers()
    return client

def create_training_matrix(X_loc, y_loc, frac_use_data = 1, calc_scale_pos_weight = False):
    with open(X_loc, "rb") as file:
        X = pickle.load(file) 
    with open(y_loc, "rb") as file:
        y = pickle.load(file)
    
    scale_pos_weight = np.where(y == 0)[0].size / np.where(y == 1)[0].size
    
    num_features = X.shape[1]
    
    num_use_data = int(np.floor(X.shape[0] * frac_use_data))
    X = X[:num_use_data]
    y = y[:num_use_data]
    print("Tot num of train particles:", X.shape[0])
    print("Num use train particles:", num_use_data)

    X = da.from_array(X,chunks=(chunk_size,num_features))

    y = da.from_array(y,chunks=(chunk_size))
    print("converted to array")
        
    print("X Number of total bytes:", X.nbytes, "X Number of Gigabytes:", (X.nbytes)/(10**9))
    print("y Number of total bytes:", y.nbytes, "y Number of Gigabytes:", (y.nbytes)/(10**9))
    
    dqmatrix = xgb.dask.DaskDMatrix(client, X, y)
    print("converted to DaskQuantileDMatrix")
     
    if calc_scale_pos_weight:
        return dqmatrix, scale_pos_weight
    else:
        return dqmatrix


if __name__ == "__main__":      
    t1 = time.time()
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        print("SLURM_CPUS_PER_TASK is not defined.")
    
    if gpu_use:
        initialize(local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
    else:
        initialize(nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
    #initialize(interface="ib0", nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/", worker_class = "dask_cuda.CUDAWorker" )
    print("Initialized")
    client = Client()
    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "zvladimi@login.zaratan.umd.edu" # Change this to the address/domain of your login node

    logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
    
    if len(snapshot_list) > 1:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
    else:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"

    save_location = path_to_xgboost + specific_save

    model_save_location = save_location + "models/"  
    train_dataset_loc = save_location + "datasets/" + "train_dataset.pickle"
    train_labels_loc = save_location + "datasets/" + "train_labels.pickle"
    test_dataset_loc = save_location + "datasets/" + "test_dataset.pickle"
    test_labels_loc = save_location + "datasets/" + "test_labels.pickle"
        
    dtrain,scale_pos_weight = create_training_matrix(train_dataset_loc, train_labels_loc, frac_use_data=1, calc_scale_pos_weight=True)
    dtest = create_training_matrix(test_dataset_loc, test_labels_loc, frac_use_data=1, calc_scale_pos_weight=False)
    print("scale_pos_weight:", scale_pos_weight)
        
    if os.path.isfile(model_save_location + model_name + ".json"):
        bst = xgb.Booster()
        bst.load_model(model_save_location + model_name + ".json")
        print("Loaded Booster")
    else:
        t3 = time.time()
        with open(train_dataset_loc, "rb") as file:
            X = pickle.load(file)
        with open(train_labels_loc, "rb") as file:
            y = pickle.load(file)
        print("Tot num of train particles:", X.shape[0])

        scale_pos_weight = np.where(y == 0)[0].size / np.where(y == 1)[0].size
    
        num_features = X.shape[1]
    
        num_use_data = int(np.floor(X.shape[0] * frac_training_data))

        #rng = np.random.default_rng(seed=11)
        #print(num_use_data, X.shape[0])
        #use_idxs = rng.choice(num_use_data,num_use_data,replace=False, shuffle=False)
        #print(use_idxs)
        X = X[:num_use_data]
        y = y[:num_use_data]
        print("Num use train particles:", num_use_data)

        X = da.from_array(X,chunks=(chunk_size, num_features))
        y = da.from_array(y,chunks=(chunk_size))
        print("converted to array")

        print("X Number of total bytes:", X.nbytes, "X Number of Gigabytes:", (X.nbytes)/(10**9))
        print("y Number of total bytes:", y.nbytes, "y Number of Gigabytes:", (y.nbytes)/(10**9))
    
        #print(client.scheduler_info())
        print(client)
        dtrain = xgb.dask.DaskDMatrix(client, X, y)
        print("converted to DaskQuantileDMatrix")

        with open(test_dataset_loc, "rb") as file:
            X = pickle.load(file)
        with open(test_labels_loc, "rb") as file:
            y = pickle.load(file)
        print("Tot num of test particles:", X.shape[0])

        num_features = X.shape[1]

        X = da.from_array(X,chunks=(chunk_size, num_features))
        y = da.from_array(y,chunks=(chunk_size))
        print("converted to array")

        print("X Number of total bytes:", X.nbytes, "X Number of Gigabytes:", (X.nbytes)/(10**9))
        print("y Number of total bytes:", y.nbytes, "y Number of Gigabytes:", (y.nbytes)/(10**9))

        dtest = xgb.dask.DaskDMatrix(client, X, y)
        print("converted to DaskDMatrix")

        del X
        del y
        
        print("scale_pos_weight:", scale_pos_weight)
        
        t4 = time.time()
        print("Dask Matrixes setup:", np.round(((t4-t3)/60),2), "min")
        print("Start train")
        output = dxgb.train(
            client,
            {
            "verbosity": 1,
            "tree_method": "hist",
            "n_estimators": 200,
            # "nthread": cpus_per_task, 
            # Golden line for GPU training
            # "device": "cuda",
            'scale_pos_weight': scale_pos_weight,
            'max_depth':4,
            },  
            dtrain,
            num_boost_round=250,
            #evals=[(dtrain, "train")],
            evals=[(dtrain, "train"), (dtest,"test")],
            early_stopping_rounds=30,            
            )
        bst = output["booster"]
        history = output["history"]
        create_directory(save_location + "models/")
        
        print(model_save_location + model_name + ".json")
        bst.save_model(model_save_location + model_name + ".json")

        #print("Evaluation history:", history)
        plt.figure(figsize=(10,7))
        plt.plot(history["train"]["rmse"], label="Training loss")
        plt.plot(history["test"]["rmse"], label="Validation loss")
        plt.axvline(21, color="gray", label="Optimal tree number")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(model_save_location + "training_loss_graph.png")

    #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
     #               locals().items())), key= lambda x: -x[1])[:25]:
      #  print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))      
    
    # you can pass output directly into `predict` too.
    
    del dtrain
    del dtest

    # with open(train_dataset_loc, "rb") as file:
    #     X = pickle.load(file)
    # with open(train_labels_loc, "rb") as file:
    #     y = pickle.load(file)
    # X = da.from_array(X,chunks=(chunk_size,X.shape[1]))
    
    # train_prediction = dxgb.inplace_predict(client, bst, X).compute()
    # train_prediction = np.round(train_prediction)

    # print("Train Report")
    #print(classification_report(y, train_prediction))


    # del X
    # del y

    t1 = time.time()
    with open(test_dataset_loc, "rb") as file:
        X_np = pickle.load(file)
    with open(test_labels_loc, "rb") as file:
        y_np = pickle.load(file)
    X = da.from_array(X_np,chunks=(chunk_size,X_np.shape[1]))
    
    test_prediction = dxgb.inplace_predict(client, bst, X).compute()

    test_prediction = np.round(test_prediction)
    t2 = time.time()
    print("Predictions finished:", np.round((t2-t1),2),"sec", np.round(((t2-t1)/60),2), "min")
    # print("Test Report")
    # print(classification_report(y_np, test_prediction))


    with open(save_location + "datasets/" + "test_dataset_all_keys.pickle", "rb") as file:
        test_all_keys = pickle.load(file)

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
    compare_density_prf(radii=X_np[:,scaled_radii_loc], actual_prf_all=density_prf_all_within, actual_prf_1halo=density_prf_1halo_within, mass=ptl_mass, orbit_assn=test_prediction, prf_bins=bins, title = model_name + " Predicts", show_graph = False, save_graph = True, save_location = save_location)
    plot_r_rv_tv_graph(test_prediction, X_np[:,scaled_radii_loc], X_np[:,rad_vel_loc], X_np[:,tang_vel_loc], y_np, model_name + " Predicts", num_bins, show = False, save = True, save_location=save_location)
    #ssgraph_acc_by_bin(test_prediction, y_np, X_np[:,scaled_radii_loc], num_bins, model_name + " Predicts", plot = False, save = True, save_location = save_location)
 

