import dask_cudf
from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import xgboost as xgb
from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix

from sklearn.metrics import classification_report
import pickle
import time
import os
import sys
import numpy as np
from data_and_loading_functions import create_directory

import matplotlib as plt

from guppy import hpy
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
###############################################################################################################

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


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

    model_save_location = save_location + "models/" + model_name + "/"

    train_dataset_loc = save_location + "datasets/" + "train_dataset.pickle"
    test_dataset_loc = save_location + "datasets/" + "test_dataset.pickle"
            
        
    with open(train_dataset_loc, "rb") as file:
        train_dataset = pickle.load(file) 
            
    num_features = train_dataset[:,2:].shape[1]
        
    num_use_data = int(np.floor(train_dataset.shape[0] * 0.5))
    print("Tot num of train particles:",train_dataset.shape[0])
    print("Num use train particles:", num_use_data)
    scale_pos_weight = np.where(train_dataset[:num_use_data,1] == 0)[0].size / np.where(train_dataset[:num_use_data,1] == 1)[0].size
    X = da.from_array(train_dataset[:num_use_data,2:],chunks=(10000,num_features))
    y = da.from_array(train_dataset[:num_use_data,1],chunks=(10000))
    print("converted to array")
        
    print("X Number of total bytes:",X.nbytes, "X Number of Gigabytes:", (X.nbytes)/(10**9))
    print("y Number of total bytes:",y.nbytes, "y Number of Gigabytes:", (y.nbytes)/(10**9))
    del train_dataset
    #h = hpy()
    #print(h.heap())
    dtrain = xgb.dask.DaskQuantileDMatrix(client, X, y)
    print("converted to DaskQuantileDMatrix")
    #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
    #        locals().items())), key= lambda x: -x[1])[:25]:
    #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    del X
    del y

    with open(test_dataset_loc, "rb") as file:
        test_dataset = pickle.load(file)

    num_features = test_dataset[:,2:].shape[1]

    print("Tot num of test particles:",test_dataset.shape[0])
    
    X = da.from_array(test_dataset[:,2:],chunks=(1000000,num_features))
    y = da.from_array(test_dataset[:,1],chunks=(1000000))
    print("converted to array")

    print("X Number of total bytes:",X.nbytes, "X Number of Gigabytes:", (X.nbytes)/(10**9))
    print("y Number of total bytes:",y.nbytes, "y Number of Gigabytes:", (y.nbytes)/(10**9))
    del test_dataset
    #h = hpy()
    #print(h.heap())
    dtest = xgb.dask.DaskQuantileDMatrix(client, X, y)
    print("converted to DaskQuantileDMatrix")
    #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
    #        locals().items())), key= lambda x: -x[1])[:25]:
    #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    del X
    del y
        
    if os.path.isfile("/home/zvladimi/scratch/MLOIS/xgboost_datasets_plots/sparta_cbol_l0125_n1024_90to79_6.0r200msearch/models/big_model.json"):
        bst = xgb.Booster()
        bst.load_model("/home/zvladimi/scratch/MLOIS/xgboost_datasets_plots/sparta_cbol_l0125_n1024_90to79_6.0r200msearch/models/big_model.json")
        print("Loaded Booster")
    else:
        print("Start train")
        output = dxgb.train(
            client,
            {
            "verbosity": 1,
            "tree_method": "hist",
            # Golden line for GPU training
            "device": "cuda",
            'scale_pos_weight': scale_pos_weight,
            'max_depth':4,
            },
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dtest,"test")],
            early_stopping_rounds=20
            )
        bst = output["booster"]
        history = output["history"]
        bst.save_model(save_location + "models/big_model.json")
        print("Evaluation history:", history)
        results = bst.evals_result()

        plt.figure(figsize=(10,7))
        plt.plot(results["validation_0"]["rmse"], label="Training loss")
        plt.plot(results["validation_1"]["rmse"], label="Validation loss")
        plt.axvline(21, color="gray", label="Optimal tree number")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("/home/zvladimi/scratch/MLOIS/loss_graph.png")
    #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
     #               locals().items())), key= lambda x: -x[1])[:25]:
      #  print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))      
    
    # you can pass output directly into `predict` too.
    prediction = dxgb.predict(client, bst, dtrain)
    print("Made Predictions")
    prediction = np.round(prediction)
    del bst
    del dtrain
    h = hpy()
    print(h.heap())
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                    locals().items())), key= lambda x: -x[1])[:25]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    #for i in range(10):
     #   print(h[i].byvia)

    with open(dataset_loc, "rb") as file:
        dataset = pickle.load(file)
    print(classification_report(dataset[:num_use_data,1], prediction))
        
