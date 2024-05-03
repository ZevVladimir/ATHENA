from dask import array as da
from dask.distributed import Client

from contextlib import contextmanager

import xgboost as xgb
from xgboost import dask as dxgb

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

from utils.data_and_loading_functions import create_directory, load_or_pickle_SPARTA_data, conv_halo_id_spid, find_closest_z
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
p_snap = config.getint("XGBOOST","p_snap")
c_snap = config.getint("XGBOOST","c_snap")
model_type = config["XGBOOST"]["model_type"]
model_sparta_file = config["XGBOOST"]["model_sparta_file"]
model_name = model_type + "_" + model_sparta_file
radii_splits = config.get("XGBOOST","rad_splits").split(',')
snapshot_list = [p_snap, c_snap]
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

from utils.visualization_functions import *
from utils.ML_support import *

import subprocess

try:
    subprocess.check_output('nvidia-smi')
    gpu_use = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    gpu_use = False

if on_zaratan:
    from dask_mpi import initialize
    from mpi4py import MPI
    from distributed.scheduler import logger
    import socket
    #from dask_jobqueue import SLURMCluster
else:
    from dask_cuda import LocalCUDACluster
    from cuml.metrics.accuracy import accuracy_score #TODO fix cupy installation??
    from sklearn.metrics import make_scorer
    import dask_ml.model_selection as dcv
###############################################################################################################
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta
@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))

def get_CUDA_cluster():
    cluster = LocalCUDACluster(
                               device_memory_limit='10GB',
                               jit_unspill=True)
    client = Client(cluster)
    return client



def accuracy_score_wrapper(y, y_hat): 
    y = y.astype("float32") 
    return accuracy_score(y, y_hat, convert_dtype=True)

def do_HPO(model, gridsearch_params, scorer, X, y, mode='gpu-Grid', n_iter=10):
    if mode == 'gpu-grid':
        clf = dcv.GridSearchCV(model,
                               gridsearch_params,
                               cv=N_FOLDS,
                               scoring=scorer)
    elif mode == 'gpu-random':
        clf = dcv.RandomizedSearchCV(model,
                               gridsearch_params,
                               cv=N_FOLDS,
                               scoring=scorer,
                               n_iter=n_iter)

    else:
        print("Unknown Option, please choose one of [gpu-grid, gpu-random]")
        return None, None
    res = clf.fit(X, y,eval_metric='rmse')
    print("Best clf and score {} {}\n---\n".format(res.best_estimator_, res.best_score_))
    return res.best_estimator_, res

def print_acc(model, X_train, y_train, X_test, y_test, mode_str="Default"):
    """
        Trains a model on the train data provided, and prints the accuracy of the trained model.
        mode_str: User specifies what model it is to print the value
    """
    y_pred = model.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test.astype('float32'), convert_dtype=True)
    print("{} model accuracy: {}".format(mode_str, score))

def make_preds(client, bst, X_np, y_np, report_name="Classification Report", print_report=False):
    X = da.from_array(X_np,chunks=(chunk_size,X_np.shape[1]))
    
    preds = dxgb.inplace_predict(client, bst, X).compute()
    preds = np.round(preds)
    preds = preds.astype(np.int8)
    
    return preds

if __name__ == "__main__":
    if on_zaratan:
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            print("SLURM_CPUS_PER_TASK is not defined.")
        if gpu_use:
            initialize(local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        else:
            initialize(nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        print("Initialized")
        client = Client()
        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        login_node_address = "zvladimi@login.zaratan.umd.edu" # Change this to the address/domain of your login node

        logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
    else:
        client = get_CUDA_cluster()
        
    if len(snapshot_list) > 1:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "search/"
    else:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "search/"

    if gpu_use:
        model_name = model_name + "_frac_" + str(frac_training_data) + "_gpu"
    else:
        model_name = model_name + "_frac_" + str(frac_training_data) + "_cpu"

    save_location = path_to_xgboost + specific_save
    dataset_location = save_location + "datasets/"
    model_save_location = save_location + model_type + "_" + str(frac_training_data) + "_gpu/"  
    gen_plot_save_location = model_save_location + "plots/"
    create_directory(model_save_location)
    create_directory(gen_plot_save_location)
    
    train_dataset_loc = dataset_location + "train_within_rad_dataset.pickle"
    train_labels_loc = dataset_location + "train_within_rad_labels.pickle"
    test_dataset_loc = dataset_location + "test_dataset.pickle"
    test_labels_loc = dataset_location + "test_labels.pickle"
    train_keys_loc = dataset_location + "train_dataset_all_keys.pickle"
    test_keys_loc = dataset_location + "test_dataset_all_keys.pickle"
    
    if os.path.isfile(model_save_location + "model_info.pickle"):
        with open(model_save_location + "model_info.pickle", "rb") as pickle_file:
            model_info = pickle.load(pickle_file)
    else:
        model_info = {
            'Misc Info':{
                'Model trained on': model_sparta_file,
                'Snapshots used': snapshot_list,
                'Max Radius': search_rad,
                }}
    
    if os.path.isfile(model_save_location + model_name + ".json"):
        bst = xgb.Booster()
        bst.load_model(model_save_location + model_name + ".json")
        params = model_info.get('Training Info',{}).get('Training Params')
        print("Loaded Booster")
    else:
        print("Training Set:")
        with open(train_dataset_loc, "rb") as file:
            train_X = pickle.load(file) 
        with open(train_labels_loc, "rb") as file:
            train_y = pickle.load(file)
        with open(train_keys_loc, "rb") as file:
            train_features = pickle.load(file)
        train_features = train_features.tolist()

        dtrain,X_train,y_train,scale_pos_weight = create_dmatrix(client, train_X, train_y, train_features, chunk_size=chunk_size, frac_use_data=frac_training_data, calc_scale_pos_weight=True)
        del train_X
        del train_y
        del train_features
        
        print("Testing set:")
        with open(test_dataset_loc, "rb") as file:
            test_X = pickle.load(file) 
        with open(test_labels_loc, "rb") as file:
            test_y = pickle.load(file)
        with open(test_keys_loc, "rb") as file:
            test_features = pickle.load(file)
        test_features = test_features.tolist()
        dtest,X_test,y_test = create_dmatrix(client, test_X, test_y, test_features, chunk_size=chunk_size, frac_use_data=1, calc_scale_pos_weight=False)
        del test_X
        del test_y
        del test_features
        print("scale_pos_weight:", scale_pos_weight)
        
        if on_zaratan == False and do_hpo == True and os.path.isfile(model_save_location + "used_params.pickle") == False:  
            params = {
            # Parameters that we are going to tune.
            'max_depth':np.arange(2,4,1),
            # 'min_child_weight': 1,
            'learning_rate':np.arange(0.01,1.01,.1),
            'scale_pos_weight':np.arange(scale_pos_weight,scale_pos_weight+10,1),
            'lambda':np.arange(0,3,.5),
            'alpha':np.arange(0,3,.5),
            # 'subsample': 1,
            # 'colsample_bytree': 1,
            }
        
            N_FOLDS = 5
            N_ITER = 25
            
            model = dxgb.XGBClassifier(tree_method='gpu_hist', n_estimators=100, use_label_encoder=False, scale_pos_weight=scale_pos_weight)
            accuracy_wrapper_scorer = make_scorer(accuracy_score_wrapper)
            cuml_accuracy_scorer = make_scorer(accuracy_score, convert_dtype=True)
            print_acc(model, X_train, y_train, X_test, y_test)
            
            mode = "gpu-random"

            if os.path.isfile(model_save_location + "hyper_param_res.pickle") and os.path.isfile(model_save_location + "hyper_param_results.pickle"):
                with open(model_save_location + "hyper_param_res.pickle", "rb") as pickle_file:
                    res = pickle.load(pickle_file)
                with open(model_save_location + "hyper_param_results.pickle", "rb") as pickle_file:
                    results = pickle.load(pickle_file)
            else:
                with timed("XGB-"+mode):
                    res, results = do_HPO(model,
                                            params,
                                            cuml_accuracy_scorer,
                                            X_train,
                                            y_train,
                                            mode=mode,
                                            n_iter=N_ITER)
                with open(model_save_location + "hyper_param_res.pickle", "wb") as pickle_file:
                    pickle.dump(res, pickle_file)
                with open(model_save_location + "hyper_param_results.pickle", "wb") as pickle_file:
                    pickle.dump(results, pickle_file)
                    
                print("Searched over {} parameters".format(len(results.cv_results_['mean_test_score'])))
                print_acc(res, X_train, y_train, X_test, y_test, mode_str=mode)
                print("Best params", results.best_params_)
                
                params = results.best_params_
                
                model_info['Training Info']={
                'Fraction of Training Data Used': frac_training_data,
                'Trained on GPU': gpu_use,
                'HPO used': do_hpo,
                'Training Params': params,
                }            
                
        elif 'Training Info' in model_info: 
            params = model_info.get('Training Info',{}).get('Training Params')
        else:
            params = {
                "verbosity": 1,
                "tree_method": "hist",
                # Golden line for GPU training
                "scale_pos_weight":scale_pos_weight,
                "device": "cuda",
                }
            model_info['Training Info']={
                'Fraction of Training Data Used': frac_training_data,
                'Training Params': params}
            
        print("Starting train using params:", params)
        output = dxgb.train(
            client,
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train"), (dtest,"test")],
            early_stopping_rounds=10,            
            )
        bst = output["booster"]
        history = output["history"]
        bst.save_model(model_save_location + model_name + ".json")

        plt.figure(figsize=(10,7))
        plt.plot(history["train"]["rmse"], label="Training loss")
        plt.plot(history["test"]["rmse"], label="Validation loss")
        plt.axvline(21, color="gray", label="Optimal tree number")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(gen_plot_save_location + "training_loss_graph.png")
    
        del dtrain
        del dtest

    # with timed("Train Predictions"):
    #     with open(train_dataset_loc, "rb") as file:
    #         train_X = pickle.load(file)
    #     with open(train_labels_loc, "rb") as file:
    #         train_y = pickle.load(file)
    #     train_preds = make_preds(client, bst, train_X, train_y, report_name="Train Report", print_report=False)
    # with timed("Train Plots"):
    #     eval_model(model_info=model_info,sparta_file=model_sparta_file,X=train_X, y=train_y,preds=train_preds,dataset_type="Train",dataset_location=dataset_location,model_save_location=model_save_location,p_red_shift=p_red_shift,dens_prf=False, r_rv_tv=True,misclass=True)
    
    with timed("Test Predictions"):
        with open(test_dataset_loc, "rb") as file:
            test_X = pickle.load(file)
        with open(test_labels_loc, "rb") as file:
            test_y = pickle.load(file)
        test_preds = make_preds(client, bst, test_X, test_y, report_name="Test Report", print_report=False)
    with timed("Test Plots"):
        eval_model(model_info=model_info,sparta_file=curr_sparta_file,X=test_X, y=test_y,preds=test_preds,dataset_type="Test", dataset_location=dataset_location,model_save_location=model_save_location,p_red_shift=p_red_shift,dens_prf=True, r_rv_tv=True, misclass=True)    
        
    bst.save_model(model_save_location + model_name + ".json")

    feature_important = bst.get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    pos = np.arange(len(keys))

    fig, ax = plt.subplots(1, figsize=(15,10))
    ax.barh(pos,values)
    ax.set_yticks(pos, keys)
    fig.savefig(gen_plot_save_location + "feature_importance.png")

    # tree_num = 2
    # xgb.plot_tree(bst, num_trees=tree_num)
    # fig = plt.gcf()
    # fig.set_size_inches(110, 25)
    # fig.savefig('/home/zvladimi/MLOIS/Random_figures/' + model_name + 'tree' + str(tree_num) + '.png')

    with open(model_save_location + "model_info.pickle", "wb") as pickle_file:
        pickle.dump(model_info, pickle_file) 
    client.close()