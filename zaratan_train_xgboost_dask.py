from dask import array as da
from dask.distributed import Client
#from dask_jobqueue import SLURMCluster

from dask_mpi import initialize
from mpi4py import MPI
from distributed.scheduler import logger
import socket

from cuml.metrics.accuracy import accuracy_score
from sklearn.metrics import make_scorer
import dask_ml.model_selection as dcv
from contextlib import contextmanager

import xgboost as xgb
from xgboost import dask as dxgb

from colossus.cosmology import cosmology
from sklearn.metrics import classification_report
import pickle
import time
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from data_and_loading_functions import create_directory, load_or_pickle_SPARTA_data, conv_halo_id_spid
from visualization_functions import *
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

snapshot_list = [p_snap, c_snap]
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")

model_name = config["XGBOOST"]["model_name"]
do_hpo = config.getboolean("XGBOOST","hpo")
frac_training_data = 1


# size float32 is 4 bytes
#chunk_size = int(np.floor(1e9 / (num_save_ptl_params * 4))/4)
chunk_size = 100000

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

def accuracy_score_wrapper(y, y_hat): 
    y = y.astype("float32") 
    return accuracy_score(y, y_hat, convert_dtype=True)

def do_HPO(model, gridsearch_params, scorer, X, y, mode='gpu-Grid', n_iter=10):
    if mode == 'gpu-grid':
        print("gpu-grid selected")
        clf = dcv.GridSearchCV(model,
                               gridsearch_params,
                               cv=N_FOLDS,
                               scoring=scorer)
    elif mode == 'gpu-random':
        print("gpu-random selected")
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

    X = da.from_array(X,chunks=(chunk_size, num_features))
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


    if gpu_use:
        model_name = model_name + "_frac_" + str(frac_training_data) + "_gpu_model"
    else:
        model_name = model_name + "_frac_" + str(frac_training_data) + "_cpu_model"
    save_location = path_to_xgboost + specific_save
    model_save_location = save_location + model_name + "/"  
    plot_save_location = model_save_location + "plots/"
    create_directory(model_save_location)
    create_directory(plot_save_location)

    

    

    if gpu_use:
        model_name = str(frac_training_data) + "_gpu_model"
    else:
        model_name = str(frac_training_data) + "_cpu_model"

    train_dataset_loc = dataset_location + "train_dataset.pickle"
    train_labels_loc = dataset_location + "train_labels.pickle"
    test_dataset_loc = dataset_location + "test_dataset.pickle"
    test_labels_loc = dataset_location + "test_labels.pickle"
    
    t2 = time.time()
    print("Dask Client setup:", np.round(((t2-t1)/60),2), "min")


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
    
        num_features = X.shape[1]
    
        num_use_data = int(np.floor(X.shape[0] * frac_training_data))

        #rng = np.random.default_rng(seed=11)
        #print(num_use_data, X.shape[0])
        #use_idxs = rng.choice(num_use_data,num_use_data,replace=False, shuffle=False)
        #print(use_idxs)
        X = X[:num_use_data]
        y = y[:num_use_data]
        scale_pos_weight = np.where(y == 0)[0].size / np.where(y == 1)[0].size
        print("Num use train particles:", num_use_data)

        X_train = da.from_array(X,chunks=(chunk_size, num_features))
        y_train = da.from_array(y,chunks=(chunk_size))
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

        X_test = da.from_array(X,chunks=(chunk_size, num_features))
        y_test = da.from_array(y,chunks=(chunk_size))
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
        
        if do_hpo == True and os.path.isfile(model_save_location + "best_params.pickle") == False:  
            params = {
            # Parameters that we are going to tune.
            'max_depth':np.arange(2,6,1),
            # 'min_child_weight': 1,
            'learning_rate':np.arange(0.01,1.01,.1),
            'scale_pos_weight':np.arange(1,10,.1)
            # 'subsample': 1,
            # 'colsample_bytree': 1,
            }
        
            N_FOLDS = 5
            N_ITER = 25
            
            model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=750, use_label_encoder=False)
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
                params["tree_method"]= "hist"
                
                with open(model_save_location + "best_params.pickle", "wb") as pickle_file:
                    pickle.dump(results.best_params_, pickle_file)
                
                file = open(model_save_location + "model_info.txt", 'w')
                file.write("SPARTA File: " +curr_sparta_file+ "\n")
                snap_str = "Snapshots used: "
                for snapshot in snapshot_list:
                    snap_str += (str(snapshot) + "_")
                file.write(snap_str)
                file.write("Search Radius: " + str(search_rad))
                file.write("Fraction of training data used: "+str(frac_training_data)+"\n")
                for item in results.best_params_.items():
                    file.write(str(item[0]) + ": " + str(item[1]) + "\n")
                file.close()
                
                print(results)
                # df_gridsearch = pd.DataFrame(results.cv_results_)
                # print(df_gridsearch)
                # heatmap = sns.heatmap(df_gridsearch, annot=True)
                # fig = heatmap.get_figure()
                # fig.savefig(plot_save_location + "param_heatmap.png")

        elif os.path.isfile(model_save_location + "best_params.pickle"):
            with open(model_save_location + "best_params.pickle", "rb") as pickle_file:
                params = pickle.load(pickle_file)
        else:
            params = {
                "verbosity": 1,
                "tree_method": "hist",
                # Golden line for GPU training
                "device": "cuda",
                'scale_pos_weight': scale_pos_weight,
            }
            
        if os.path.isfile(model_save_location + model_name + ".json"):
            bst = xgb.Booster()
            # bst.load_model("/home/zvladimi/MLOIS/0.25_cpu_model.json")
            bst.load_model(model_save_location + model_name + ".json")
            print("Loaded Booster")
        else:
            print("Starting train using params:", params)
            output = dxgb.train(
                client,
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train"), (dtest,"test")],
                early_stopping_rounds=20,            
                )
            bst = output["booster"]
            history = output["history"]
            bst.save_model(model_save_location + model_name + ".json")
            #print("Evaluation history:", history)
            plt.figure(figsize=(10,7))
            plt.plot(history["train"]["rmse"], label="Training loss")
            plt.plot(history["test"]["rmse"], label="Validation loss")
            plt.axvline(21, color="gray", label="Optimal tree number")
            plt.xlabel("Number of trees")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(plot_save_location + "training_loss_graph.png")
        print("Start train")
        output = dxgb.train(
            client,
            params,  
            dtrain,
            num_boost_round=250,
            evals=[(dtrain, "train"), (dtest,"test")],
            early_stopping_rounds=20,            
            )
        bst = output["booster"]
        history = output["history"]

        bst.save_model(model_save_location + model_name + ".json")

        t5 = time.time()
        print("Training time:", np.round(((t5-t4)/60),2), "min")
        #print("Evaluation history:", history)
        plt.figure(figsize=(10,7))
        plt.plot(history["train"]["rmse"], label="Training loss")
        plt.plot(history["test"]["rmse"], label="Validation loss")
        plt.axvline(21, color="gray", label="Optimal tree number")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("/home/zvladimi/scratch/MLOIS/Random_figures/" + model_name + "_training_loss_graph.png")
    #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
     #               locals().items())), key= lambda x: -x[1])[:25]:
      #  print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))      
    
    # you can pass output directly into `predict` too.
    
        del dtrain
        del dtest
    
    bst = client.scatter(bst)
    
    t6 = time.time()
    #with open(train_dataset_loc, "rb") as file:
    #    X = pickle.load(file)
    #with open(train_labels_loc, "rb") as file:
    #    y = pickle.load(file)
    #X = da.from_array(X,chunks=(chunk_size,X.shape[1]))
    
    #train_prediction = dxgb.inplace_predict(client, bst, X)
    #train_prediction = train_prediction.compute()
    #train_prediction = np.round(train_prediction)

    #print("Train Report")
    #print(classification_report(y, train_prediction))
    
    t7 = time.time()
    #print("Training predictions:", np.round(((t7-t6)/60),2), "min")

    #del X
    #del y

    with open(test_dataset_loc, "rb") as file:
        X_np = pickle.load(file)
    with open(test_labels_loc, "rb") as file:
        y_np = pickle.load(file)

    X = da.from_array(X_np,chunks=(chunk_size,X_np.shape[1]))
    
    test_prediction = dxgb.inplace_predict(client, bst, X)
    test_prediction = test_prediction.compute()
    test_prediction = np.round(test_prediction)
    

    #print("Test Report")
    #print(classification_report(y_np, test_prediction))

    
    t8 = time.time()
    print("Testing predictions:", np.round(((t8-t7)/60),2), "min")

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
    compare_density_prf(radii=X_np[:,scaled_radii_loc], act_mass_prf_all=density_prf_all_within, act_mass_prf_1halo=density_prf_1halo_within, mass=ptl_mass, orbit_assn=test_prediction, prf_bins=bins, title = model_name + " Predicts", show_graph = False, save_graph = True, save_location = plot_save_location)
    plot_r_rv_tv_graph(test_prediction, X_np[:,scaled_radii_loc], X_np[:,rad_vel_loc], X_np[:,tang_vel_loc], y_np, model_name + " Predicts", num_bins, show = False, save = True, save_location=plot_save_location)
    #ssgraph_acc_by_bin(test_prediction, y_np, X_np[:,scaled_radii_loc], num_bins, model_name + " Predicts", plot = False, save = True, save_location = plot_save_location)

    client.shutdown()
        
