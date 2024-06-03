from dask import array as da
from dask.distributed import Client
import dask.dataframe as dd
from dask import delayed

import xgboost as xgb
from xgboost import dask as dxgb
    
from sklearn.metrics import classification_report
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import re
import pandas as pd
from colossus.lss import peaks
from colossus.cosmology import cosmology
cosmol = cosmology.setCosmology("bolshoi")

import subprocess

from utils.data_and_loading_functions import create_directory, timed
from utils.visualization_functions import *
from utils.ML_support import *

def parse_ranges(ranges_str):
    ranges = []
    for part in ranges_str.split(','):
        start, end = map(int, part.split('-'))
        ranges.append((start, end))
    return ranges
def create_nu_string(nu_list):
    return '_'.join('-'.join(map(str, tup)) for tup in nu_list)

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

snap_format = config["MISC"]["snap_format"]
global prim_only
prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
p_red_shift = config.getfloat("SEARCH","p_red_shift")
model_sims = json.loads(config.get("DATASET","model_sims"))

model_type = config["XGBOOST"]["model_type"]
radii_splits = config.get("XGBOOST","rad_splits").split(',')
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
test_halos_ratio = config.getfloat("DATASET","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
do_hpo = config.getboolean("XGBOOST","hpo")
frac_training_data = config.getfloat("XGBOOST","frac_train_data")
# size float32 is 4 bytes
chunk_size = int(np.floor(1e9 / (num_save_ptl_params * 4)))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))
train_rad = config.getint("XGBOOST","training_rad")
nu_splits = config["XGBOOST"]["nu_splits"]
nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

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
from pygadgetreader import readsnap, readheader # type: ignore
from sparta_tools import sparta # type: ignore

def split_calc_name(sim):
    sim_pat = r"cbol_l(\d+)_n(\d+)"
    match = re.search(sim_pat, sim)
    if match:
        sparta_name = match.group(0)
    sim_search_pat = sim_pat + r"_(\d+)r200m"
    match = re.search(sim_search_pat, sim)
    if match:
        search_name = match.group(0)
    
    return sparta_name, search_name

def transform_string(input_str):
    # Define the regex pattern to match the string parts
    pattern = r"_([\d]+)to([\d]+)"
    
    # Search for the pattern in the input string
    match = re.search(pattern, input_str)
    
    if not match:
        raise ValueError("Input string:",input_str, "does not match the expected format.")

    # Extract the parts of the string
    prefix = input_str[:match.start()]
    first_number = match.group(1)
    
    # Construct the new string
    new_string = f"{first_number}_{prefix}"
    
    return new_string

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

def cut_by_rad(df, rad_cut):
    return df[df['p_Scaled_radii'] <= rad_cut]

def split_by_nu(df,nus,halo_first,halo_n):    
    mask = pd.Series([False] * nus.shape[0])
    for start, end in nu_splits:
        mask[np.where((nus >= start) & (nus <= end))[0]] = True
    
    halo_n = halo_n[mask]
    halo_first = halo_first[mask]
    halo_last = halo_first + halo_n

    use_idxs = np.concatenate([np.arange(start, end) for start, end in zip(halo_first, halo_last)])

    return df.iloc[use_idxs]

def reform_df(folder_path):
    hdf5_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.h5'):
            hdf5_files.append(f)
    hdf5_files.sort()

    dfs = []
    for file in hdf5_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_hdf(file_path) 
        dfs.append(df) 
    return pd.concat(dfs, ignore_index=True)

def sort_files(folder_path):
    hdf5_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.h5'):
            hdf5_files.append(f)
    hdf5_files.sort()
    return hdf5_files
    
def sim_info_for_nus(sim):
    sparta_name, sparta_search_name = split_calc_name(sim)
            
    with h5py.File(path_to_SPARTA_data + sparta_name + "/" +  sparta_search_name + ".hdf5","r") as f:
        dic_sim = {}
        grp_sim = f['simulation']
        for f in grp_sim.attrs:
            dic_sim[f] = grp_sim.attrs[f]
    
    all_red_shifts = dic_sim['snap_z']
    p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
    use_z = all_red_shifts[p_sparta_snap]
    p_snap_loc = transform_string(sim)
    with open(path_to_pickle + p_snap_loc + "/ptl_mass.pickle", "rb") as pickle_file:
        ptl_mass = pickle.load(pickle_file)
    
    return ptl_mass, use_z


def reform_datasets(client,sim,folder_path,rad_cut=None,filter_nu=None):
    ptl_files = sort_files(folder_path + "/ptl_info/")
    
    # Function to process each file
    @delayed
    def process_file(ptl_index):
        ptl_path = folder_path + f"/ptl_info/ptl_{ptl_index}.h5"
        halo_path = folder_path + f"/halo_info/halo_{ptl_index}.h5"
        
        ptl_df = pd.read_hdf(ptl_path)
        halo_df = pd.read_hdf(halo_path)
        
        # Operations on halo_df
        halo_df["Halo_first"] = halo_df["Halo_first"] - halo_df["Halo_first"][0]
        
        # Compute necessary values
        ptl_mass, use_z = sim_info_for_nus(sim)
        nus = np.array(peaks.peakHeight((halo_df["Halo_n"][:] * ptl_mass), use_z))
        
        # Apply filters
        if filter_nu:
            ptl_df = split_by_nu(ptl_df, nus, halo_df["Halo_first"], halo_df["Halo_n"])
        ptl_df = cut_by_rad(ptl_df, rad_cut=rad_cut)
        
        # Calculate scalar position weight
        scal_pos_weight = calc_scal_pos_weight(ptl_df)
        
        return ptl_df, scal_pos_weight

    # Create delayed tasks for each file
    delayed_results = [process_file(i) for i in range(len(ptl_files))]

    # Compute the results in parallel
    results = client.compute(delayed_results, sync=True)

    # Unpack the results
    ddfs, sim_scal_pos_weight = [], []
    for ptl_df, scal_pos_weight in results:
        scatter_df = client.scatter(ptl_df)
        dask_df = dd.from_delayed(scatter_df)
        ddfs.append(dask_df)
    sim_scal_pos_weight.append(scal_pos_weight)
    
    return dd.concat(ddfs),sim_scal_pos_weight

def calc_scal_pos_weight(df):
    count_negatives = (df['Orbit_infall'] == 0).sum()
    count_positives = (df['Orbit_infall'] == 1).sum()

    scale_pos_weight = count_negatives / count_positives
    return scale_pos_weight

def load_data(client, dset_name, rad_cut=search_rad, filter_nu=False):
    dask_dfs = []
    all_scal_pos_weight = []
    
    for sim in model_sims:
        files_loc = path_to_calc_info + sim + "/" + dset_name

        with timed("Reformed " + dset_name + " Dataset: " + sim):
            ptl_ddf,sim_scal_pos_weight = reform_datasets(client,sim,files_loc,rad_cut=rad_cut,filter_nu=filter_nu)   

        all_scal_pos_weight.append(sim_scal_pos_weight)
        
        dask_dfs.append(ptl_ddf)
    
    act_scale_pos_weight = np.average(np.array(all_scal_pos_weight))
    
    return dd.concat(dask_dfs),act_scale_pos_weight
    
if __name__ == "__main__":
    feature_columns = ["p_Scaled_radii","p_Radial_vel","p_Tangential_vel","c_Scaled_radii","c_Radial_vel","c_Tangential_vel"]
    target_column = ["Orbit_infall"]
    
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
             
    combined_name = ""
    for i,sim in enumerate(model_sims):
        pattern = r"(\d+)to(\d+)"
        match = re.search(pattern, sim)

        if match:
            curr_snap_list = [match.group(1), match.group(2)] 
        else:
            print("Pattern not found in the string.")
        parts = sim.split("_")
        combined_name += parts[1] + parts[2] + "s" + parts[4] 
        if i != len(model_sims)-1:
            combined_name += "_"
        
    model_name = model_type + "_" + combined_name + "nu" + nu_string

    model_save_loc = path_to_xgboost + combined_name + "/" + model_name + "/"
    gen_plot_save_loc = model_save_loc + "plots/"
    create_directory(model_save_loc)
    create_directory(gen_plot_save_loc)
     
    if os.path.isfile(model_save_loc + "model_info.pickle"):
        with open(model_save_loc + "model_info.pickle", "rb") as pickle_file:
            model_info = pickle.load(pickle_file)
    else:
        train_sims = ""
        for i,sim in enumerate(model_sims):
            if i != len(model_sims) - 1:
                train_sims += sim + ", "
            else:
                train_sims += sim
        model_info = {
            'Misc Info':{
                'Model trained on': train_sims,
                'Max Radius': search_rad,
                'Nus Used': nu_string,
                }}
    
    # Load previously trained booster or start the training process
    if os.path.isfile(model_save_loc + model_name + ".json"):
        bst = xgb.Booster()
        bst.load_model(model_save_loc + model_name + ".json")
        params = model_info.get('Training Info',{}).get('Training Params')
        print("Loaded Booster")
    else:
        print("Making Datasets")
        train_files = []
        test_files = []
        halo_train_files = []
        train_nus = []
 
        train_data,scale_pos_weight = load_data(client,"Train",rad_cut=train_rad,filter_nu=True)
        test_data,test_scale_pos_weight = load_data(client,"Test")
        
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]

        dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
        dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)

        if on_zaratan == False and do_hpo == True and os.path.isfile(model_save_loc + "used_params.pickle") == False:  
            params = {
            # Parameters that we are going to tune.
            'max_depth':np.arange(2,4,1),
            # 'min_child_weight': 1,
            'learning_rate':np.arange(0.01,1.01,.1),
            'scale_pos_weight':np.arange(scale_pos_weight,scale_pos_weight+10,1),
            'lambda':np.arange(0,3,.5),
            'alpha':np.arange(0,3,.5),
            'subsample': 0.5,
            # 'colsample_bytree': 1,
            }
        
            N_FOLDS = 5
            N_ITER = 25
            
            model = dxgb.XGBClassifier(tree_method='hist', device="cuda", eval_metric="logloss", n_estimators=100, use_label_encoder=False, scale_pos_weight=scale_pos_weight)
            accuracy_wrapper_scorer = make_scorer(accuracy_score_wrapper)
            cuml_accuracy_scorer = make_scorer(accuracy_score, convert_dtype=True)
            print_acc(model, X_train, y_train, X_test, y_test)
            
            mode = "gpu-random"
            #TODO fix so it takes from model_info.pickle
            if os.path.isfile(model_save_loc + "hyper_param_res.pickle") and os.path.isfile(model_save_loc + "hyper_param_results.pickle"):
                with open(model_save_loc + "hyper_param_res.pickle", "rb") as pickle_file:
                    res = pickle.load(pickle_file)
                with open(model_save_loc + "hyper_param_results.pickle", "rb") as pickle_file:
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
                with open(model_save_loc + "hyper_param_res.pickle", "wb") as pickle_file:
                    pickle.dump(res, pickle_file)
                with open(model_save_loc + "hyper_param_results.pickle", "wb") as pickle_file:
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
                "scale_pos_weight":scale_pos_weight,
                "device": "cuda",
                "subsample": 0.5,
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
        bst.save_model(model_save_loc + model_name + ".json")

        plt.figure(figsize=(10,7))
        plt.plot(history["train"]["rmse"], label="Training loss")
        plt.plot(history["test"]["rmse"], label="Validation loss")
        plt.axvline(21, color="gray", label="Optimal tree number")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(gen_plot_save_loc + "training_loss_graph.png")
    
        del dtrain
        del dtest
    
    with timed("Model Evaluation"):
        plot_loc = model_save_loc + "Test_" + combined_name + "/plots/"
        create_directory(plot_loc)
        
        halo_files = []

        halo_dfs = []
        for sim in model_sims:
            halo_dfs.append(reform_df(path_to_calc_info + sim + "/Test/halo_info/"))

        halo_df = pd.concat(halo_dfs)
        
        test_data,test_scale_pos_weight = load_data(client,"Test")
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        eval_model(model_info, client, bst, use_sims=model_sims, dst_type="Test", X=X_test, y=y_test, halo_ddf=halo_df, combined_name=combined_name, plot_save_loc=plot_loc, dens_prf = True, r_rv_tv = True, misclass=True)
   
    bst.save_model(model_save_loc + model_name + ".json")

    feature_important = bst.get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    pos = np.arange(len(keys))

    fig, ax = plt.subplots(1, figsize=(15,10))
    ax.barh(pos,values)
    ax.set_yticks(pos, keys)
    fig.savefig(gen_plot_save_loc + "feature_importance.png")

    with open(model_save_loc + "model_info.pickle", "wb") as pickle_file:
        pickle.dump(model_info, pickle_file) 
    client.close()