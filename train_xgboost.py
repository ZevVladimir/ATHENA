from dask import array as da
from dask.distributed import Client
import dask.dataframe as dd


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
import subprocess
from colossus.cosmology import cosmology


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
config.read(os.getcwd() + "/config.ini")
rand_seed = config.getint("MISC","random_seed")
on_zaratan = config.getboolean("MISC","on_zaratan")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
sim_cosmol = config["MISC"]["sim_cosmol"]

path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
else:
    sim_pat = r"cbol_l(\d+)_n(\d+)"
match = re.search(sim_pat, curr_sparta_file)
if match:
    sparta_name = match.group(0)
path_to_hdf5_file = path_to_SPARTA_data + sparta_name + "/" + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
path_to_xgboost = config["PATHS"]["path_to_xgboost"]

model_sims = json.loads(config.get("XGBOOST","model_sims"))

model_type = config["XGBOOST"]["model_type"]
do_hpo = config.getboolean("XGBOOST","hpo")
frac_training_data = config.getfloat("XGBOOST","frac_train_data")
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))
train_rad = config.getint("XGBOOST","training_rad")
nu_splits = config["XGBOOST"]["nu_splits"]
retrain = config.getint("XGBOOST","retrain")

reduce_rad = config.getfloat("XGBOOST","reduce_rad")
reduce_perc = config.getfloat("XGBOOST", "reduce_perc")

weight_rad = config.getfloat("XGBOOST","weight_rad")
min_weight = config.getfloat("XGBOOST","min_weight")
opt_wghts = config.getboolean("XGBOOST","opt_wghts")
opt_scale_rad = config.getboolean("XGBOOST","opt_scale_rad")

dens_prf_plt = config.getboolean("XGBOOST","dens_prf_plt")
misclass_plt = config.getboolean("XGBOOST","misclass_plt")
fulldist_plt = config.getboolean("XGBOOST","fulldist_plt")
per_err_plt = config.getboolean("XGBOOST","per_err_plt")

nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

if sim_cosmol == "planck13-nbody":
    cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
else:
    cosmol = cosmology.setCosmology(sim_cosmol) 

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

if __name__ == "__main__":
    feat_cols = ["p_Scaled_radii","p_Radial_vel","p_Tangential_vel","c_Scaled_radii","c_Radial_vel","c_Tangential_vel"]
    tar_col = ["Orbit_infall"]
    
    if on_zaratan:
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            print("SLURM_CPUS_PER_TASK is not defined.")
        if use_gpu:
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
    
    scale_rad=False
    use_weights=False
    
    if reduce_rad > 0 and reduce_perc > 0 and not opt_scale_rad:
        scale_rad = True
    if weight_rad > 0 and min_weight > 0 and not opt_wghts:
        use_weights=True
        
    combined_name = get_combined_name(model_sims) 
        
    model_dir = model_type + "_" + combined_name + "nu" + nu_string 

    if scale_rad:
        model_dir += "scl_rad" + str(reduce_rad) + "_" + str(reduce_perc)
    if use_weights:
        model_dir += "wght" + str(weight_rad) + "_" + str(min_weight)
        
    # model_name =  model_dir + combined_name

    model_save_loc = path_to_xgboost + combined_name + "/" + model_dir + "/"
    gen_plot_save_loc = model_save_loc + "plots/"
     
    if os.path.isfile(model_save_loc + "model_info.pickle") and retrain < 2:
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
                'Nus Used': nu_string,
                }}
    
    # Load previously trained booster or start the training process
    if os.path.isfile(model_save_loc + model_dir + ".json") and retrain < 1:
        bst = xgb.Booster()
        bst.load_model(model_save_loc + model_dir + ".json")
        params = model_info.get('Training Info',{}).get('Training Params')
        print("Loaded Booster")
    else:
        print("Making Datasets") 
            
        scale_rad_info = {
            "Used scale_rad": scale_rad,
            "Reducing radius start": reduce_rad,
            "Reducing radius percent": reduce_perc,
        }
            
        weight_rad_info = {
            "Used weighting by radius": use_weights,
            "Optimized weights": opt_wghts,
            "Start for weighting": weight_rad,
            "Minimum weight assign": min_weight
        }
            
        #TODO add if statements for if scale rad or weight rad 
        if scale_rad and use_weights:
            train_data,scale_pos_weight,train_weights,bin_edges = load_data(client,model_sims,"Train",scale_rad=scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        elif scale_rad and not use_weights:
            train_data,scale_pos_weight,bin_edges = load_data(client,model_sims,"Train",scale_rad=scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        elif not scale_rad and use_weights:
            train_data,scale_pos_weight,train_weights = load_data(client,model_sims,"Train",scale_rad=scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        else:
            train_data,scale_pos_weight = load_data(client,model_sims,"Train",scale_rad=scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)

        print("scale_pos_weight:",scale_pos_weight)
        X_train = train_data[feat_cols]
        y_train = train_data[tar_col]

        if opt_wghts or opt_scale_rad:
            params = {
                    "verbosity": 0,
                    "tree_method": "hist",
                    "scale_pos_weight":scale_pos_weight,
                    "max_depth":4,
                    "device": "cuda",
                    "subsample": 0.5,
                    'objective': 'binary:logistic',
                    }

            halo_files = []

            halo_dfs = []
            for sim in model_sims:
                halo_dfs.append(reform_df(path_to_calc_info + sim + "/Train/halo_info/"))

            halo_df = pd.concat(halo_dfs)

        if opt_wghts:  
            use_weights = True          
            opt_weight_rad, opt_min_weight, opt_weight_exp = optimize_weights(client,params,train_data,halo_df,model_sims,feat_cols,tar_col)
            
            train_weights = weight_by_rad(X_train["p_Scaled_radii"].values.compute(),y_train.compute().values.flatten(), opt_weight_rad, opt_min_weight)
            dask_weights = []
            scatter_weight = client.scatter(train_weights)
            dask_weight = dd.from_delayed(scatter_weight) 
            dask_weights.append(dask_weight)
                
            train_weights = dd.concat(dask_weights)
            
            train_weights = train_weights.repartition(npartitions=X_train.npartitions)
            
            model_dir += "wght" + str(np.round(opt_weight_rad,3)) + "_" + str(np.round(opt_min_weight,3)) + "_" + str(np.round(opt_weight_exp,3))
            # model_dir += "wght" + str(np.round(opt_weight_rad,3)) + "_" + str(np.round(opt_min_weight,3)) + "_" + str(np.round(opt_weight_exp,3))
            weight_rad_info = {
                "Used weighting by radius": use_weights,
                "Optimized weights": opt_wghts,
                "Start for weighting": opt_weight_rad,
                "Minimum weight assign": opt_min_weight,
                "Weight exponent": opt_weight_exp
            }
        
        if opt_scale_rad:
            scale_rad = True
            num_bins=100
            bin_edges = np.logspace(np.log10(0.001),np.log10(10),num_bins)
            opt_reduce_rad, opt_reduce_perc = optimize_scale_rad(client,params,train_data,halo_df,model_sims,feat_cols,tar_col)
            scld_data = scale_by_rad(train_data.compute(),bin_edges,opt_reduce_rad,opt_reduce_perc)
            
            scatter_train = client.scatter(scld_data)
            scld_train_data = dd.from_delayed(scatter_train)
            
            X_train = scld_train_data[feat_cols]
            y_train = scld_train_data[tar_col]
            
            # model_name += "scl_rad" + str(np.round(opt_reduce_rad,3)) + "_" + str(np.round(opt_reduce_perc,3))
            model_dir += "scl_rad" + str(np.round(opt_reduce_rad,3)) + "_" + str(np.round(opt_reduce_perc,3))
            scale_rad_info = {
            "Used scale_rad": scale_rad,
            "Optimized Scale by Radius": opt_scale_rad,
            "Reducing radius start": opt_reduce_rad,
            "Reducing radius percent": opt_reduce_perc,
            }
        
        model_info["Training Data Adjustments"] = {
            "Scaling by Radii": scale_rad_info,
            "Weighting by Radii": weight_rad_info,
        }
        
        model_save_loc = path_to_xgboost + combined_name + "/" + model_dir + "/"
        gen_plot_save_loc = model_save_loc + "plots/"
        create_directory(model_save_loc)
        create_directory(gen_plot_save_loc)
        
        plot_orb_inf_dist(50, train_data["p_Scaled_radii"].values.compute(), train_data["Orbit_infall"].values.compute(), gen_plot_save_loc)
        
        if use_weights:
            fig,ax = plt.subplots(1)
            ax.hist(train_weights.values.compute())
            fig.savefig(path_to_MLOIS + "Random_figs/weight_dist.png")
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train, weight=train_weights)
        else:
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
        
        if 'Training Info' in model_info: 
            params = model_info.get('Training Info',{}).get('Training Params')
        else:
            params = {
                "verbosity": 1,
                "tree_method": "hist",
                "scale_pos_weight":scale_pos_weight,
                "max_depth":4,
                "device": "cuda",
                "subsample": 0.5,
                'objective': 'binary:logistic',
                }
            model_info['Training Info']={
                'Fraction of Training Data Used': frac_training_data,
                'Training Params': params}
        with timed("Trained Model"):
            print("Starting train using params:", params)
            output = dxgb.train(
                client,
                params,
                dtrain,
                num_boost_round=100,
                # evals=[(dtrain, "train"),(dtest, "test")],
                evals=[(dtrain, "train")],
                early_stopping_rounds=10,      
                )
            bst = output["booster"]
            history = output["history"]
            bst.save_model(model_save_loc + model_dir + ".json")
            with open(model_save_loc + "model_info.pickle", "wb") as pickle_file:
                pickle.dump(model_info, pickle_file)

            plt.figure(figsize=(10,7))
            plt.plot(history["train"]["logloss"], label="Training loss")
            plt.xlabel("Number of trees")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(gen_plot_save_loc + "training_loss_graph.png") 
        
            del dtrain
            # del dtest
    if not on_zaratan:
        xgb.plot_tree(bst, num_trees=10, rankdir='LR')
        fig = plt.gcf()
        fig.set_size_inches(150, 100)

        fig.savefig(gen_plot_save_loc + "ex_tree.png",bbox_inches="tight")
        
    with timed("Model Evaluation"):
        plot_loc = model_save_loc + "Test_" + combined_name + "/plots/"
        create_directory(plot_loc)
        
        halo_files = []

        halo_dfs = []
        for sim in model_sims:
            halo_dfs.append(reform_df(path_to_calc_info + sim + "/Test/halo_info/"))

        halo_df = pd.concat(halo_dfs)
        
        test_data,test_scale_pos_weight = load_data(client,model_sims,"Test",limit_files=False)
        X_test = test_data[feat_cols]
        y_test = test_data[tar_col]
        
        eval_model(model_info, client, bst, use_sims=model_sims, dst_type="Test", X=X_test, y=y_test, halo_ddf=halo_df, combined_name=combined_name, plot_save_loc=plot_loc, dens_prf=dens_prf_plt,missclass=misclass_plt,full_dist=fulldist_plt,per_err=per_err_plt)
   
    bst.save_model(model_save_loc + model_dir + ".json")

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
