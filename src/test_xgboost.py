from dask.distributed import Client
import xgboost as xgb
import pickle
import os
import numpy as np
import json
import multiprocessing as mp
import pandas as pd

from utils.ML_support import get_CUDA_cluster, get_combined_name, parse_ranges, create_nu_string, reform_df, load_data, eval_model
from utils.data_and_loading_functions import create_directory, timed, save_pickle
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")

on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")

ML_dset_path = config["PATHS"]["ML_dset_path"]
path_to_models = config["PATHS"]["path_to_models"]

model_sims = json.loads(config.get("XGBOOST","model_sims"))
dask_task_cpus = config.getint("XGBOOST","dask_task_cpus")
model_type = config["XGBOOST"]["model_type"]
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))

reduce_rad = config.getfloat("XGBOOST","reduce_rad")
reduce_perc = config.getfloat("XGBOOST", "reduce_perc")

weight_rad = config.getfloat("XGBOOST","weight_rad")
min_weight = config.getfloat("XGBOOST","min_weight")
opt_wghts = config.getboolean("XGBOOST","opt_wghts")
opt_scale_rad = config.getboolean("XGBOOST","opt_scale_rad")

dens_prf_plt = config.getboolean("XGBOOST","dens_prf_plt")
misclass_plt = config.getboolean("XGBOOST","misclass_plt")
fulldist_plt = config.getboolean("XGBOOST","fulldist_plt")
io_frac_plt = config.getboolean("XGBOOST","io_frac_plt")
per_err_plt = config.getboolean("XGBOOST","per_err_plt")

nu_splits = config["XGBOOST"]["nu_splits"]
nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

if on_zaratan:
    from dask_mpi import initialize
    from distributed.scheduler import logger
    import socket
elif not on_zaratan and not use_gpu:
    from dask.distributed import LocalCluster

###############################################################################################################

if __name__ == "__main__":
    feature_columns = ["p_Scaled_radii","p_Radial_vel","p_Tangential_vel","c_Scaled_radii","c_Radial_vel","c_Tangential_vel"]
    target_column = ["Orbit_infall"]
    
    if use_gpu:
        mp.set_start_method("spawn")

    if on_zaratan:            
        if use_gpu:
            initialize(local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        else:
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
            else:
                print("SLURM_CPUS_PER_TASK is not defined.")
            initialize(nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")

        print("Initialized")
        client = Client()
        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        login_node_address = "zvladimi@login.zaratan.umd.edu" # Change this to the address/domain of your login node

        logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
    else:
        if use_gpu:
            client = get_CUDA_cluster()
        else:
            tot_ncpus = mp.cpu_count()
            n_workers = int(np.floor(tot_ncpus / dask_task_cpus))
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=dask_task_cpus,
                memory_limit='5GB'  
            )
            client = Client(cluster)
    
    # Adjust name based off what things are being done to the model. This keeps each model unique
    model_comb_name = get_combined_name(model_sims) 
    scale_rad=False
    use_weights=False
    if reduce_rad > 0 and reduce_perc > 0:
        scale_rad = True
    if weight_rad > 0 and min_weight > 0:
        use_weights=True    
    
    model_dir = model_type + "_" + model_comb_name + "nu" + nu_string 
    
    if scale_rad:
        model_dir += "scl_rad" + str(reduce_rad) + "_" + str(reduce_perc)
    if use_weights:
        model_dir += "wght" + str(weight_rad) + "_" + str(min_weight)
    
    model_save_loc = path_to_models + model_comb_name + "/" + model_dir + "/"

    # Try loading the model if it can't be thats an error!
    try:
        bst = xgb.Booster()
        bst.load_model(model_save_loc + model_dir + ".json")
        bst.set_param({"device": "cuda:0"})
        print("Loaded Model Trained on:",model_sims)
    except:
        print("Couldn't load Booster Located at: " + model_save_loc + model_dir + ".json")
    
    # Try loading the model info it it can't that's an error!
    try:
        with open(model_save_loc + "model_info.pickle", "rb") as pickle_file:
            model_info = pickle.load(pickle_file)
    except FileNotFoundError:
        print("Model info could not be loaded please ensure the path is correct or rerun train_xgboost.py")
    
    # Loop through each set of test sims in the user inputted list
    for curr_test_sims in test_sims:
        test_comb_name = get_combined_name(curr_test_sims) 
            
        #TODO check that the right sims and datasets are chosen
        print("Testing on:", curr_test_sims)
        # Loop through and/or for Train/Test/All datasets and evaluate the model
        for dset_name in eval_datasets:
            with timed("Model Evaluation on " + dset_name + " dataset"):             
                plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
                create_directory(plot_loc)
                
                # Load the halo information
                halo_files = []
                halo_dfs = []
                if dset_name == "Full":    
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_df(ML_dset_path + sim + "/" + "Train" + "/halo_info/"))
                        halo_dfs.append(reform_df(ML_dset_path + sim + "/" + "Test" + "/halo_info/"))
                else:
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_df(ML_dset_path + sim + "/" + dset_name + "/halo_info/"))

                halo_df = pd.concat(halo_dfs)
                
                # Load the particle information
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)

                X = data[feature_columns]
                y = data[target_column]

                eval_model(model_info, client, bst, use_sims=curr_test_sims, dst_type=dset_name, X=X, y=y, halo_ddf=halo_df, combined_name=test_comb_name, plot_save_loc=plot_loc,dens_prf=dens_prf_plt,missclass=misclass_plt,\
                    full_dist=fulldist_plt,io_frac=io_frac_plt,per_err=per_err_plt, split_nu=True)
                del data 
                del X
                del y
        
        save_pickle(model_info,model_save_loc + "model_info.pickle")

    client.close()
