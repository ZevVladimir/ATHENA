from dask import array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from contextlib import contextmanager

import xgboost as xgb
from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix
from sklearn.metrics import classification_report
import pickle
import os
import sys
import numpy as np
import json
import re
from colossus.cosmology import cosmology
import multiprocessing as mp

from scipy.optimize import minimize

from utils.ML_support import *
from utils.update_vis_fxns import plot_log_vel
from utils.data_and_loading_functions import create_directory, timed
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
sim_cosmol = config["MISC"]["sim_cosmol"]
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

sim_cosmol = config["MISC"]["sim_cosmol"]
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
p_red_shift = config.getfloat("SEARCH","p_red_shift")
radii_splits = config.get("XGBOOST","rad_splits").split(',')
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
test_halos_ratio = config.getfloat("XGBOOST","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
do_hpo = config.getboolean("XGBOOST","hpo")
# size float32 is 4 bytes
chunk_size = int(np.floor(1e9 / (num_save_ptl_params * 4)))
frac_training_data = 1
model_sims = json.loads(config.get("XGBOOST","model_sims"))
model_type = config["XGBOOST"]["model_type"]
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))

dens_prf_plt = config.getboolean("XGBOOST","dens_prf_plt")
misclass_plt = config.getboolean("XGBOOST","misclass_plt")
fulldist_plt = config.getboolean("XGBOOST","fulldist_plt")
io_frac_plt = config.getboolean("XGBOOST","io_frac_plt")
per_err_plt = config.getboolean("XGBOOST","per_err_plt")

if sim_cosmol == "planck13-nbody":
    cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
else:
    cosmol = cosmology.setCosmology(sim_cosmol) 

if use_gpu:
    from cuml.metrics.accuracy import accuracy_score #TODO fix cupy installation??
    from sklearn.metrics import make_scorer
    import dask_ml.model_selection as dcv
    import cudf
elif not use_gpu and on_zaratan:
    from dask_mpi import initialize
    from mpi4py import MPI
    from distributed.scheduler import logger
    import socket
    #from dask_jobqueue import SLURMCluster
    
    
def find_optimal_params(log_phys_vel, radii, labels):
    # Objective function to minimize (total number of misclassified particles)
    def objective(params):
        slope, intercept = params
        line_y = slope * radii + intercept
        line_preds = np.zeros(radii.size) 
        line_preds[log_phys_vel <= line_y] = 1

        # Calculate misclassifications
        labels_np = labels["Orbit_infall"].values
        num_inc_inf = np.where((line_preds == 1) & (labels_np == 0))[0].shape[0]
        num_inc_orb = np.where((line_preds == 0) & (labels_np == 1))[0].shape[0]

        # Total number of misclassified particles
        tot_num_inc = num_inc_orb + num_inc_inf
        return tot_num_inc

    # Initial guess for slope and intercept
    initial_guess = [-0.8, 0.15]
    
    # Minimize the objective function
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    optimal_params = result.x
    return optimal_params
    
if __name__ == "__main__":
    feature_columns = ["p_Scaled_radii","p_Radial_vel","p_Tangential_vel","c_Scaled_radii","c_Radial_vel","c_Tangential_vel"]
    target_column = ["Orbit_infall"]
    
    if use_gpu:
        mp.set_start_method("spawn")

    if not use_gpu and on_zaratan:
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
        
    # model_name =  model_dir + model_comb_name
    
    model_save_loc = path_to_xgboost + model_comb_name + "/" + model_dir + "/"

    try:
        bst = xgb.Booster()
        bst.load_model(model_save_loc + model_dir + ".json")
        bst.set_param({"device": "cuda:0"})
        print("Loaded Model Trained on:",model_sims)
    except:
        print("Couldn't load Booster Located at: " + model_save_loc + model_dir + ".json")
        
    try:
        with open(model_save_loc + "model_info.pickle", "rb") as pickle_file:
            model_info = pickle.load(pickle_file)
    except FileNotFoundError:
        print("Model info could not be loaded please ensure the path is correct or rerun train_xgboost.py")
        
    for curr_test_sims in test_sims:
        test_comb_name = get_combined_name(curr_test_sims) 
            
        #TODO check that the right sims and datasets are chosen
        print("Testing on:", curr_test_sims)
        for dset_name in eval_datasets:
            with timed("Model Evaluation on " + dset_name + " dataset"):             
                plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
                create_directory(plot_loc)
                
                halo_files = []
                halo_dfs = []
                if dset_name == "Full":    
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + "Train" + "/halo_info/"))
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + "Test" + "/halo_info/"))
                else:
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + dset_name + "/halo_info/"))

                halo_df = pd.concat(halo_dfs)
                
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)

                X_df = data[feature_columns]
                y_df = data[target_column]
                
       
        phys_vel = X_df["p_Radial_vel"]**2 + X_df["p_Tangential_vel"]**2
        log_phys_vel = np.log10(phys_vel)
        log_phys_vel = log_phys_vel.compute()
        radii = X_df["p_Scaled_radii"].compute()
        labels = y_df.compute()
        opt_params = find_optimal_params(log_phys_vel,radii,labels)
        
        plot_log_vel(log_phys_vel,radii,labels,plot_loc,add_line=opt_params)