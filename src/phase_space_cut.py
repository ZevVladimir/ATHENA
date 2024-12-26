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
from utils.update_vis_fxns import plot_log_vel, compare_prfs_nu
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
        num_inc_inf = np.where((line_preds == 1) & (labels == 0))[0].shape[0]
        num_inc_orb = np.where((line_preds == 0) & (labels == 1))[0].shape[0]

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
            with timed("Loading data: " + dset_name + " dataset"):             
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
                
       
        phys_vel = np.sqrt(X_df["p_Radial_vel"]**2 + X_df["p_Tangential_vel"]**2)
        log_phys_vel = np.log10(phys_vel)
        
        fltr_condition = X_df["p_Radial_vel"] < 0
        fltr_log_phys_vel = log_phys_vel[fltr_condition].compute()
        fltr_labels = y_df[fltr_condition].compute().values.flatten()
        fltr_radii = X_df["p_Scaled_radii"][fltr_condition].compute()
        
        with timed("Find optimal parameters"):
            opt_params = find_optimal_params(fltr_log_phys_vel,fltr_radii,fltr_labels)
            print(opt_params)
            
        radii = X_df["p_Scaled_radii"].compute().values.flatten()
        radial_vel = X_df["p_Radial_vel"].compute().values.flatten()
        labels = y_df.compute().values.flatten()
        log_phys_vel = log_phys_vel.compute().values.flatten()
        
        with timed("Plot cut"):
            plot_log_vel(log_phys_vel,radii,labels,plot_loc,add_line=opt_params)
        
        slope, intercept = opt_params
        line_y = slope * radii + intercept
        line_preds = np.zeros(radii.size)

        # Orbiting if radial velocity is positive
        # Also orbiting if negative radial velocity but on the left of the line
        # Everything else is infalling
        line_preds[radial_vel > 0] = 1  
        on_left_side = log_phys_vel < line_y  
        line_preds[(radial_vel <= 0) & on_left_side] = 1  
        
        
        halo_first = halo_df["Halo_first"].values
        halo_n = halo_df["Halo_n"].values
        all_idxs = halo_df["Halo_indices"].values

        
        with timed("Prep for dens prf"):
            all_z = []
            all_rhom = []
            # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
            sim_splits = np.where(halo_first == 0)[0]

            # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
            # stacked simulations such that they correspond to the larger dataset and not one specific simulation
            if len(curr_test_sims) > 1:
                for i,sim in enumerate(curr_test_sims):
                    # The first sim remains the same
                    if i == 0:
                        continue
                    # Else if it isn't the final sim 
                    elif i < len(curr_test_sims) - 1:
                        halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
                    # Else if the final sim
                    else:
                        halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
            
            # Get the redshifts for each simulation's primary snapshot
            for i,sim in enumerate(curr_test_sims):
                with open(path_to_calc_info + sim + "/config.pickle", "rb") as file:
                    config_dict = pickle.load(file)
                    curr_z = config_dict["p_snap_info"]["red_shift"][()]
                    all_z.append(curr_z)
                    all_rhom.append(cosmol.rho_m(curr_z))
                    h = config_dict["p_snap_info"]["h"][()]
            
            tot_num_halos = halo_n.shape[0]
            min_disp_halos = int(np.ceil(0.3 * tot_num_halos))
            
            act_mass_prf_all, act_mass_prf_orb,all_masses,bins = load_sprta_mass_prf(sim_splits,all_idxs,curr_test_sims)
            act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb
            
            calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m = create_stack_mass_prf(sim_splits,radii=X_df["p_Scaled_radii"].values.compute(), halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=line_preds, prf_bins=bins, use_mp=True, all_z=all_z)

            # Halos that get returned with a nan R200m mean that they didn't meet the required number of ptls within R200m and so we need to filter them from our calculated profiles and SPARTA profiles 
            small_halo_fltr = np.isnan(calc_r200m)
            act_mass_prf_all[small_halo_fltr,:] = np.nan
            act_mass_prf_orb[small_halo_fltr,:] = np.nan
            act_mass_prf_inf[small_halo_fltr,:] = np.nan

            # Calculate the density by divide the mass of each bin by the volume of that bin's radius
            calc_dens_prf_all = calculate_density(calc_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
            calc_dens_prf_orb = calculate_density(calc_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
            calc_dens_prf_inf = calculate_density(calc_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
            
            act_dens_prf_all = calculate_density(act_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
            act_dens_prf_orb = calculate_density(act_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
            act_dens_prf_inf = calculate_density(act_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        with timed("Split by nu"):
            all_prf_lst = []
            orb_prf_lst = []
            inf_prf_lst = []
            cpy_plt_nu_splits = plt_nu_splits.copy()
            for i,nu_split in enumerate(cpy_plt_nu_splits):
                # Take the second element of the where to filter by the halos (?)
                fltr = np.where((calc_nus > nu_split[0]) & (calc_nus < nu_split[1]))[0]
                if fltr.shape[0] > 25:
                    all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
                    orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
                    inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
                else:
                    plt_nu_splits.remove(nu_split)

        with timed("Compare density profile"):
            compare_prfs_nu(plt_nu_splits,len(cpy_plt_nu_splits),all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_loc,title="ps_cut_dens_")
        with timed("Plot cut"):
            plot_log_vel(log_phys_vel,radii,labels,plot_loc,add_line=opt_params)