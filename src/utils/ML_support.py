from dask import array as da
import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client
import xgboost as xgb
from xgboost import dask as dxgb

import numpy as np
import os
import pickle
import json
import h5py
import re
import pandas as pd
from colossus.lss import peaks
import warnings
from dask.distributed import Client
import multiprocessing as mp

from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import accuracy_score
from functools import partial

from .data_and_loading_functions import load_SPARTA_data, conv_halo_id_spid, timed, split_data_by_halo, parse_ranges, load_pickle, load_config, get_comp_snap_info, create_directory, set_cosmology
from .vis_fxns import plot_full_ptl_dist, plot_miss_class_dist, compare_prfs, compare_split_prfs, inf_orb_frac
from .calculation_functions import create_mass_prf, create_stack_mass_prf, filter_prf, calculate_density, calc_mass_acc_rate, calc_t_dyn
from sparta_tools import sparta 

##################################################################################################################
# LOAD CONFIG PARAMETERS
config_dict = load_config(os.getcwd() + "/config.ini")
rand_seed = config_dict["MISC"]["random_seed"]
curr_sparta_file = config_dict["SPARTA_DATA"]["curr_sparta_file"]
debug_indiv_dens_prf = config_dict["MISC"]["debug_indiv_dens_prf"]
pickle_data = config_dict["MISC"]["pickle_data"]

snap_path = config_dict["SNAP_DATA"]["snap_path"]

SPARTA_output_path = config_dict["SPARTA_DATA"]["sparta_output_path"]
pickled_path = config_dict["PATHS"]["pickled_path"]
ML_dset_path = config_dict["PATHS"]["ml_dset_path"]
debug_plt_path = config_dict["PATHS"]["debug_plt_path"]

on_zaratan = config_dict["DASK_CLIENT"]["on_zaratan"]
use_gpu = config_dict["DASK_CLIENT"]["use_gpu"]
dask_task_ncpus = config_dict["DASK_CLIENT"]["dask_task_ncpus"]

file_lim = config_dict["TRAIN_MODEL"]["file_lim"]

reduce_rad = config_dict["OPTIMIZE"]["reduce_rad"]
reduce_perc = config_dict["OPTIMIZE"]["reduce_perc"]

weight_rad = config_dict["OPTIMIZE"]["weight_rad"]
min_weight = config_dict["OPTIMIZE"]["min_weight"]
weight_exp = config_dict["OPTIMIZE"]["weight_exp"]

hpo_loss = config_dict["OPTIMIZE"]["hpo_loss"]
plt_nu_splits = config_dict["EVAL_MODEL"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)

plt_macc_splits = config_dict["EVAL_MODEL"]["plt_macc_splits"]
plt_macc_splits = parse_ranges(plt_macc_splits)
min_halo_nu_bin = config_dict["EVAL_MODEL"]["min_halo_nu_bin"]
linthrsh = config_dict["EVAL_MODEL"]["linthrsh"]
lin_nbin = config_dict["EVAL_MODEL"]["lin_nbin"]
log_nbin = config_dict["EVAL_MODEL"]["log_nbin"]
lin_rvticks = config_dict["EVAL_MODEL"]["lin_rvticks"]
log_rvticks = config_dict["EVAL_MODEL"]["log_rvticks"]
lin_tvticks = config_dict["EVAL_MODEL"]["lin_tvticks"]
log_tvticks = config_dict["EVAL_MODEL"]["log_tvticks"]
lin_rticks = config_dict["EVAL_MODEL"]["lin_rticks"]
log_rticks = config_dict["EVAL_MODEL"]["log_rticks"]

# From the input simulation name extract the simulation name (ex: cbol_l0063_n0256) and the SPARTA hdf5 output name (ex: cbol_l0063_n0256_4r200m_1-5v200m)
def split_sparta_hdf5_name(sim):
    # Get just the sim name of the form cbol_ (or cpla_) then the size of the box lxxxx and the number of particles in it nxxxx
    sim_pat = r"cbol_l(\d+)_n(\d+)"
    match = re.search(sim_pat, sim)
    if not match:
        sim_pat = r"cpla_l(\d+)_n(\d+)"
        match = re.search(sim_pat,sim)
        
    if match:
        sim_name = match.group(0)
           
    # now get the full name that includes the search radius in R200m and the velocity limit in v200m
    sim_search_pat = sim_pat + r"_(\d+)r200m_(\d+)v200m"
    name_match = re.search(sim_search_pat, sim)
    
    # also check if there is a decimal for v200m
    if not name_match:
        sim_search_pat = sim_pat + r"_(\d+)r200m_(\d+)-(\d+)v200m"
        name_match = re.search(sim_search_pat, sim)
    
    if name_match:
        search_name = name_match.group(0)
        
    if not name_match and not match:
        print("Couldn't read sim name correctly:",sim)
        print(match)
    
    return sim_name, search_name

sim_name, search_name = split_sparta_hdf5_name(curr_sparta_file)
snap_path = snap_path + sim_name + "/"

###############################################################################################################
if on_zaratan:
    from dask_mpi import initialize
    from distributed.scheduler import logger
    import socket
elif not on_zaratan and not use_gpu:
    from dask.distributed import LocalCluster
elif use_gpu:
    from dask_cuda import LocalCUDACluster
###############################################################################################################

# Instantiate a dask cluster with GPUs
def get_CUDA_cluster():
    cluster = LocalCUDACluster(
                               device_memory_limit='10GB',
                               jit_unspill=True)
    client = Client(cluster)
    return client

def setup_client():
    if use_gpu:
        mp.set_start_method("spawn")

    if on_zaratan:            
        if use_gpu:
            initialize(local_directory = "/home/zvladimi/scratch/ATHENA/dask_logs/")
        else:
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
            else:
                print("SLURM_CPUS_PER_TASK is not defined.")
            initialize(nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/ATHENA/dask_logs/")

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
            n_workers = int(np.floor(tot_ncpus / dask_task_ncpus))
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=dask_task_ncpus,
                memory_limit='5GB'  
            )
            client = Client(cluster)
    return client

# Make predictions using the model. Requires the inputs to be a dask dataframe. Can either return the predictions still as a dask dataframe or as a numpy array
def make_preds(client, bst, X, dask = False, threshold = 0.5):
    if dask:
        preds = dxgb.predict(client,bst,X)
        preds = preds.map_partitions(lambda df: (df >= threshold).astype(int))
        return preds
    else:
        preds = dxgb.inplace_predict(client, bst, X).compute()
        preds = (preds >= threshold).astype(np.int8)
    
    return preds

def extract_snaps(sim_name):
    match = re.search(r'((?:_\d+)+)$', sim_name)
    if not match:
        return []
    number_strs = match.group(1).split('_')[1:]  # First split is an empty string
    return [int(num) for num in number_strs]

def get_feature_labels(features, tdyn_steps):
    all_features = []
    for feature in features:
        all_features.append("p_" + feature)
    for t_dyn_step in tdyn_steps:
        for feature in features:
            all_features.append(str(t_dyn_step) + "_" + feature)
    
    return all_features

# This function prints out all the model information such as the training simulations, training parameters, and results
# The results are split by simulation that the model was tested on and reports the misclassification rate on each population
def print_model_prop(model_dict, indent=''):
    # If using from command line and passing path to the pickled dictionary instead of dict load the dict from the file path
    if isinstance(model_dict, str):
        with open(model_dict, "rb") as file:
            model_dict = pickle.load(file)
        
    for key, value in model_dict.items():
        # use recursion for dictionaries within the dictionary
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            print_model_prop(value, indent + '  ')
        # if the value is instead a list join them all together with commas
        elif isinstance(value, list):
            print(f"{indent}{key}: {', '.join(map(str, value))}")
        else:
            print(f"{indent}{key}: {value}")

def get_model_name(model_type, trained_on_sims, hpo_done=False, opt_param_dict=None):
    model_name = f"{model_type}_{get_combined_name(trained_on_sims)}"
    
    if hpo_done and opt_param_dict:
        param_str = "_".join(f"{k}_{v}" for k, v in opt_param_dict.items())
        model_name += f"_hpo_{param_str}"
        
    return model_name

# Convert a simulation's name to where the primary snapshot location is in the pickled data (ex: cbol_l0063_n0256_4r200m_1-5v200m_190to166 -> 190_cbol_l0063_n0256_4r200m_1-5v200m)
def get_pickle_path_for_sim(input_str):
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

# Use a function to assign weights for XGBoost training for each particle based on their radii
def weight_by_rad(radii,orb_inf,use_weight_rad=weight_rad,use_min_weight=min_weight,use_weight_exp=weight_exp,weight_inf=False,weight_orb=True):
    weights = np.ones(radii.shape[0])

    if weight_inf and weight_orb:
        mask = (radii > use_weight_rad)
    elif weight_orb:
        mask = (radii > use_weight_rad) & (orb_inf == 1)
    elif weight_inf:
        mask = (radii > use_weight_rad) & (orb_inf == 0)
    else:
        print("No weights calculated. Make sure to set weight_inf and/or weight_orb = True")
        return pd.DataFrame(weights)
    weights[mask] = (np.exp((np.log(use_min_weight)/(np.max(radii)-use_weight_rad)) * (radii[mask]-use_weight_rad)))**use_weight_exp

    return pd.DataFrame(weights)

# For each radial bin beyond an inputted radius randomly reduce the number of particles present to a certain percentage of the number of particles within that inptuted radius
def scale_by_rad(data,bin_edges,use_red_rad=reduce_rad,use_red_perc=reduce_perc):
    radii = data["p_Scaled_radii"].values
    max_ptl = int(np.floor(np.where(radii<use_red_rad)[0].shape[0] * use_red_perc))
    filter_data = []
    
    for i in range(len(bin_edges) - 1):
        bin_mask = (radii >= bin_edges[i]) & (radii < bin_edges[i+1])
        bin_data = data[bin_mask]
        if len(bin_data) > int(max_ptl) and bin_edges[i] > 1:
            bin_data = bin_data.sample(n=max_ptl,replace=False)
        filter_data.append(bin_data)
    
    filter_data = pd.concat(filter_data) 
    
    return filter_data

# Returns a simulation's mass used and the redshift of the primary snapshot
def sim_mass_p_z(sim,config_params):
    sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
    
    curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" +  sparta_search_name + ".hdf5"    
    
    with h5py.File(curr_sparta_HDF5_path,"r") as f:
        dic_sim = {}
        grp_sim = f['simulation']
        for f in grp_sim.attrs:
            dic_sim[f] = grp_sim.attrs[f]
    
    p_red_shift = config_params["all_snap_info"]["prime_snap_info"]["red_shift"]
    
    param_paths = [["simulation","particle_mass"]]
            
    sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name)
    ptl_mass = sparta_params[sparta_param_names[0]]
    
    return ptl_mass, p_red_shift

# Returns an inputted dataframe with only the halos that fit within the inputted ranges of nus (peak height)
def filter_df_with_nus(df,nus,halo_first,halo_n, nu_splits):    
    # First masks which halos are within the inputted nu ranges
    mask = pd.Series([False] * nus.shape[0])

    for start, end in nu_splits:
        mask[np.where((nus >= start) & (nus <= end))[0]] = True
    
    # Then get the indices of all the particles that belong to these halos and combine them into another mask which returns only the wanted particles    
    halo_n = halo_n[mask]
    halo_first = halo_first[mask]
    halo_last = halo_first + halo_n
 
    use_idxs = np.concatenate([np.arange(start, end) for start, end in zip(halo_first, halo_last)])

    return df.iloc[use_idxs], halo_n, halo_first

# Goes through a folder where a dataset's hdf5 files are stored and reforms them into one pandas dataframe (in order)
def reform_dataset_dfs(folder_path):
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

# Sorts a dataset's .h5 files such that they are in ascending numerical order and if desired can return a limited number of them based off the file_lim parameter in the config file
def sort_and_lim_files(folder_path,limit_files=False):
    hdf5_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.h5'):
            hdf5_files.append(f)
    hdf5_files.sort()
    if file_lim > 0 and file_lim < len(hdf5_files) and limit_files:
        hdf5_files = hdf5_files[:file_lim]
    return hdf5_files
    
# Split a dataframe so that each one is below an inputted maximum memory size
def split_dataframe(df, max_size, weights=None, use_weights = False):
    total_size = df.memory_usage(index=True).sum()
    num_splits = int(np.ceil(total_size / max_size))
    chunk_size = int(np.ceil(len(df) / num_splits))
    print("splitting Dataframe into:",num_splits,"dataframes")
    
    split_dfs = []
    split_weights = []
    for i in range(0, len(df), chunk_size):
        split_dfs.append(df.iloc[i:i + chunk_size])
        if use_weights:
            split_weights.append(weights[i:i+chunk_size])
    
    if use_weights:
        return split_dfs, split_weights
    else:
        return split_dfs

# Function to process a file in a dataset's folder: combines them all, performs any desired filtering, calculates weights if desired, and calculates scaled position weight
# Also splits the dataframe into smaller dataframes based of inputted maximum memory size
def process_file(folder_path, file_index, ptl_mass, use_z, bin_edges, max_mem, sim_cosmol, filter_nu, scale_rad, use_weights, nu_splits):
    @delayed
    def delayed_task():
        cosmol = set_cosmology(sim_cosmol)
        # Get all the snap folders being used
        all_snap_fldrs = []
        for snap_fldr in os.listdir(folder_path + "/ptl_info/"):
            if os.path.isdir(os.path.join(folder_path + "/ptl_info/", snap_fldr)):
                all_snap_fldrs.append(snap_fldr)
        
        # Since they are just numbers we can first sort them and then sort them in descending order (primary snaps should always be the largest value)
        all_snap_fldrs.sort()
        all_snap_fldrs.reverse()
        
        # Stack column wise all the snapshots for each particle file
        ptl_df_list = []
        for snap_fldr in all_snap_fldrs:
            ptl_path = f"{folder_path}/ptl_info/{snap_fldr}/ptl_{file_index}.h5"
            ptl_df_list.append(pd.read_hdf(ptl_path))
        ptl_df = pd.concat(ptl_df_list,axis=1)

        halo_path = f"{folder_path}/halo_info/halo_{file_index}.h5"
        halo_df = pd.read_hdf(halo_path)

        # reset indices for halo_first halo_n indexing
        halo_df["Halo_first"] = halo_df["Halo_first"] - halo_df["Halo_first"][0]
        
        # Calculate peak heights for each halo
        nus = np.array(peaks.peakHeight((halo_df["Halo_n"][:] * ptl_mass), use_z))
        
        # Filter by nu and/or by radius
        if filter_nu:
            ptl_df, upd_halo_n, upd_halo_first = filter_df_with_nus(ptl_df, nus, halo_df["Halo_first"], halo_df["Halo_n"], nu_splits)
        if scale_rad:
            ptl_df = scale_by_rad(ptl_df,bin_edges)
            
        weights = (
            weight_by_rad(ptl_df["p_Scaled_radii"].values, ptl_df["Orbit_infall"].values, 
                          weight_inf=False, weight_orb=True) if use_weights else None
        )

        # Calculate scale position weight
        scal_pos_weight = calc_scal_pos_weight(ptl_df)

        # If the dataframe is too large split it up
        if ptl_df.memory_usage(index=True).sum() > max_mem:
            ptl_dfs = split_dataframe(ptl_df, max_mem)
            if use_weights:
                ptl_dfs, weights = split_dataframe(ptl_df,max_mem,weights,use_weights=True)
        else:
            ptl_dfs = [ptl_df]
            if use_weights:
                weights = [weights]
        
        return ptl_dfs,scal_pos_weight,weights
    return delayed_task()

# Combines the results of the processing of each file in the folder into one dataframe for the data and list for the scale position weights and an array of weights if desired
def combine_results(results, client, use_weights):
    # Unpack the results
    ddfs,scal_pos_weights,dask_weights = [], [], []
    
    for res in results:
        ddfs.extend(res[0])
        scal_pos_weights.append(res[1]) # We append since scale position weight is just a number
        if use_weights:
            dask_weights.extend(res[2])
            
    all_ddfs = dd.concat([dd.from_delayed(client.scatter(df)) for df in ddfs])
    if use_weights:
        all_weights = dd.concat([dd.from_delayed(client.scatter(w)) for w in dask_weights])
        return all_ddfs, scal_pos_weights, all_weights

    return all_ddfs, scal_pos_weights

# Combines all the files in a dataset's folder into one dask dataframe and a list for the scale position weights and an array of weights if desired 
def reform_datasets_nested(client,ptl_mass,use_z,max_mem,bin_edges,sim_cosmol,folder_path,scale_rad=False,use_weights=False,filter_nu=None,limit_files=False,nu_splits=None):
    snap_n_files = len(os.listdir(folder_path + "/ptl_info/"))
    n_files = np.min([snap_n_files,file_lim]) 
    
    delayed_results = []
    for file_index in range(n_files):
        # Create delayed tasks for each file
        delayed_results.append(process_file(
                folder_path, file_index, ptl_mass, use_z, bin_edges,
                max_mem, sim_cosmol, filter_nu, scale_rad, use_weights, nu_splits))
    
    # Compute the results in parallel
    results = client.compute(delayed_results, sync=True)

    return combine_results(results, client, use_weights)
    
# Calculates the scaled position weight for a dataset. Which is used to weight the model towards the population with less particles (should be the orbiting population)
def calc_scal_pos_weight(df):
    count_negatives = (df['Orbit_infall'] == 0).sum()
    count_positives = (df['Orbit_infall'] == 1).sum()

    scale_pos_weight = count_negatives / count_positives
    return scale_pos_weight

# Loads all the data for the inputted list of simulations into one dataframe. Finds the scale position weight for the dataset and any adjusted weighting for it if desired
def load_data(client, sims, dset_name, sim_cosmol, bin_edges = None, limit_files = False, scale_rad=False, use_weights=False, filter_nu=False, nu_splits=None):
    dask_dfs = []
    all_scal_pos_weight = []
    all_weights = []
        
    for sim in sims:
        with open(ML_dset_path + sim + "/dset_params.pickle","rb") as f:
            dset_params = pickle.load(f)
        # Get mass and redshift for this simulation
        ptl_mass, use_z = sim_mass_p_z(sim,dset_params)
        max_mem = int(np.floor(dset_params["HDF5 Mem Size"] / 2))
        
        if dset_name == "Full":
            datasets = ["Train", "Test"]
        else:
            datasets = [dset_name]

        for dataset in datasets:
            with timed(f"Reformed {dataset} Dataset: {sim}"): 
                dataset_path = f"{ML_dset_path}{sim}/{dataset}"
                if use_weights:
                    ptl_ddf,sim_scal_pos_weight, weights = reform_datasets_nested(client,ptl_mass,use_z,max_mem,bin_edges,sim_cosmol,dataset_path,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files,nu_splits=nu_splits)  
                    all_weights.append(weights)
                else:
                    ptl_ddf,sim_scal_pos_weight = reform_datasets_nested(client,ptl_mass,use_z,max_mem,bin_edges,sim_cosmol,dataset_path,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files,nu_splits=nu_splits)  
                all_scal_pos_weight.append(sim_scal_pos_weight)
                dask_dfs.append(ptl_ddf)
                    
    all_scal_pos_weight = np.average(np.concatenate([np.array(sublist).flatten() for sublist in all_scal_pos_weight]))
    act_scale_pos_weight = np.average(all_scal_pos_weight)

    all_dask_dfs = dd.concat(dask_dfs)
    
    if use_weights:
        return all_dask_dfs,act_scale_pos_weight, dd.concat(all_weights)
    else:
        return all_dask_dfs,act_scale_pos_weight

# Reconstruct SPARTA's mass profiles and stack them together for a list of sims
def load_sparta_mass_prf(sim_splits,all_idxs,use_sims,ret_r200m=False):                
    mass_prf_all_list = []
    mass_prf_1halo_list = []
    all_r200m_list = []
    all_masses = []
    
    for i,sim in enumerate(use_sims):
        # Get the halo indices corresponding to this simulation
        if i < len(use_sims) - 1:
            use_idxs = all_idxs[sim_splits[i]:sim_splits[i+1]]
        else:
            use_idxs = all_idxs[sim_splits[i]:]
        
        sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
        # find the snapshots for this simulation
        curr_snap_list = extract_snaps(sim)
        
        with open(ML_dset_path + sim + "/dset_params.pickle", "rb") as file:
            dset_params = pickle.load(file)
            
        p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"]
        
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"      
        
        param_paths = [["halos","id"],["simulation","particle_mass"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name)
        halos_ids = sparta_params[sparta_param_names[0]][:,p_sparta_snap]
        ptl_mass = sparta_params[sparta_param_names[1]]
 
        use_halo_ids = halos_ids[use_idxs]
        #TODO don't use sparta.load anymore
        sparta_output = sparta.load(filename=curr_sparta_HDF5_path, halo_ids=use_halo_ids, log_level=0)
        new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta re-sort the indices
        
        mass_prf_all_list.append(sparta_output['anl_prf']['M_all'][new_idxs,p_sparta_snap,:])
        mass_prf_1halo_list.append(sparta_output['anl_prf']['M_1halo'][new_idxs,p_sparta_snap,:])

        all_r200m_list.append(sparta_output['halos']['R200m'][:,p_sparta_snap])

        all_masses.append(ptl_mass)

    mass_prf_all = np.vstack(mass_prf_all_list)
    mass_prf_1halo = np.vstack(mass_prf_1halo_list)
    all_r200m = np.concatenate(all_r200m_list)
    
    bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
    bins = np.insert(bins, 0, 0)

    if ret_r200m:
        return mass_prf_all,mass_prf_1halo,all_masses,bins,all_r200m
    else:
        return mass_prf_all,mass_prf_1halo,all_masses,bins

# Evaluate an input model by generating plots of comparisons between the model's predictions and SPARTA
def eval_model(model_info, preds, use_sims, dst_type, X, y, halo_ddf, sim_cosmol, plot_save_loc, all_tdyn_steps, dens_prf = False,missclass=False,full_dist=False,io_frac=False,split_nu=False,split_macc=False): 
    cosmol = set_cosmology(sim_cosmol)        
        
    if pickle_data:
        plt_data_loc = plot_save_loc + "pickle_plt_data/"
        create_directory(plt_data_loc)
    
    num_bins = 50

    # Generate a comparative density profile
    if dens_prf:
        halo_first = halo_ddf["Halo_first"].values
        halo_n = halo_ddf["Halo_n"].values
        all_idxs = halo_ddf["Halo_indices"].values

        all_z = []
        all_rhom = []
        # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
        sim_splits = np.where(halo_first == 0)[0]

        # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
        # stacked simulations such that they correspond to the larger dataset and not one specific simulation
        if len(use_sims) > 1:
            for i,sim in enumerate(use_sims):
                # The first sim remains the same
                if i == 0:
                    continue
                # Else if it isn't the final sim 
                elif i < len(use_sims) - 1:
                    halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
                # Else if the final sim
                else:
                    halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
        
        # Get the redshifts for each simulation's primary snapshot
        for i,sim in enumerate(use_sims):
            with open(ML_dset_path + sim + "/dset_params.pickle", "rb") as file:
                dset_params = pickle.load(file)
                curr_z = dset_params["all_snap_info"]["prime_snap_info"]["red_shift"][()]
                curr_rho_m = dset_params["all_snap_info"]["prime_snap_info"]["rho_m"][()]
                all_z.append(curr_z)
                all_rhom.append(curr_rho_m)
                h = dset_params["all_snap_info"]["prime_snap_info"]["h"][()]
        
        tot_num_halos = halo_n.shape[0]
        min_disp_halos = int(np.ceil(0.3 * tot_num_halos))
        
        # Get SPARTA's mass profiles
        act_mass_prf_all, act_mass_prf_orb,all_masses,bins = load_sparta_mass_prf(sim_splits,all_idxs,use_sims)
        act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb
        
        snap_list = extract_snaps(use_sims[0])
        
        # Create mass profiles from the model's predictions
        prime_radii = X["p_Scaled_radii"].values.compute()
        calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m = create_stack_mass_prf(sim_splits,radii=prime_radii, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds.values, prf_bins=bins, use_mp=True, all_z=all_z)
        my_mass_prf_all, my_mass_prf_orb, my_mass_prf_inf, my_nus, my_r200m = create_stack_mass_prf(sim_splits,radii=prime_radii, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=y.compute().values.flatten(), prf_bins=bins, use_mp=True, all_z=all_z)
        # Halos that get returned with a nan R200m mean that they didn't meet the required number of ptls within R200m and so we need to filter them from our calculated profiles and SPARTA profiles 
        small_halo_fltr = np.isnan(calc_r200m)
        act_mass_prf_all[small_halo_fltr,:] = np.nan
        act_mass_prf_orb[small_halo_fltr,:] = np.nan
        act_mass_prf_inf[small_halo_fltr,:] = np.nan
        
        all_prfs = [calc_mass_prf_all, act_mass_prf_all]
        orb_prfs = [calc_mass_prf_orb, act_mass_prf_orb]
        inf_prfs = [calc_mass_prf_inf, act_mass_prf_inf]

        # Calculate the density by divide the mass of each bin by the volume of that bin's radius
        calc_dens_prf_all = calculate_density(calc_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        calc_dens_prf_orb = calculate_density(calc_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        calc_dens_prf_inf = calculate_density(calc_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        
        act_dens_prf_all = calculate_density(act_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        act_dens_prf_orb = calculate_density(act_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        act_dens_prf_inf = calculate_density(act_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        
        if debug_indiv_dens_prf > 0:
            my_dens_prf_orb = calculate_density(my_mass_prf_orb*h,bins[1:],my_r200m*h,sim_splits,all_rhom)
            my_dens_prf_all = calculate_density(my_mass_prf_all*h,bins[1:],my_r200m*h,sim_splits,all_rhom)
            my_dens_prf_inf = calculate_density(my_mass_prf_inf*h,bins[1:],my_r200m*h,sim_splits,all_rhom)
        
            ratio = np.where(act_dens_prf_all != 0, calc_dens_prf_all / act_dens_prf_all, np.nan)

            # Compute the difference for each halo (using range: max - min)
            diff = np.nanmax(ratio, axis=1) - np.nanmin(ratio, axis=1)

            # If you want the top k halos with the largest differences, use:
            k = 5  # Example value
            big_halo_loc = np.argsort(diff)[-k:]
        
            for i in range(k):
                all_prfs = [my_mass_prf_all[big_halo_loc[i]], act_mass_prf_all[big_halo_loc[i]]]
                orb_prfs = [my_mass_prf_orb[big_halo_loc[i]], act_mass_prf_orb[big_halo_loc[i]]]
                inf_prfs = [my_mass_prf_inf[big_halo_loc[i]], act_mass_prf_inf[big_halo_loc[i]]]
                compare_prfs(all_prfs,orb_prfs,inf_prfs,bins[1:],lin_rticks,debug_plt_path,sim + "_" + str(i)+"_mass",prf_func=None)

            for i in range(k):
                all_prfs = [my_dens_prf_all[big_halo_loc[i]], act_dens_prf_all[big_halo_loc[i]]]
                orb_prfs = [my_dens_prf_orb[big_halo_loc[i]], act_dens_prf_orb[big_halo_loc[i]]]
                inf_prfs = [my_dens_prf_inf[big_halo_loc[i]], act_dens_prf_inf[big_halo_loc[i]]]
                compare_prfs(all_prfs,orb_prfs,inf_prfs,bins[1:],lin_rticks,debug_plt_path,sim + "_" + str(i)+"_dens",prf_func=None)
                
        curr_halos_r200m_list = []
        past_halos_r200m_list = []
        
        for i,sim in enumerate(use_sims):
            if i < len(use_sims) - 1:
                curr_idxs = all_idxs[sim_splits[i]:sim_splits[i+1]]
            else:
                curr_idxs = all_idxs[sim_splits[i]:]
            dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
            curr_z = dset_params["all_snap_info"]["prime_snap_info"]["red_shift"][()]
            p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"][()]
            snap_dir_format = dset_params["snap_dir_format"]
            snap_format = dset_params["snap_format"]
            
            sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
            curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"
                    
            # Load the halo's positions and radii
            param_paths = [["halos","R200m"]]
            sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name)
            p_halos_r200m = sparta_params[sparta_param_names[0]][:,p_sparta_snap]

            p_halos_r200m = p_halos_r200m[curr_idxs]

            # If we want the density profiles to only consist of halos of a specific peak height (nu) bin 
            if split_nu:
                nu_all_prf_lst = []
                nu_orb_prf_lst = []
                nu_inf_prf_lst = []
            
                cpy_plt_nu_splits = plt_nu_splits.copy()
                for i,nu_split in enumerate(cpy_plt_nu_splits):
                    # Take the second element of the where to filter by the halos (?)
                    fltr = np.where((calc_nus > nu_split[0]) & (calc_nus < nu_split[1]))[0]

                    if fltr.shape[0] > min_halo_nu_bin:
                        nu_all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
                        nu_orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
                        nu_inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
                    else:
                        plt_nu_splits.remove(nu_split)
                compare_split_prfs(plt_nu_splits,len(cpy_plt_nu_splits),nu_all_prf_lst,nu_orb_prf_lst,nu_inf_prf_lst,bins[1:],lin_rticks,plot_save_loc,title="nu_dens_med_",prf_func=np.nanmedian, prf_name_0="ML Model", prf_name_1="SPARTA")
            if split_macc and len(all_tdyn_steps) >= 1:
                if dset_params["t_dyn_steps"] :
                    if dset_params["t_dyn_steps"][0] == 1:
                        # we can just use the secondary snap here because if it was already calculated for 1 dynamical time forago
                        past_z = dset_params["all_snap_info"]["comp_"+str(all_tdyn_steps[0]) + "_tdstp_snap_info"]["red_shift"][()] 
                        c_sparta_snap = dset_params["all_snap_info"]["comp_"+str(all_tdyn_steps[0]) + "_tdstp_snap_info"]["sparta_snap"][()]
                    else:
                        # If the prior secondary snap is not 1 dynamical time ago get that information
                        
                        with h5py.File(curr_sparta_HDF5_path,"r") as f:
                            dic_sim = {}
                            grp_sim = f['simulation']
                            for f in grp_sim.attrs:
                                dic_sim[f] = grp_sim.attrs[f]
                            
                        all_sparta_z = dic_sim['snap_z']
                        
                        t_dyn = calc_t_dyn(p_halos_r200m[np.where(p_halos_r200m > 0)[0][0]], curr_z)
                        c_snap_dict = get_comp_snap_info(t_dyn=t_dyn, t_dyn_step=1, cosmol = cosmol, p_red_shift=curr_z, all_sparta_z=all_sparta_z,snap_dir_format=snap_dir_format,snap_format=snap_format,snap_path=snap_path)
                        c_sparta_snap = c_snap_dict["sparta_snap"]
                    c_halos_r200m = sparta_params[sparta_param_names[0]][:,c_sparta_snap]
                    c_halos_r200m = c_halos_r200m[curr_idxs]
                    
                    curr_halos_r200m_list.append(p_halos_r200m)
                    past_halos_r200m_list.append(c_halos_r200m)
                    
                curr_halos_r200m = np.concatenate(curr_halos_r200m_list)
                past_halos_r200m = np.concatenate(past_halos_r200m_list)
                macc_all_prf_lst = []
                macc_orb_prf_lst = []
                macc_inf_prf_lst = []
                
                calc_maccs = calc_mass_acc_rate(curr_halos_r200m,past_halos_r200m,curr_z,past_z)
                cpy_plt_macc_splits = plt_macc_splits.copy()
                for i,macc_split in enumerate(cpy_plt_macc_splits):
                    # Take the second element of the where to filter by the halos (?)
                    fltr = np.where((calc_maccs > macc_split[0]) & (calc_maccs < macc_split[1]))[0]
                    if fltr.shape[0] > 25:
                        macc_all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
                        macc_orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
                        macc_inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
                    else:
                        plt_macc_splits.remove(macc_split)

                
                compare_split_prfs(plt_macc_splits,len(cpy_plt_macc_splits),macc_all_prf_lst,macc_orb_prf_lst,macc_inf_prf_lst,bins[1:],lin_rticks,plot_save_loc,title= "macc_dens_", split_name="\Gamma", prf_name_0="ML Model", prf_name_1="SPARTA")
            if not split_nu and not split_macc:
                all_prf_lst = filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos)
                orb_prf_lst = filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos)
                inf_prf_lst = filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos)
                
                # Ignore warnigns about taking mean/median of empty slices and division by 0 that are expected with how the profiles are handled
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    compare_prfs(all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_save_loc,title="dens_med_",prf_func=np.nanmedian)
                    compare_prfs(all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_save_loc,title="dens_avg_",prf_func=np.nanmean)
    #TODO generalize this to any number of snapshots?
    # Both the missclassification and distribution and infalling orbiting ratio plots are 2D histograms and the model parameters and allow for linear-log split scaling
    if missclass or full_dist or io_frac:       
        p_corr_labels=y.compute().values.flatten()
        p_ml_labels=preds.values
        p_r=X["p_Scaled_radii"].values.compute()
        p_rv=X["p_Radial_vel"].values.compute()
        p_tv=X["p_Tangential_vel"].values.compute()
        c_r=X[str(all_tdyn_steps[0]) + "_Scaled_radii"].values.compute()
        c_rv=X[str(all_tdyn_steps[0]) +"_Radial_vel"].values.compute()
        
        split_scale_dict = {
            "linthrsh":linthrsh, 
            "lin_nbin":lin_nbin,
            "log_nbin":log_nbin,
            "lin_rvticks":lin_rvticks,
            "log_rvticks":log_rvticks,
            "lin_tvticks":lin_tvticks,
            "log_tvticks":log_tvticks,
            "lin_rticks":lin_rticks,
            "log_rticks":log_rticks,
        }
    
    # All parameter Distribution in 2D histograms
    if full_dist:
        plot_full_ptl_dist(p_corr_labels=p_corr_labels,p_r=p_r,p_rv=p_rv,p_tv=p_tv,c_r=c_r,c_rv=c_rv,split_scale_dict=split_scale_dict,num_bins=num_bins,save_loc=plot_save_loc)
    # All parameter Distribution of misclassifications of the model in 2D histograms
    if missclass:
        # Dataset name is used to save to model info the misclassification rates
        curr_sim_name = ""
        for sim in use_sims:
            curr_sim_name += sim
            curr_sim_name += "_"
        curr_sim_name += dst_type
        plot_miss_class_dist(p_corr_labels=p_corr_labels,p_ml_labels=p_ml_labels,p_r=p_r,p_rv=p_rv,p_tv=p_tv,c_r=c_r,c_rv=c_rv,split_scale_dict=split_scale_dict,num_bins=num_bins,save_loc=plot_save_loc,model_info=model_info,dataset_name=curr_sim_name)
    # All parameter Distribution of the ratio between the number of infalling and number of orbiting particlesin 2D histograms
    if io_frac:
        inf_orb_frac(p_corr_labels=p_corr_labels,p_r=p_r,p_rv=p_rv,p_tv=p_tv,c_r=c_r,c_rv=c_rv,split_scale_dict=split_scale_dict,num_bins=num_bins,save_loc=plot_save_loc)
       
# Loss function based off how close the reproduced density profile is to the actual profile
# Can use either only the orbiting profile, only the infalling one, or both
def dens_prf_loss(halo_ddf,use_sims,radii,labels,use_orb_prf,use_inf_prf):
    halo_first = halo_ddf["Halo_first"].values
    halo_n = halo_ddf["Halo_n"].values
    all_idxs = halo_ddf["Halo_indices"].values

    # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
    sim_splits = np.where(halo_first == 0)[0]
    # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
    # stacked simulations such that they correspond to the larger dataset and not one specific simulation
    if len(use_sims) != 1:
        for i in range(1,len(use_sims)):
            if i < len(use_sims) - 1:
                halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
            else:
                halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
               
    sparta_mass_prf_all,sparta_mass_prf_orb,all_masses,bins = load_sparta_mass_prf(sim_splits,all_idxs,use_sims)
    sparta_mass_prf_inf = sparta_mass_prf_all - sparta_mass_prf_orb
    sparta_mass_prf_orb = np.sum(sparta_mass_prf_orb,axis=0)
    sparta_mass_prf_inf = np.sum(sparta_mass_prf_inf,axis=0)
    #TODO make this robust for multiple sims
    calc_mass_prf_all,calc_mass_prf_orb,calc_mass_prf_inf = create_mass_prf(radii,labels,bins,all_masses[0]) 
    
    if use_orb_prf:
        use_orb = np.where(sparta_mass_prf_orb > 0)[0]
        orb_loss = np.sum(np.abs((sparta_mass_prf_orb[use_orb] - calc_mass_prf_orb[use_orb]) / sparta_mass_prf_orb[use_orb])) / bins.size
        if orb_loss == np.nan:
            orb_loss = 50
        elif orb_loss == np.inf:
            orb_loss == 50
    
    if use_inf_prf:
        use_inf = np.where(sparta_mass_prf_inf > 0)[0]
        inf_loss = np.sum(np.abs((sparta_mass_prf_inf[use_inf] - calc_mass_prf_inf[use_inf]) / sparta_mass_prf_inf[use_inf])) / bins.size
        if inf_loss == np.nan:
            inf_loss = 50
        elif inf_loss == np.inf:
            inf_loss == 50
    
    if use_orb_prf and use_inf_prf:
        print(orb_loss,inf_loss,orb_loss+inf_loss)
        return orb_loss+inf_loss
    elif use_orb_prf:
        print(orb_loss)
        return orb_loss
    elif use_inf_prf:
        print(inf_loss)
        return inf_loss

# Objective function for the optimization of a model that is adjusting the weighting of particles based on radius
def weight_objective(params,client,model_params,ptl_ddf,halo_ddf,use_sims,feat_cols,tar_col):
    train_dst,val_dst,train_halos,val_halos = split_data_by_halo(0.6,halo_ddf,ptl_ddf,return_halo=True)

    X_train = train_dst[feat_cols]
    y_train = train_dst[tar_col]
    
    X_val = val_dst[feat_cols]
    y_val = val_dst[tar_col]
    
    curr_weight_rad, curr_min_weight, curr_weight_exp = params

    train_radii = X_train["p_Scaled_radii"].values.compute()

    weights = weight_by_rad(train_radii,y_train.compute().values.flatten(), curr_weight_rad, curr_min_weight, curr_weight_exp)

    dask_weights = []
    scatter_weight = client.scatter(weights)
    dask_weight = dd.from_delayed(scatter_weight) 
    dask_weights.append(dask_weight)
        
    train_weights = dd.concat(dask_weights)
    
    train_weights = train_weights.repartition(npartitions=X_train.npartitions)

        
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train, weight=train_weights)
    
    output = dxgb.train(
                client,
                model_params,
                dtrain,
                num_boost_round=50,
                # evals=[(dtrain, "train"),(dtest, "test")],
                evals=[(dtrain, "train")],
                early_stopping_rounds=5,      
                )
    bst = output["booster"]

    y_pred = make_preds(client, bst, X_val, report_name="Report", print_report=False)
    y_val = y_val.compute().values.flatten()
    
    # multiply the accuracies by -1 because we want to maximize them but we are using minimization
    if hpo_loss == "all":
        accuracy = -1 * accuracy_score(y_val, y_pred)
    elif hpo_loss == "orb":
        only_orb = np.where(y_val == 1)[0]
        accuracy = -1 * accuracy_score(y_val[only_orb], y_pred.iloc[only_orb].values)
    elif hpo_loss == "inf":
        only_inf = np.where(y_val == 0)[0]
        accuracy = -1 * accuracy_score(y_val[only_inf], y_pred.iloc[only_inf].values)
    elif hpo_loss == "mprf_all":
        val_radii = X_val["p_Scaled_radii"].values.compute()
        accuracy = dens_prf_loss(val_halos,use_sims,val_radii,y_pred,use_orb_prf=True,use_inf_prf=True)
    elif hpo_loss == "mprf_orb":
        val_radii = X_val["p_Scaled_radii"].values.compute()
        accuracy = dens_prf_loss(val_halos,use_sims,val_radii,y_pred,use_orb_prf=True,use_inf_prf=False)
    elif hpo_loss == "mprf_inf":
        val_radii = X_val["p_Scaled_radii"].values.compute()
        accuracy = dens_prf_loss(val_halos,use_sims,val_radii,y_pred,use_orb_prf=False,use_inf_prf=True)

    return accuracy

# Objective function for the optimization of a model that is adjusting the number of particles beyond a certain radius based on a percetnage of particles within that radius
def scal_rad_objective(params,client,model_params,ptl_ddf,halo_ddf,use_sims,feat_cols,tar_col):
    train_dst,val_dst,train_halos,val_halos = split_data_by_halo(client,0.5,halo_ddf,ptl_ddf,return_halo=True)
    
    X_val = val_dst[feat_cols]
    y_val = val_dst[tar_col]
    
    data_pd = train_dst.compute()
    num_bins=100
    bin_edges = np.logspace(np.log10(0.001),np.log10(10),num_bins)
    scld_data = scale_by_rad(data_pd,bin_edges,params[0],params[1])
    
    scatter_scld = client.scatter(scld_data)
    scld_data_ddf = dd.from_delayed(scatter_scld)

    X_train = scld_data_ddf[feat_cols]
    y_train = scld_data_ddf[tar_col]
    
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    
    output = dxgb.train(
                client,
                model_params,
                dtrain,
                num_boost_round=50,
                # evals=[(dtrain, "train"),(dtest, "test")],
                evals=[(dtrain, "train")],
                early_stopping_rounds=5,      
                )
    bst = output["booster"]

    y_pred = make_preds(client, bst, X_val, report_name="Report", print_report=False)
    y_val = y_val.compute().values.flatten()
    only_orb = np.where(y_val == 1)[0]
    only_inf = np.where(y_val == 0)[0]

    accuracy = accuracy_score(y_val, y_pred)
    # accuracy = accuracy_score(y[only_orb], y_pred.iloc[only_orb].values)
    # accuracy = accuracy_score(y[only_inf], y_pred.iloc[only_inf].values)
    
    # val_radii = X_val["p_Scaled_radii"].values.compute()
    # accuracy = -1 * dens_prf_loss(val_halos,use_sims,val_radii,y_pred)
    
    return -accuracy

# Prints which iteration of optimization and the current parameters
def print_iteration(res):
    iteration = len(res.x_iters)
    print(f"Iteration {iteration}: Current params: {res.x_iters[-1]}, Current score: {res.func_vals[-1]}")

def optimize_weights(client,model_params,ptl_ddf,halo_ddf,use_sims,feat_cols,tar_col):
    print("Start Optimization of Weights")
    space  = [Real(0.1, 5.0, name='weight_rad'),
            Real(0.001, 0.2, name='min_weight'),
            Real(0.1,10,name='weight_exp')]

    objective_with_params = partial(weight_objective,client=client,model_params=model_params,ptl_ddf=ptl_ddf,halo_ddf=halo_ddf,use_sims=use_sims,feat_cols=feat_cols,tar_col=tar_col)
    res = gp_minimize(objective_with_params, space, n_calls=50, random_state=0, callback=[print_iteration])

    print("Best parameters: ", res.x)
    print("Best accuracy: ", -res.fun)
    
    return res.x[0],res.x[1],res.x[2]
    
def optimize_scale_rad(client,model_params,ptl_ddf,halo_ddf,use_sims,feat_cols,tar_col):    
    print("Start Optimization of Scaling Radii")
    space  = [Real(0.1, 5.0, name='reduce_rad'),
            Real(0.0001, 0.25, name='reduce_perc')]

    objective_with_params = partial(scal_rad_objective,client=client,model_params=model_params,ptl_ddf=ptl_ddf,halo_ddf=halo_ddf,use_sims=use_sims,feat_cols=feat_cols,tar_col=tar_col)
    res = gp_minimize(objective_with_params, space, n_calls=50, random_state=0, callback=[print_iteration])

    print("Best parameters: ", res.x)
    print("Best accuracy: ", -res.fun)
    
    return res.x[0],res.x[1]
    
def get_combined_name(model_sims):
    combined_name = ""
    for i,sim in enumerate(model_sims):
        split_string = sim.split('_')
        snap_list = extract_snaps(sim)
        
        r_patt = r'(\d+-\d+|\d+)r'
        r_match = re.search(r_patt,split_string[3])

        
        v_patt = r'(\d+-\d+|\d+)v'
        v_match = re.search(v_patt, split_string[4])


        cond_string = split_string[0] + split_string[1] + split_string[2] + "s" 
        cond_string = cond_string + "_".join(str(x) for x in snap_list)
        # can add these for more information per name
        #+ "r" + r_match.group(1) + "v" + v_match.group(1) + "s" + split_string[5]
        
        combined_name += cond_string
    
    return combined_name
    
def filter_ddf(X, y = None, preds = None, fltr_dic = None, col_names = None, max_size=500):
    with timed("Filter DF"):
        full_filter = None
        if fltr_dic is not None:
            if "X_filter" in fltr_dic:
                for feature, (operator, value) in fltr_dic["X_filter"].items():
                    if operator == '>':
                        condition = X[feature] > value
                    elif operator == '<':
                        condition = X[feature] < value
                    elif operator == '>=':
                        condition = X[feature] >= value
                    elif operator == '<=':
                        condition = X[feature] <= value
                    elif operator == '==':
                        if value == "nan":
                            condition = X[feature].isna()
                        else:
                            condition = X[feature] == value
                    elif operator == '!=':
                        condition = X[feature] != value
                        
                    if feature == next(iter(fltr_dic[next(iter(fltr_dic))])):
                        full_filter = condition
                    else:
                        full_filter &= condition
                
            if "label_filter" in fltr_dic:
                for feature, value in fltr_dic["label_filter"].items():
                    if feature == "act":
                        condition = y["Orbit_infall"] == value
                    elif feature == "pred":
                        condition = preds == value
                    if feature == next(iter(fltr_dic[next(iter(fltr_dic))])):
                        full_filter = condition
                    else:
                        full_filter &= condition

            
            X = X[full_filter]
        nrows = X.shape[0].compute()
            
        if nrows > max_size and max_size > 0:
            sample = max_size / nrows
        else:
            sample = 1.0
            
        if sample > 0 and sample < 1:
            X = X.sample(frac=sample,random_state=rand_seed)
        
        if col_names != None:
            X.columns = col_names
            
        # Return the filtered array and the indices of the original array that remain
        return X.compute(), X.index.values.compute()
    
# Can set max_size to 0 to include all the particles
def shap_with_filter(explainer, X, y, preds, fltr_dic = None, col_names = None, max_size=500):
    X_fltr,fltr = filter_ddf(X, y, preds, fltr_dic = fltr_dic, col_names = col_names, max_size=max_size)
    return explainer(X_fltr), explainer.shap_values(X_fltr), X_fltr
    
    