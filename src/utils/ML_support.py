import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client
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
    snap_list = [int(num) for num in number_strs]
    snap_list.sort()
    snap_list.reverse()
    return snap_list

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

def get_model_name(model_type, trained_on_sims):
    model_name = f"{model_type}_{get_combined_name(trained_on_sims)}"        
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
            
    sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, pickle_data=pickle_data)
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
def split_dataframe(df, max_size):
    total_size = df.memory_usage(index=True).sum()
    num_splits = int(np.ceil(total_size / max_size))
    chunk_size = int(np.ceil(len(df) / num_splits))
    print("splitting Dataframe into:",num_splits,"dataframes")
    
    split_dfs = []
    split_weights = []
    for i in range(0, len(df), chunk_size):
        split_dfs.append(df.iloc[i:i + chunk_size])

    return split_dfs

# Function to process a file in a dataset's folder: combines them all, performs any desired filtering, calculates weights if desired, and calculates scaled position weight
# Also splits the dataframe into smaller dataframes based of inputted maximum memory size
def process_file(folder_path, file_index, ptl_mass, use_z, max_mem, sim_cosmol, filter_nu, nu_splits):
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

        # Calculate scale position weight
        scal_pos_weight = calc_scal_pos_weight(ptl_df)

        # If the dataframe is too large split it up
        if ptl_df.memory_usage(index=True).sum() > max_mem:
            ptl_dfs = split_dataframe(ptl_df, max_mem)
        else:
            ptl_dfs = [ptl_df]
        
        return ptl_dfs,scal_pos_weight
    return delayed_task()

# Combines the results of the processing of each file in the folder into one dataframe for the data and list for the scale position weights and an array of weights if desired
def combine_results(results, client):
    # Unpack the results
    ddfs,scal_pos_weights = [], []
    
    for res in results:
        ddfs.extend(res[0])
        scal_pos_weights.append(res[1]) # We append since scale position weight is just a number
            
    all_ddfs = dd.concat([dd.from_delayed(client.scatter(df)) for df in ddfs])

    return all_ddfs, scal_pos_weights

# Combines all the files in a dataset's folder into one dask dataframe and a list for the scale position weights and an array of weights if desired 
def reform_datasets_nested(client, ptl_mass, use_z, max_mem, sim_cosmol, folder_path, prime_snap, filter_nu=None, limit_files=False, nu_splits=None):
    snap_n_files = len(os.listdir(folder_path + "/ptl_info/" + str(prime_snap)+"/"))
    n_files = snap_n_files
    if limit_files:
        n_files = np.min([snap_n_files,file_lim]) 
    
    delayed_results = []
    for file_index in range(n_files):
        # Create delayed tasks for each file
        delayed_results.append(process_file(
                folder_path, file_index, ptl_mass, use_z,
                max_mem, sim_cosmol, filter_nu, nu_splits))
    
    # Compute the results in parallel
    results = client.compute(delayed_results, sync=True)

    return combine_results(results, client)
    
# Calculates the scaled position weight for a dataset. Which is used to weight the model towards the population with less particles (should be the orbiting population)
def calc_scal_pos_weight(df):
    count_negatives = (df['Orbit_infall'] == 0).sum()
    count_positives = (df['Orbit_infall'] == 1).sum()

    scale_pos_weight = count_negatives / count_positives
    return scale_pos_weight

# Loads all the data for the inputted list of simulations into one dataframe. Finds the scale position weight for the dataset and any adjusted weighting for it if desired
def load_data(client, sims, dset_name, sim_cosmol, prime_snap, limit_files = False, filter_nu=False, nu_splits=None):
    dask_dfs = []
    all_scal_pos_weight = []
        
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
                ptl_ddf,sim_scal_pos_weight = reform_datasets_nested(client,ptl_mass,use_z,max_mem,sim_cosmol,dataset_path,prime_snap,filter_nu=filter_nu,limit_files=limit_files,nu_splits=nu_splits)  
                all_scal_pos_weight.append(sim_scal_pos_weight)
                dask_dfs.append(ptl_ddf)
                    
    all_scal_pos_weight = np.average(np.concatenate([np.array(sublist).flatten() for sublist in all_scal_pos_weight]))
    act_scale_pos_weight = np.average(all_scal_pos_weight)

    all_dask_dfs = dd.concat(dask_dfs)
    
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
        
        param_paths = [["halos","id"],["simulation","particle_mass"],["anl_prf","M_all"],["anl_prf","M_1halo"],["halos","R200m"],["config","anl_prf","r_bins_lin"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, pickle_data=pickle_data)
        halos_ids = sparta_params[sparta_param_names[0]][:,p_sparta_snap]
        ptl_mass = sparta_params[sparta_param_names[1]]
 
        use_halo_ids = halos_ids[use_idxs]

        mass_prf_all_list.append(sparta_params[sparta_param_names[2]][:,p_sparta_snap,:])
        mass_prf_1halo_list.append(sparta_params[sparta_param_names[3]][:,p_sparta_snap,:])

        all_r200m_list.append(sparta_params[sparta_param_names[4]][:,p_sparta_snap])

        all_masses.append(ptl_mass)

    mass_prf_all = np.vstack(mass_prf_all_list)
    mass_prf_1halo = np.vstack(mass_prf_1halo_list)
    all_r200m = np.concatenate(all_r200m_list)
    
    bins = sparta_params[sparta_param_names[5]]
    bins = np.insert(bins, 0, 0)

    if ret_r200m:
        return mass_prf_all,mass_prf_1halo,all_masses,bins,all_r200m
    else:
        return mass_prf_all,mass_prf_1halo,all_masses,bins


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
                for feature, conditions in fltr_dic["X_filter"].items():
                    if not isinstance(conditions, list):
                        conditions = [conditions]
                    for operator, value in conditions:
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
                            
                        full_filter = condition if full_filter is None else full_filter & condition
                
            if "label_filter" in fltr_dic:
                for feature, value in fltr_dic["label_filter"].items():
                    if feature == "sparta":
                        print(y)
                        if isinstance(y, (dd.DataFrame, pd.DataFrame)):
                            y = y["Orbit_infall"]      
                        condition = y == value
                    elif feature == "pred":
                        if isinstance(preds, (dd.DataFrame, pd.DataFrame)):
                            preds = preds["preds"]
                        condition = preds == value
                        if isinstance(condition, dd.DataFrame):
                            condition = condition["preds"]
                            condition = condition.reset_index(drop=True)
                            
                    full_filter = condition if full_filter is None else full_filter & condition

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
        return X.compute(),full_filter.compute()
  
def filter_df(X, y=None, preds=None, fltr_dic=None, col_names=None, max_size=500, rand_seed=42):
    full_filter = pd.Series(True, index=X.index)

    if fltr_dic is not None:
        if "X_filter" in fltr_dic:
            for feature, conditions in fltr_dic["X_filter"].items():
                if not isinstance(conditions, list):
                    conditions = [conditions]
                for operator, value in conditions:
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
                    else:
                        raise ValueError(f"Unknown operator: {operator}")
                    full_filter &= condition

        if "label_filter" in fltr_dic:
            for feature, value in fltr_dic["label_filter"].items():
                if feature == "sparta":
                    if isinstance(y, pd.DataFrame):
                        y = y["Orbit_infall"]
                    condition = y == value
                elif feature == "pred":
                    if isinstance(preds, pd.DataFrame):
                        preds = preds["preds"]
                    condition = preds == value
                else:
                    raise ValueError(f"Unknown label_filter feature: {feature}")
                full_filter &= condition

        X = X[full_filter]

    # Sample if needed
    nrows = len(X)
    if nrows > max_size and max_size > 0:
        sample_frac = max_size / nrows
        X = X.sample(frac=sample_frac, random_state=rand_seed)

    if col_names is not None:
        X.columns = col_names

    return X, full_filter  
    
# Can set max_size to 0 to include all the particles
def shap_with_filter(explainer, X, y, preds, fltr_dic = None, col_names = None, max_size=500):
    X_fltr,fltr = filter_ddf(X, y, preds, fltr_dic = fltr_dic, col_names = col_names, max_size=max_size)
    return explainer(X_fltr), explainer.shap_values(X_fltr), X_fltr
    
    