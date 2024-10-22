import numpy as np
import os
import sys
import pickle
from dask import array as da
import dask.dataframe as dd
from dask import delayed
import xgboost as xgb
from xgboost import dask as dxgb
import json
import h5py
import re
import matplotlib.pyplot as plt
import pandas as pd
from colossus.lss import peaks
from dask.distributed import Client
import time
import multiprocessing as mp

from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import accuracy_score
from functools import partial

from utils.data_and_loading_functions import load_or_pickle_SPARTA_data, find_closest_z, conv_halo_id_spid, timed, split_data_by_halo
from utils.visualization_functions import compare_density_prf, plot_per_err
from utils.update_vis_fxns import plot_full_ptl_dist, plot_miss_class_dist
from utils.calculation_functions import create_mass_prf
from sparta_tools import sparta # type: ignore
from colossus.cosmology import cosmology

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
model_sims = json.loads(config.get("XGBOOST","model_sims"))

global p_red_shift
p_red_shift = config.getfloat("SEARCH","p_red_shift")
search_rad = config.getfloat("SEARCH","search_rad")

file_lim = config.getint("XGBOOST","file_lim")
model_type = config["XGBOOST"]["model_type"]
train_rad = config.getint("XGBOOST","training_rad")
nu_splits = config["XGBOOST"]["nu_splits"]

reduce_rad = config.getfloat("XGBOOST","reduce_rad")
reduce_perc = config.getfloat("XGBOOST", "reduce_perc")

weight_rad = config.getfloat("XGBOOST","weight_rad")
min_weight = config.getfloat("XGBOOST","min_weight")
weight_exp = config.getfloat("XGBOOST","weight_exp")

hpo_loss = config.get("XGBOOST","hpo_loss")

nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

linthrsh = config.getfloat("XGBOOST","linthrsh")
lin_nbin = config.getint("XGBOOST","lin_nbin")
log_nbin = config.getint("XGBOOST","log_nbin")
lin_rvticks = json.loads(config.get("XGBOOST","lin_rvticks"))
log_rvticks = json.loads(config.get("XGBOOST","log_rvticks"))
lin_tvticks = json.loads(config.get("XGBOOST","lin_tvticks"))
log_tvticks = json.loads(config.get("XGBOOST","log_tvticks"))
lin_rticks = json.loads(config.get("XGBOOST","lin_rticks"))
log_rticks = json.loads(config.get("XGBOOST","log_rticks"))

if sim_cosmol == "planck13-nbody":
    cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
else:
    cosmol = cosmology.setCosmology(sim_cosmol) 

if use_gpu:
    from dask_cuda import LocalCUDACluster
    from cuml.metrics.accuracy import accuracy_score #TODO fix cupy installation??
    from sklearn.metrics import make_scorer
    import dask_ml.model_selection as dcv
    import cudf
    import dask_cudf as dc
elif not use_gpu and on_zaratan:
    from dask_mpi import initialize
    from mpi4py import MPI
    from distributed.scheduler import logger
    import socket
    #from dask_jobqueue import SLURMCluster
elif not on_zaratan:
    from dask_cuda import LocalCUDACluster
    
    
def get_CUDA_cluster():
    cluster = LocalCUDACluster(
                               device_memory_limit='10GB',
                               jit_unspill=True)
    client = Client(cluster)
    return client

def make_preds(client, bst, X, dask = False, threshold = 0.5, report_name="Classification Report", print_report=False):
    if dask:
        preds = dxgb.predict(client,bst,X)
        preds = preds.map_partitions(lambda df: (df >= threshold).astype(int))
        return preds
    else:
        preds = dxgb.inplace_predict(client, bst, X).compute()
        preds = (preds >= threshold).astype(np.int8)
    
    return preds

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
            
def create_dmatrix(client, X_cpu, y_cpu, features, chunk_size, frac_use_data = 1, calc_scale_pos_weight = False):    
    scale_pos_weight = np.where(y_cpu == 0)[0].size / np.where(y_cpu == 1)[0].size
    
    num_features = X_cpu.shape[1]
    
    num_use_data = int(np.floor(X_cpu.shape[0] * frac_use_data))
    print("Tot num of particles:", X_cpu.shape[0], "Num use particles:", num_use_data)
    X = da.from_array(X_cpu,chunks=(chunk_size,num_features))
    y = da.from_array(y_cpu,chunks=(chunk_size))
        
    print("X Number of total bytes:", X.nbytes, "X Number of Gigabytes:", (X.nbytes)/(10**9))
    print("y Number of total bytes:", y.nbytes, "y Number of Gigabytes:", (y.nbytes)/(10**9))
    
    dqmatrix = xgb.dask.DaskDMatrix(client, X, y, feature_names=features)
    
    if calc_scale_pos_weight:
        return dqmatrix, X, y_cpu, scale_pos_weight 
    return dqmatrix, X, y_cpu

def split_calc_name(sim):
    sim_pat = r"cbol_l(\d+)_n(\d+)"
    match = re.search(sim_pat, sim)
    if not match:
        sim_pat = r"cpla_l(\d+)_n(\d+)"
        match = re.search(sim_pat,sim)
        
    if match:
        sparta_name = match.group(0)
           
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

def weight_by_rad(radii,orb_inf,use_weight_rad=weight_rad,use_min_weight=min_weight,use_weight_exp=weight_exp,weight_inf=False,weight_orb=True):
    weights = np.ones(radii.shape[0])

    if weight_inf and weight_orb:
        mask = (radii > use_weight_rad)
    elif weight_orb:
        mask = (radii > use_weight_rad) & (orb_inf == 1)
    elif weight_inf:
        mask = (radii > use_weight_rad) & (orb_inf == 0)
    else:
        print("No weights calculated set weight_inf and/or weight_orb = True")
        return pd.DataFrame(weights)
    weights[mask] = (np.exp((np.log(use_min_weight)/(np.max(radii)-use_weight_rad)) * (radii[mask]-use_weight_rad)))**use_weight_exp

    return pd.DataFrame(weights)

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

def sort_files(folder_path,limit_files=False):
    hdf5_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.h5'):
            hdf5_files.append(f)
    hdf5_files.sort()
    if file_lim > 0 and file_lim < len(hdf5_files) and limit_files:
        hdf5_files = hdf5_files[:file_lim]
    return hdf5_files
    
def sim_info_for_nus(sim,config_params):
    sparta_name, sparta_search_name = split_calc_name(sim)
            
    with h5py.File(path_to_SPARTA_data + sparta_name + "/" +  sparta_search_name + ".hdf5","r") as f:
        dic_sim = {}
        grp_sim = f['simulation']
        for f in grp_sim.attrs:
            dic_sim[f] = grp_sim.attrs[f]
    
    p_red_shift = config_params["p_red_shift"]
    
    all_red_shifts = dic_sim['snap_z']
    p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
    use_z = all_red_shifts[p_sparta_snap]
    p_snap_loc = transform_string(sim)
    with open(path_to_pickle + p_snap_loc + "/ptl_mass.pickle", "rb") as pickle_file:
        ptl_mass = pickle.load(pickle_file)
    
    return ptl_mass, use_z

def split_dataframe(df, max_size, weights=None, use_weights = False):
    # Split a dataframe so that each one is below a maximum memory size
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

def reform_datasets(client,config_params,sim,folder_path,scale_rad=False,use_weights=False,filter_nu=None,limit_files=False):
    ptl_files = sort_files(folder_path + "/ptl_info/",limit_files=limit_files)
    
    # Function to process each file
    @delayed
    def process_file(ptl_index):
        ptl_path = folder_path + f"/ptl_info/ptl_{ptl_index}.h5"
        halo_path = folder_path + f"/halo_info/halo_{ptl_index}.h5"

        ptl_df = pd.read_hdf(ptl_path)
        halo_df = pd.read_hdf(halo_path)

        # reset indices for halo_first halo_n indexing
        halo_df["Halo_first"] = halo_df["Halo_first"] - halo_df["Halo_first"][0]
        
        # find nu values for halos in this chunk
        ptl_mass, use_z = sim_info_for_nus(sim,config_params)
        nus = np.array(peaks.peakHeight((halo_df["Halo_n"][:] * ptl_mass), use_z))
        
        # Filter by nu and by radius
        if filter_nu:
            ptl_df = split_by_nu(ptl_df, nus, halo_df["Halo_first"], halo_df["Halo_n"])
        if scale_rad:
            num_bins=100
            bin_edges = np.logspace(np.log10(0.001),np.log10(10),num_bins)
            ptl_df = scale_by_rad(ptl_df,bin_edges)
        if use_weights:
            weights = weight_by_rad(ptl_df["p_Scaled_radii"].values,ptl_df["Orbit_infall"].values,weight_inf=False,weight_orb=True)
        
        # Calculate scale position weight
        scal_pos_weight = calc_scal_pos_weight(ptl_df)

        # If the dataframe is too large split it up
        max_mem = int(np.floor(config_params["HDF5 Mem Size"] / 2))
        if ptl_df.memory_usage(index=True).sum() > max_mem:
            ptl_dfs = split_dataframe(ptl_df, max_mem)
            if use_weights:
                ptl_dfs, weights = split_dataframe(ptl_df,max_mem,weights,use_weights=True)
        else:
            ptl_dfs = [ptl_df]
            if use_weights:
                weights = [weights]
        
        if scale_rad and use_weights:
            return ptl_dfs,scal_pos_weight,weights,bin_edges
        elif scale_rad and not use_weights:
            return ptl_dfs,scal_pos_weight,bin_edges
        elif not scale_rad and use_weights:
            return ptl_dfs,scal_pos_weight,weights
        else:
            return ptl_dfs,scal_pos_weight

    # Create delayed tasks for each file
    delayed_results = [process_file(i) for i in range(len(ptl_files))]

    # Compute the results in parallel
    results = client.compute(delayed_results, sync=True)

    # Unpack the results
    ddfs,sim_scal_pos_weight,dask_weights = [], [], []
    if scale_rad and use_weights:    
        for ptl_df, scal_pos_weight, weight, bin_edge in results:
            scatter_df = client.scatter(ptl_df)
            dask_df = dd.from_delayed(scatter_df)
            if use_gpu:
                dask_df = dask_df.map_partitions(cudf.from_pandas)
            ddfs.append(dask_df)
            sim_scal_pos_weight.append(scal_pos_weight)
            scatter_weight = client.scatter(weight)
            dask_weight = dd.from_delayed(scatter_weight) 
            dask_weights.append(dask_weight)
            bin_edges = bin_edge
        if use_gpu:
            all_ddfs = dc.concat(ddfs,axis=0)
        else:
            all_ddfs = dd.concat(ddfs)
        return all_ddfs,sim_scal_pos_weight,dd.concat(dask_weights),bin_edges
    elif scale_rad and not use_weights:
        for ptl_df, scal_pos_weight, bin_edge in results:
            scatter_df = client.scatter(ptl_df)
            dask_df = dd.from_delayed(scatter_df)
            if use_gpu:
                dask_df = dask_df.map_partitions(cudf.from_pandas)
            ddfs.append(dask_df)
            sim_scal_pos_weight.append(scal_pos_weight)
            bin_edges = bin_edge
        if use_gpu:
            all_ddfs = dc.concat(ddfs,axis=0)
        else:
            all_ddfs = dd.concat(ddfs)
        return all_ddfs,sim_scal_pos_weight,bin_edges
    elif not scale_rad and use_weights:
        for ptl_df, scal_pos_weight, weight in results:
            scatter_df = client.scatter(ptl_df)
            dask_df = dd.from_delayed(scatter_df)
            if use_gpu:
                dask_df = dask_df.map_partitions(cudf.from_pandas)
            ddfs.append(dask_df)
            sim_scal_pos_weight.append(scal_pos_weight)
            scatter_weight = client.scatter(weight)
            dask_weight = dd.from_delayed(scatter_weight) 
            dask_weights.append(dask_weight)
        if use_gpu:
            all_ddfs = dc.concat(ddfs,axis=0)
        else:
            all_ddfs = dd.concat(ddfs)
        return all_ddfs,sim_scal_pos_weight,dd.concat(dask_weights)
    else:
        for ptl_df, scal_pos_weight in results:
            scatter_df = client.scatter(ptl_df)
            dask_df = dd.from_delayed(scatter_df)
            if use_gpu:
                dask_df = dask_df.map_partitions(cudf.from_pandas)
            ddfs.append(dask_df)    
            sim_scal_pos_weight.append(scal_pos_weight)
        if use_gpu:
            all_ddfs = dc.concat(ddfs,axis=0)
        else:
            all_ddfs = dd.concat(ddfs)
        return all_ddfs,sim_scal_pos_weight
        
def calc_scal_pos_weight(df):
    count_negatives = (df['Orbit_infall'] == 0).sum()
    count_positives = (df['Orbit_infall'] == 1).sum()

    scale_pos_weight = count_negatives / count_positives
    return scale_pos_weight

def load_data(client, sims, dset_name, limit_files = False, scale_rad=False, use_weights=False, filter_nu=False):
    dask_dfs = []
    all_scal_pos_weight = []
    all_weights = []
    
    for sim in sims:
        files_loc = path_to_calc_info + sim + "/" + dset_name
        with open(path_to_calc_info + sim + "/config.pickle","rb") as f:
            config_params = pickle.load(f)
        
        if dset_name == "Full":
            with timed("Reformed " + "Train" + " Dataset: " + sim):    
                files_loc = path_to_calc_info + sim + "/" + "Train"
                if scale_rad and use_weights:
                    ptl_ddf,sim_scal_pos_weight, weights, bin_edges = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files)  
                elif scale_rad and not use_weights:
                    ptl_ddf,sim_scal_pos_weight, bin_edges = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files)
                elif not scale_rad and use_weights:
                    ptl_ddf,sim_scal_pos_weight, weights = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files) 
                else:
                    ptl_ddf,sim_scal_pos_weight = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files)  
            print("num of partitions:",ptl_ddf.npartitions)
            all_scal_pos_weight.append(sim_scal_pos_weight)
            dask_dfs.append(ptl_ddf)
            if use_weights:
                all_weights.append(weights)
            with timed("Reformed " + "Test" + " Dataset: " + sim):
                files_loc = path_to_calc_info + sim + "/" + "Test"
                if scale_rad and use_weights:
                    ptl_ddf,sim_scal_pos_weight, weights, bin_edges = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files)  
                elif scale_rad and not use_weights:
                    ptl_ddf,sim_scal_pos_weight, bin_edges = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files)
                elif not scale_rad and use_weights:
                    ptl_ddf,sim_scal_pos_weight, weights = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files) 
                else:
                    ptl_ddf,sim_scal_pos_weight = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files) 
            print("num of partitions:",ptl_ddf.npartitions)
            all_scal_pos_weight.append(sim_scal_pos_weight)
            dask_dfs.append(ptl_ddf)
            if use_weights:
                all_weights.append(weights)
        else:
            with timed("Reformed " + dset_name + " Dataset: " + sim):  
                if scale_rad and use_weights:
                    ptl_ddf,sim_scal_pos_weight, weights, bin_edges = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files)  
                elif scale_rad and not use_weights:
                    ptl_ddf,sim_scal_pos_weight, bin_edges = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files)
                elif not scale_rad and use_weights:
                    ptl_ddf,sim_scal_pos_weight, weights = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files) 
                else:
                    ptl_ddf,sim_scal_pos_weight = reform_datasets(client,config_params,sim,files_loc,scale_rad=scale_rad,use_weights=use_weights,filter_nu=filter_nu,limit_files=limit_files)   
            print("num of partitions:",ptl_ddf.npartitions)
            all_scal_pos_weight.append(sim_scal_pos_weight)
            dask_dfs.append(ptl_ddf)
            if use_weights:
                all_weights.append(weights)

    all_scal_pos_weight = np.average(np.concatenate([np.array(sublist).flatten() for sublist in all_scal_pos_weight]))
    act_scale_pos_weight = np.average(all_scal_pos_weight)
    
    if use_gpu:
        all_dask_dfs = dc.concat(dask_dfs,axis=0)
    else:
        all_dask_dfs = dd.concat(dask_dfs)
    
    if scale_rad and use_weights:
        return all_dask_dfs,act_scale_pos_weight, dd.concat(all_weights), bin_edges
    elif scale_rad and not use_weights:
        return all_dask_dfs,act_scale_pos_weight, bin_edges
    elif not scale_rad and use_weights:
        return all_dask_dfs,act_scale_pos_weight, dd.concat(all_weights)
    else:
        return all_dask_dfs,act_scale_pos_weight

def load_sprta_mass_prf(sim_splits,all_idxs,use_sims):                
    mass_prf_all_list = []
    mass_prf_1halo_list = []
    all_masses = []
    
    for i,sim in enumerate(use_sims):
        # Get the halo indices corresponding to this simulation
        if i < len(use_sims) - 1:
            use_idxs = all_idxs[sim_splits[i]:sim_splits[i+1]]
        else:
            use_idxs = all_idxs[sim_splits[i]:]
        
        
        sparta_name, sparta_search_name = split_calc_name(sim)
        # find the snapshots for this simulation
        snap_pat = r"(\d+)to(\d+)"
        match = re.search(snap_pat, sim)
        if match:
            curr_snap_list = [match.group(1), match.group(2)] 
        
        with open(path_to_calc_info + sim + "/config.pickle", "rb") as file:
            config_dict = pickle.load(file)
            
            curr_z = config_dict["p_snap_info"]["red_shift"][()]
            curr_snap_dir_format = config_dict["snap_dir_format"]
            curr_snap_format = config_dict["snap_format"]
            new_p_snap, curr_z = find_closest_z(curr_z,path_to_snaps + sparta_name + "/",curr_snap_dir_format,curr_snap_format)
            p_scale_factor = 1/(1+curr_z)
            
        with h5py.File(path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5","r") as f:
            dic_sim = {}
            grp_sim = f['simulation']

            for attr in grp_sim.attrs:
                dic_sim[attr] = grp_sim.attrs[attr]
        
        all_red_shifts = dic_sim['snap_z']
        p_sparta_snap = np.abs(all_red_shifts - curr_z).argmin()
        
        halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, parent_id, ptl_mass = load_or_pickle_SPARTA_data(sparta_search_name, p_scale_factor, curr_snap_list[0], p_sparta_snap)

        use_halo_ids = halos_id[use_idxs]
        
        sparta_output = sparta.load(filename=path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5", halo_ids=use_halo_ids, log_level=0)
        new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta resort the indices
        
        mass_prf_all_list.append(sparta_output['anl_prf']['M_all'][new_idxs,p_sparta_snap,:])
        mass_prf_1halo_list.append(sparta_output['anl_prf']['M_1halo'][new_idxs,p_sparta_snap,:])
        all_masses.append(ptl_mass)

    mass_prf_all = np.vstack(mass_prf_all_list)
    mass_prf_1halo = np.vstack(mass_prf_1halo_list)
    
    bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
    bins = np.insert(bins, 0, 0)

    return mass_prf_all,mass_prf_1halo,all_masses,bins

def eval_model(model_info, client, model, use_sims, dst_type, X, y, halo_ddf, combined_name, plot_save_loc, dens_prf = False,missclass=False,full_dist=False,per_err=False): 
    with timed("Predictions"):
        print(f"Starting predictions for {y.size.compute():.3e} particles")
        preds = make_preds(client, model, X, report_name="Report", print_report=False)

    X = X.compute()
    y = y.compute()
    
    X_scatter = client.scatter(X)
    X = dd.from_delayed(X_scatter)
    y_scatter = client.scatter(y)
    y = dd.from_delayed(y_scatter)

    
    num_bins = 30

    if dens_prf:
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

        sparta_mass_prf_all, sparta_mass_prf_1halo,all_masses,bins = load_sprta_mass_prf(sim_splits,all_idxs,use_sims)
        compare_density_prf(sim_splits,radii=X["p_Scaled_radii"].values.compute(), halo_first=halo_first, halo_n=halo_n, act_mass_prf_all=sparta_mass_prf_all, act_mass_prf_orb=sparta_mass_prf_1halo, mass=all_masses, orbit_assn=preds.values, prf_bins=bins, title="", save_location=plot_save_loc, use_mp=True)
    
    if missclass or full_dist:       
        p_corr_labels=y.compute().values.flatten()
        p_ml_labels=preds.values
        p_r=X["p_Scaled_radii"].values.compute()
        p_rv=X["p_Radial_vel"].values.compute()
        p_tv=X["p_Tangential_vel"].values.compute()
        c_r=X["c_Scaled_radii"].values.compute()
        c_rv=X["c_Radial_vel"].values.compute()
        
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
    
    if full_dist:
        plot_full_ptl_dist(p_corr_labels=p_corr_labels,p_r=p_r,p_rv=p_rv,p_tv=p_tv,c_r=c_r,c_rv=c_rv,split_scale_dict=split_scale_dict,num_bins=num_bins,save_loc=plot_save_loc)
    if missclass:
        curr_sim_name = ""
        for sim in use_sims:
            curr_sim_name += sim
            curr_sim_name += "_"
        curr_sim_name += dst_type
        plot_miss_class_dist(p_corr_labels=p_corr_labels,p_ml_labels=p_ml_labels,p_r=p_r,p_rv=p_rv,p_tv=p_tv,c_r=c_r,c_rv=c_rv,split_scale_dict=split_scale_dict,num_bins=num_bins,save_loc=plot_save_loc,model_info=model_info,dataset_name=curr_sim_name)
            
    if per_err:
        with h5py.File(path_to_hdf5_file,"r") as f:
            dic_sim = {}
            grp_sim = f['config']['anl_prf']
            for f in grp_sim.attrs:
                dic_sim[f] = grp_sim.attrs[f]
            bins = dic_sim["r_bins_lin"]
        plot_per_err(bins,X["p_Scaled_radii"].values.compute(),y.compute().values.flatten(),preds.values,plot_save_loc, "$r/r_{200m}$","rad")
        # plot_per_err(bins,X["p_Radial_vel"].values.compute(),y.compute().values.flatten(),preds.values,plot_save_loc, "$v_r/v_{200m}$","rad_vel")
        # plot_per_err(bins,X["p_Tangential_vel"].values.compute(),y.compute().values.flatten(),preds.values,plot_save_loc, "$v_t/v_{200m}$","tang_vel")

def print_tree(bst,tree_num):# if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
    # stacked simulations such that they correspond to the larger dataset and not one specific simulation
    if len(use_sims) != 1:
        for i in range(1,len(use_sims)):
            if i < len(use_sims) - 1:
                halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
            else:
                halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
    fig, ax = plt.subplots(figsize=(400, 10))
    xgb.plot_tree(bst, num_trees=tree_num, ax=ax)
    plt.savefig(path_to_MLOIS + "Random_figs/tree_plot.png")
       
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
               
    sparta_mass_prf_all,sparta_mass_prf_orb,all_masses,bins = load_sprta_mass_prf(sim_splits,all_idxs,use_sims)
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
        accuracy = -1 * accuracy_score(y[only_inf], y_pred.iloc[only_inf].values)
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
        
        r_patt = r'(\d+-\d+|\d+)r'
        r_match = re.search(r_patt,split_string[3])

        
        v_patt = r'(\d+-\d+|\d+)v'
        v_match = re.search(v_patt, split_string[4])


        cond_string = split_string[0] + split_string[1] + split_string[2] 
        # can add these for more information per name
        #+ "r" + r_match.group(1) + "v" + v_match.group(1) + "s" + split_string[5]
        
        combined_name += cond_string
    
    return combined_name
    
def shap_with_filter(explainer, X, y, preds, fltr_dic = None, col_names = None, max_size=500):
    with timed("Filter DF"):
        full_filter = None
        if fltr_dic is not None:
            print("hi")
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
                    print("curr feature:",feature)
                    print("filter type",type(full_filter))
                    print("condition type:",type(condition))
                    if feature == next(iter(fltr_dic[next(iter(fltr_dic))])):
                        full_filter = condition
                    else:
                        full_filter &= condition

            X = X[full_filter]
            
        nrows = X.shape[0].compute()
        if nrows > max_size:
            sample = max_size / nrows
        else:
            sample = 1.0
            
        if sample > 0 and sample < 1:
            X = X.sample(frac=sample,random_state=rand_seed)
        
        if col_names != None:
            X.columns = col_names
        
        X = X.compute()

    return explainer(X), explainer.shap_values(X), X
    
    