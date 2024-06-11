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

from utils.data_and_loading_functions import load_or_pickle_SPARTA_data, find_closest_z, conv_halo_id_spid, timed
from utils.visualization_functions import compare_density_prf, plot_r_rv_tv_graph, plot_misclassified
from sparta_tools import sparta # type: ignore

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
sim_pat = r"cbol_l(\d+)_n(\d+)"
match = re.search(sim_pat, curr_sparta_file)
if match:
    sparta_name = match.group(0)
path_to_snaps = path_to_snaps + sparta_name + "/"
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
model_type = config["XGBOOST"]["model_type"]
train_rad = config.getint("XGBOOST","training_rad")
nu_splits = config["XGBOOST"]["nu_splits"]

nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

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

def get_CUDA_cluster():
    cluster = LocalCUDACluster(
                               device_memory_limit='10GB',
                               jit_unspill=True)
    client = Client(cluster)
    return client

def make_preds(client, bst, X, y_np, report_name="Classification Report", print_report=False):
    #X = da.from_array(X_np,chunks=(chunk_size,X_np.shape[1]))
    
    preds = dxgb.inplace_predict(client, bst, X).compute()
    preds = np.round(preds)
    preds = preds.astype(np.int8)
    
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

def split_dataframe(df, max_size):
    # Split a dataframe so that each one is below a maximum memory size
    total_size = df.memory_usage(index=True).sum()
    num_splits = int(np.ceil(total_size / max_size))
    chunk_size = int(np.ceil(len(df) / num_splits))
    print("splitting Dataframe into:",num_splits,"dataframes")
    
    split_dfs = []
    for i in range(0, len(df), chunk_size):
        split_dfs.append(df.iloc[i:i + chunk_size])
    
    return split_dfs

def reform_datasets(client,config_params,sim,folder_path,rad_cut=None,filter_nu=None):
    ptl_files = sort_files(folder_path + "/ptl_info/")
    
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
        ptl_df = cut_by_rad(ptl_df, rad_cut=rad_cut)

        # Calculate scale position weight
        scal_pos_weight = calc_scal_pos_weight(ptl_df)

        # If the dataframe is too large split it up
        max_mem = config_params["HDF5 Mem Size"]
        if ptl_df.memory_usage(index=True).sum() > max_mem:
            ptl_dfs = split_dataframe(ptl_df, max_mem)
        else:
            ptl_dfs = [ptl_df]
        
        return ptl_dfs,scal_pos_weight

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

def load_data(client, sims, dset_name, rad_cut=None, filter_nu=False):
    #rad_cut set to None so that the full dataset is used (determined by the search radius initially used) 
    dask_dfs = []
    all_scal_pos_weight = []
    
    for sim in sims:
        files_loc = path_to_calc_info + sim + "/" + dset_name
        with open(path_to_calc_info + sim + "/config.pickle","rb") as f:
            config_params = pickle.load(f)
        
        if rad_cut == None:
            rad_cut = config_params["search_rad"]
        
        if dset_name == "Full":
            with timed("Reformed " + "Train" + " Dataset: " + sim):    
                files_loc = path_to_calc_info + sim + "/" + "Train"
                ptl_ddf,sim_scal_pos_weight = reform_datasets(client,config_params,sim,files_loc,rad_cut=rad_cut,filter_nu=filter_nu)   
            print("num of partitions:",ptl_ddf.npartitions)
            all_scal_pos_weight.append(sim_scal_pos_weight)
            dask_dfs.append(ptl_ddf)
            with timed("Reformed " + "Test" + " Dataset: " + sim):
                files_loc = path_to_calc_info + sim + "/" + "Test"
                ptl_ddf,sim_scal_pos_weight = reform_datasets(client,config_params,sim,files_loc,rad_cut=rad_cut,filter_nu=filter_nu)   
            print("num of partitions:",ptl_ddf.npartitions)
            all_scal_pos_weight.append(sim_scal_pos_weight)
            dask_dfs.append(ptl_ddf)
        else:
            with timed("Reformed " + dset_name + " Dataset: " + sim):                   
                ptl_ddf,sim_scal_pos_weight = reform_datasets(client,config_params,sim,files_loc,rad_cut=rad_cut,filter_nu=filter_nu)   
            print("num of partitions:",ptl_ddf.npartitions)
            all_scal_pos_weight.append(sim_scal_pos_weight)
            dask_dfs.append(ptl_ddf)
    
    act_scale_pos_weight = np.average(np.array(all_scal_pos_weight))
    
    return dd.concat(dask_dfs),act_scale_pos_weight
    

def eval_model(model_info, client, model, use_sims, dst_type, X, y, halo_ddf, combined_name, plot_save_loc, dens_prf = False, r_rv_tv = False, misclass=False): 
    with timed("Predictions"):
        preds = make_preds(client, model, X, y, report_name="Report", print_report=False)
    
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
                    
        dens_prf_all_list = []
        dens_prf_1halo_list = []
        all_masses = []
        
        for i,sim in enumerate(use_sims):
            # Get the halo indices corresponding to this simulation
            if i < len(use_sims) - 1:
                use_idxs = all_idxs[sim_splits[i]:sim_splits[i+1]]
            else:
                use_idxs = all_idxs[sim_splits[i]:]
            
            # find the snapshots for this simulation
            snap_pat = r"(\d+)to(\d+)"
            match = re.search(snap_pat, sim)
            if match:
                curr_snap_list = [match.group(1), match.group(2)] 
            sim_pat = r"cbol_l(\d+)_n(\d+)"
            match = re.search(sim_pat, sim)
            if match:
                sparta_name = match.group(0)
            sim_search_pat = sim_pat + r"_(\d+)r200m"
            match = re.search(sim_search_pat, sim)
            if match:
                sparta_search_name = match.group(0)
            
            with open(path_to_calc_info + sim + "/config.pickle", "rb") as file:
                config_dict = pickle.load(file)
                
                curr_z = config_dict["p_snap_info"]["red_shift"][()]
                curr_snap_format = config_dict["snap_format"]
                new_p_snap, curr_z = find_closest_z(curr_z,path_to_snaps + sparta_name + "/",curr_snap_format)
                p_scale_factor = 1/(1+curr_z)
                
            with h5py.File(path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5","r") as f:
                dic_sim = {}
                grp_sim = f['simulation']

                for attr in grp_sim.attrs:
                    dic_sim[attr] = grp_sim.attrs[attr]
            
            all_red_shifts = dic_sim['snap_z']
            p_sparta_snap = np.abs(all_red_shifts - curr_z).argmin()
            print(all_red_shifts.shape)
            print(curr_z,p_sparta_snap)
            halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, parent_id, ptl_mass = load_or_pickle_SPARTA_data(sparta_search_name, p_scale_factor, curr_snap_list[0], p_sparta_snap)

            use_halo_ids = halos_id[use_idxs]
            
            sparta_output = sparta.load(filename=path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5", halo_ids=use_halo_ids, log_level=1)
            print(path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5")
            print(use_halo_ids.shape)
            new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta resort the indices
            
            dens_prf_all_list.append(sparta_output['anl_prf']['M_all'][new_idxs,p_sparta_snap,:])
            dens_prf_1halo_list.append(sparta_output['anl_prf']['M_1halo'][new_idxs,p_sparta_snap,:])
            all_masses.append(ptl_mass)

        dens_prf_all = np.vstack(dens_prf_all_list)
        dens_prf_1halo = np.vstack(dens_prf_1halo_list)
        
        bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
        bins = np.insert(bins, 0, 0)
        
        compare_density_prf(sim_splits,radii=X["p_Scaled_radii"].values.compute(), halo_first=halo_first, halo_n=halo_n, act_mass_prf_all=dens_prf_all, act_mass_prf_orb=dens_prf_1halo, mass=all_masses, orbit_assn=preds, prf_bins=bins, title="", save_location=plot_save_loc, use_mp=True, save_graph=True)
    
    if r_rv_tv:
        plot_r_rv_tv_graph(preds, X["p_Scaled_radii"].values.compute(), X["p_Radial_vel"].values.compute(), X["p_Tangential_vel"].values.compute(), y, title="", num_bins=num_bins, save_location=plot_save_loc)
    
    if misclass:
        preds = preds.values
        y = y.compute().values.flatten()
        c_rad = X["c_Scaled_radii"].values.compute()
        print(c_rad.shape)
        not_nan_mask = ~np.isnan(c_rad)
        print(np.where(not_nan_mask)[0].shape)
        
        plot_misclassified(p_corr_labels=y, p_ml_labels=preds, p_r=X["p_Scaled_radii"].values.compute(), p_rv=X["p_Radial_vel"].values.compute(), p_tv=X["p_Tangential_vel"].values.compute(), c_r=X["c_Scaled_radii"].values.compute(), c_rv=X["c_Radial_vel"].values.compute(), c_tv=X["c_Tangential_vel"].values.compute(),title="",num_bins=num_bins,save_location=plot_save_loc,model_info=model_info,dataset_name=dst_type + "_" + combined_name)
    

def print_tree(bst,tree_num):
    fig, ax = plt.subplots(figsize=(400, 10))
    xgb.plot_tree(bst, num_trees=tree_num, ax=ax)
    plt.savefig(path_to_MLOIS + "Random_figs/tree_plot.png")