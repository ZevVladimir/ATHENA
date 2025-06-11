import pickle
import joblib
import h5py 
import os
import pandas as pd
import numpy as np
import re
from dask import delayed
import dask.dataframe as dd
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta
from functools import reduce
import ast
import configparser
from pygadgetreader import readheader
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import peaks
import time
from contextlib import contextmanager
from colossus.utils import constants

from .calc_fxns import calc_scal_pos_weight

def parse_value(value):
    """Convert value to appropriate type (list, int, float, bool, str)."""
    try:
        return ast.literal_eval(value)  # Safely evaluates lists, ints, floats, etc.
    except (ValueError, SyntaxError):
        return value  # Keep as string if eval fails

# Load the config file into one dictionary separated by the sections as indicated in the config file
def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {key: parse_value(value) for key, value in config[section].items()}
    
    return config_dict
##################################################################################################################
# LOAD CONFIG PARAMETERS
config_params = load_config(os.getcwd() + "/config.ini")

pickled_path = config_params["PATHS"]["pickled_path"]
ML_dset_path = config_params["PATHS"]["ml_dset_path"]

SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]

save_intermediate_data = config_params["MISC"]["save_intermediate_data"]
##################################################################################################################

def save_pickle(data, path):
    with open(path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)

def load_pickle(path):
    if os.path.isfile(path):
        with open(path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    else:
        raise FileNotFoundError
    
# joblib is better for large np arrays
def save_joblib(data, path):
    joblib.dump(data,path)
    
def load_joblib(path):
    if os.path.isfile(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError

def load_ptl_param(sparta_name, param_name, snap, snap_path, save_data=False):
    # save to folder containing pickled data to be accessed easily later
    file_path = pickled_path + str(snap) + "_" + str(sparta_name) + "/" + param_name + "_" + str(snap) + ".pickle" 
    create_directory(pickled_path + str(snap) +  "_" + str(sparta_name) + "/")
    
    # check if the file has already been pickled if so just load it
    try:
        ptl_param = load_joblib(file_path)
    except FileNotFoundError:
        ptl_param = readsnap(snap_path, param_name, 'dm')
        if save_data:
            save_joblib(ptl_param,file_path)

    return ptl_param

def load_SPARTA_data(sparta_HDF5_path, param_path_list, sparta_name, save_data=False):
    create_directory(pickled_path + str(sparta_name) + "/")
    
    reload_sparta = False
    param_dict = {}

    all_save_names = []
    
    for param_path in param_path_list:
        save_name = "_".join(map(str, param_path))
        all_save_names.append(save_name)
        try:
            param = load_joblib(pickled_path + str(sparta_name) + "/" + save_name + ".pickle")
        except FileNotFoundError:
            if not reload_sparta:
                sparta_output = sparta.load(filename=sparta_HDF5_path, log_level= 0)
                reload_sparta = True
        
            param = reduce(lambda dct, key: dct[key], param_path, sparta_output)
            if save_data:
                save_joblib(param,pickled_path + str(sparta_name) +  "/" + save_name + ".pickle")

        param_dict[save_name] = param

    return param_dict,all_save_names

def load_RSTAR_data(rockstar_loc, param_list, curr_z):
    rstar_file_loc = find_closest_a_rstar(curr_z, rockstar_loc)
    
    param_data = {param: [] for param in param_list}
    col_index_map = {}

    with open(rstar_file_loc, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Only parse the column names once
                if not col_index_map:
                    header = line[1:].strip().split()
                    for i, entry in enumerate(header):
                        name = entry.split('(')[0]
                        col_index_map[name] = i
                continue  # Skip all header lines


            # Split data line and collect requested values
            parts = line.split()
            for param in param_list:
                idx = col_index_map.get(param)
                if idx is not None and len(parts) > idx:
                    param_data[param].append(float(parts[idx]))

    for param in param_list:
        param_data[param] = np.array(param_data[param])

    return param_data 

# Using dynamical time find the time tdyn_step dynamical times ago and convert that to redshift
def get_past_z(cosmol, init_z, tdyn_step, mass_def = "200m"):
    tdyn = mass_so.dynamicalTime(init_z,mass_def,"crossing")
    
    curr_time = cosmol.age(init_z)
    past_time = curr_time - (tdyn_step * tdyn)
    past_z = cosmol.age(past_time,inverse=True)
    return past_z

def alt_get_past_z(cosmol, halo_r200m, init_z, tdyn_step, little_h, mass_def = "200m"):
    G = constants.G
    halo_m200m = mass_so.R_to_M(halo_r200m, init_z, mass_def)
    curr_v200m = np.sqrt((G * halo_m200m)/halo_r200m)
    tdyn = ((2*halo_r200m)/curr_v200m) * little_h

    curr_time = cosmol.age(init_z)
    past_time = curr_time - (tdyn_step * tdyn)
    past_z = cosmol.age(past_time,inverse=True)
    return past_z

def get_comp_snap_info(cosmol, past_z, all_sparta_z, snap_dir_format, snap_format, snap_path):
    c_snap_dict = {}
    
    c_snap, idx = find_closest_z_snap(past_z,snap_path,snap_dir_format,snap_format)
    
    c_snap_dict["ptl_snap"] = c_snap

    # get constants from pygadgetreader
    c_sparta_snap = np.abs(all_sparta_z - past_z).argmin()
    c_snap_dict["sparta_snap"] = c_sparta_snap
    
    snap_z = readheader(snap_path + "snapdir_" + snap_dir_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap), 'redshift')
    
    print("Complementary snapshot:", c_snap, "Complementary redshift:", snap_z)
    print("Corresponding SPARTA loc:", c_sparta_snap, "SPARTA redshift:",all_sparta_z[c_sparta_snap])

    c_scale_factor = 1/(1+past_z)
    c_rho_m = cosmol.rho_m(past_z)
    c_hubble_const = cosmol.Hz(past_z) * 0.001 # convert to units km/s/kpc
    
    c_snap_dict["red_shift"] = past_z
    c_snap_dict["scale_factor"] = c_scale_factor
    c_snap_dict["rho_m"] = c_rho_m
    c_snap_dict["hubble_const"] = c_hubble_const

    return c_snap_dict

# Returns a simulation's mass used and the redshift of the primary snapshot
def load_sim_mass_pz(sim,config_params):
    sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
    
    curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" +  sparta_search_name + ".hdf5"    
    
    with h5py.File(curr_sparta_HDF5_path,"r") as f:
        dic_sim = {}
        grp_sim = f['simulation']
        for f in grp_sim.attrs:
            dic_sim[f] = grp_sim.attrs[f]
    
    p_red_shift = config_params["all_snap_info"]["prime_snap_info"]["red_shift"]
    
    param_paths = [["simulation","particle_mass"]]
            
    sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, save_data=save_intermediate_data)
    ptl_mass = sparta_params[sparta_param_names[0]]
    
    return ptl_mass, p_red_shift

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
        
        with open(ML_dset_path + sim + "/dset_params.pickle", "rb") as file:
            dset_params = pickle.load(file)
            
        p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"]
        
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"      
        
        param_paths = [["simulation","particle_mass"],["anl_prf","M_all"],["anl_prf","M_1halo"],["halos","R200m"],["config","anl_prf","r_bins_lin"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, save_data=save_intermediate_data)
        ptl_mass = sparta_params[sparta_param_names[0]]

        mass_prf_all_list.append(sparta_params[sparta_param_names[1]][use_idxs,p_sparta_snap,:])
        mass_prf_1halo_list.append(sparta_params[sparta_param_names[2]][use_idxs,p_sparta_snap,:])
        all_r200m_list.append(sparta_params[sparta_param_names[3]][use_idxs,p_sparta_snap])

        all_masses.append(ptl_mass)

    mass_prf_all = np.vstack(mass_prf_all_list)
    mass_prf_1halo = np.vstack(mass_prf_1halo_list)
    all_r200m = np.concatenate(all_r200m_list)
    
    # We assume bins are the same from each sparta simulation
    bins = sparta_params[sparta_param_names[4]]
    bins = np.insert(bins, 0, 0)

    if ret_r200m:
        return mass_prf_all,mass_prf_1halo,all_masses,bins,all_r200m
    else:
        return mass_prf_all,mass_prf_1halo,all_masses,bins

# Loads all the data for the inputted list of simulations into one dataframe. Finds the scale position weight for the dataset and any adjusted weighting for it if desired
def load_ML_dsets(client, sims, dset_name, sim_cosmol_list, prime_snap, file_lim=0, filter_nu=False, nu_splits=None):
    dask_dfs = []
    all_scal_pos_weight = []
        
    for i,sim in enumerate(sims):
        with open(ML_dset_path + sim + "/dset_params.pickle","rb") as f:
            dset_params = pickle.load(f)
        # Get mass and redshift for this simulation
        ptl_mass, use_z = load_sim_mass_pz(sim,dset_params)
        max_mem = int(np.floor(dset_params["HDF5 Mem Size"] / 2))
        
        if dset_name == "Full":
            datasets = ["Train", "Test"]
        else:
            datasets = [dset_name]

        for dataset in datasets:
            with timed(f"Reformed {dataset} Dataset: {sim}"): 
                dataset_path = f"{ML_dset_path}{sim}/{dataset}"
                ptl_ddf,sim_scal_pos_weight = reform_dsets_nested(client,ptl_mass,use_z,max_mem,sim_cosmol_list[i],dataset_path,prime_snap,file_lim=file_lim,filter_nu=filter_nu,nu_splits=nu_splits)  
                all_scal_pos_weight.append(sim_scal_pos_weight)
                dask_dfs.append(ptl_ddf)
                    
    all_scal_pos_weight = np.average(np.concatenate([np.array(sublist).flatten() for sublist in all_scal_pos_weight]))
    act_scale_pos_weight = np.average(all_scal_pos_weight)

    all_dask_dfs = dd.concat(dask_dfs)
    
    return all_dask_dfs,act_scale_pos_weight

@contextmanager
def timed(txt):
    print("Starting: " + txt)
    t0 = time.time()
    yield
    t1 = time.time()
    time_s = t1 - t0
    time_min = time_s / 60
    
    print("Finished: %s time: %.5fs, %.2f min\n" % (txt, time_s, time_min))
    
def create_directory(path):
    os.makedirs(path,exist_ok=True)

def clean_dir(path):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files at location:",path)
        
# Depairs the hipids into (pids, halo_idxs) We use np.vectorize because depair returns two values and we want that split in two
def depair_np(z):
    """
    Modified from https://github.com/perrygeo/pairing to use numpy functions
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    w = np.floor((np.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = (z - t).astype(int)
    x = (w - y).astype(int)
    # assert z != pair(x, y, safe=False):
    return x, y

# Obtains the highest number snapshot in the given folder path
# We can't just get the total number of folders as there might be snapshots missing
def get_num_snaps(path):
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    numbers = [int(re.search(r'\d+', d).group()) for d in folders if re.search(r'\d+', d)]
    max_number = max(numbers, default=None)
    return (max_number+1)

def find_closest_z_snap(value,snap_loc,snap_dir_format,snap_format):
    tot_num_snaps = get_num_snaps(snap_loc)
    all_z = np.ones(tot_num_snaps) * -1000
    for i in range(tot_num_snaps):
        # Sometimes not all snaps exist
        if os.path.isdir(snap_loc + "snapdir_" + snap_dir_format.format(i)):
            all_z[i] = readheader(snap_loc + "snapdir_" + snap_dir_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift')

    idx = (np.abs(all_z - value)).argmin()
    return idx, all_z[idx]

# Returns the path of the rockstar file that has the closest redshift to the inputted value
def find_closest_a_rstar(z,rockstar_loc):
    all_a = []
    for filename in os.listdir(rockstar_loc):
        match = re.search(r"hlist_(\d+\.\d+)\.list", filename)
        if match:
            a_val = float(match.group(1))
            all_a.append(a_val)

    idx = (np.abs(all_a - 1/(1+z))).argmin()
    print(rockstar_loc + "/hlist_" + str(all_a[idx]) + ".list")
    return rockstar_loc + "/hlist_" + str(all_a[idx]) + ".list"

def find_closest_snap(value, cosmol, snap_loc, snap_dir_format, snap_format):
    tot_num_snaps = get_num_snaps(snap_loc)
    all_times = np.ones(tot_num_snaps) * -1000
    for i in range(tot_num_snaps):
        # Sometimes not all snaps exist
        if os.path.isdir(snap_loc + "snapdir_" + snap_dir_format.format(i)):
            all_times[i] = cosmol.age(readheader(snap_loc + "snapdir_" + snap_dir_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift'))
    idx = (np.abs(all_times - value)).argmin()
    return idx
    
def conv_halo_id_spid(my_halo_ids, sdata, snapshot):
    sparta_idx = np.zeros(my_halo_ids.shape[0], dtype = np.int32)
    for i, my_id in enumerate(my_halo_ids):
        sparta_idx[i] = int(np.where(my_id == sdata['halos']['id'][:,snapshot])[0])
    return sparta_idx    

def parse_ranges(ranges_str):
    ranges = []
    for part in ranges_str.split(','):
        start, end = map(float, part.split('-'))
        ranges.append((start, end))
    return ranges

def create_nu_string(nu_list):
    return '_'.join('-'.join(map(str, tup)) for tup in nu_list)

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

def set_cosmology(sim_cosmol):
    if sim_cosmol == "planck13-nbody":
        cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
    else:
        cosmol = cosmology.setCosmology(sim_cosmol)
    return cosmol

def count_n_ptls_used(sims):
    for sim in sims:
        print("Curr Sim:",sim)
        print("Training numbers")
        for i,file in enumerate(os.listdir(ML_dset_path + sim + "/" + "Train" + "/halo_info/")):
            df = pd.read_hdf(ML_dset_path + sim + "/" + "Train" + "/halo_info/" + file)
            print("File " + str(i) + ":",df["Halo_n"].values.sum())
        print("Testing numbers")
        for i,file in enumerate(os.listdir(ML_dset_path + sim + "/" + "Test" + "/halo_info/")):
            df = pd.read_hdf(ML_dset_path + sim + "/" + "Test" + "/halo_info/" + file)
            print("File " + str(i) + ":",df["Halo_n"].values.sum())
        
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

def split_orb_inf(data, labels):
    infall = data[np.where(labels == 0)[0]]
    orbit = data[np.where(labels == 1)[0]]
    return infall, orbit

# Goes through a folder where a dataset's hdf5 files are stored and reforms them into one pandas dataframe (in order)
def reform_dset_dfs(folder_path):
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
    
# Split a dataframe so that each one is below an inputted maximum memory size
def split_dataframe(df, max_size):
    total_size = df.memory_usage(index=True).sum()
    num_splits = int(np.ceil(total_size / max_size))
    chunk_size = int(np.ceil(len(df) / num_splits))
    print("splitting Dataframe into:",num_splits,"dataframes")
    
    split_dfs = []
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
def reform_dsets_nested(client, ptl_mass, use_z, max_mem, sim_cosmol, folder_path, prime_snap, file_lim=0, filter_nu=None, nu_splits=None):
    snap_n_files = len(os.listdir(folder_path + "/ptl_info/" + str(prime_snap)+"/"))
    n_files = snap_n_files

    if file_lim > 0:
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

def load_all_sim_cosmols(curr_sims):
    all_sim_cosmol_list = []
    for sim in curr_sims:
        dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
        all_sim_cosmol_list.append(dset_params["cosmology"])
    
    return all_sim_cosmol_list

def load_all_tdyn_steps(curr_sims):
    all_tdyn_steps_list = []
    for sim in curr_sims:
        dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
        all_tdyn_steps_list.append(dset_params["t_dyn_steps"])
    
    return all_tdyn_steps_list
