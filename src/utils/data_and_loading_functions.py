import pickle
import h5py 
import os
import numpy as np
import multiprocessing as mp
from contextlib import contextmanager
import time
import re
import dask.dataframe as dd
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta
from functools import reduce
import ast
import configparser

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
config_dict = load_config(os.getcwd() + "/config.ini")

pickled_path = config_dict["PATHS"]["pickled_path"]

curr_sparta_file = config_dict["SPARTA_DATA"]["curr_sparta_file"]
sim_cosmol = config_dict["MISC"]["sim_cosmol"]
##################################################################################################################
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
else:
    sim_pat = r"cbol_l(\d+)_n(\d+)"
match = re.search(sim_pat, curr_sparta_file)
if match:
    sparta_name = match.group(0)
else:
    sparta_name = curr_sparta_file

num_processes = mp.cpu_count()
##################################################################################################################

def create_directory(path):
    os.makedirs(path,exist_ok=True)

@contextmanager
def timed(txt):
    print("Starting: " + txt)
    t0 = time.time()
    yield
    t1 = time.time()
    time_s = t1 - t0
    time_min = time_s / 60
    
    print("Finished: %s time: %.5fs, %.2f min\n" % (txt, time_s, time_min))

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

def save_pickle(data, path):
    with open(path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)

def load_pickle(path):
    if os.path.isfile(path):
        with open(path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    else:
        raise FileNotFoundError

def load_ptl_param(sparta_name, param_name, snap, snap_path):
    # save to folder containing pickled data to be accessed easily later
    file_path = pickled_path + str(snap) + "_" + str(sparta_name) + "/" + param_name + "_" + str(snap) + ".pickle" 
    create_directory(pickled_path + str(snap) +  "_" + str(sparta_name) + "/")
    
    # check if the file has already been pickled if so just load it
    try:
        ptl_param = load_pickle(file_path)
    except FileNotFoundError:
        ptl_param = readsnap(snap_path, param_name, 'dm')
        save_pickle(ptl_param,file_path)

    return ptl_param

def load_SPARTA_data(sparta_HDF5_path, param_path_list, sparta_name):
    create_directory(pickled_path + str(sparta_name) + "/")
    
    reload_sparta = False
    param_dict = {}

    all_save_names = []
    
    for param_path in param_path_list:
        save_name = "_".join(map(str, param_path))
        all_save_names.append(save_name)
        try:
            param = load_pickle(pickled_path + str(sparta_name) + "/" + save_name + ".pickle")
        except FileNotFoundError:
            if not reload_sparta:
                sparta_output = sparta.load(filename=sparta_HDF5_path, log_level= 0)
                reload_sparta = True
        
            param = reduce(lambda dct, key: dct[key], param_path, sparta_output)
            save_pickle(param,pickled_path + str(sparta_name) +  "/" + save_name + ".pickle")

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

def split_dataset_by_mass(halo_first, halo_n, path_to_dataset, curr_dataset):
    with h5py.File((path_to_dataset), 'r') as all_ptl_properties:
        first_prop = True
        for key in all_ptl_properties.keys():
            # only want the data important for the training now in the training dataset
            # dataset now has form HIPIDS, Orbit_Infall, Scaled Radii x num snaps, Rad Vel x num snaps, Tang Vel x num snaps
            if key != "Halo_first" and key != "Halo_n":
                if all_ptl_properties[key].ndim > 1:
                    for row in range(all_ptl_properties[key].ndim):
                        if first_prop:
                            curr_dataset = np.array(all_ptl_properties[key][halo_first:halo_first+halo_n,row])
                            first_prop = False
                        else:
                            curr_dataset = np.column_stack((curr_dataset,all_ptl_properties[key][halo_first:halo_first+halo_n,row])) 
                else:
                    if first_prop:
                        curr_dataset = np.array(all_ptl_properties[key][halo_first:halo_first+halo_n])
                        first_prop = False
                    else:
                        curr_dataset = np.column_stack((curr_dataset,all_ptl_properties[key][halo_first:halo_first+halo_n]))
    return curr_dataset

def save_dict_to_hdf5(hdf5_group, dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):  # Check if the value is a dictionary
            subgroup = hdf5_group.create_group(key)  # Create a subgroup
            save_dict_to_hdf5(subgroup, value)  # Recursively save the subdictionary
        else:
            hdf5_group.create_dataset(key, data=value)  

def save_to_hdf5(hdf5_file, data_name, dataset, chunk, max_shape):
    if isinstance(dataset, dict):
        hdf5_group = hdf5_file.create_group(data_name)
        # recursively deal with dictionaries
        save_dict_to_hdf5(hdf5_group, dataset)   
    else: 
        if data_name not in list(hdf5_file.keys()):
            hdf5_file.create_dataset(data_name, data = dataset, chunks = chunk, maxshape = max_shape, dtype=dataset.dtype)
        # with a new file adding on additional data to the datasets
        elif data_name in list(hdf5_file.keys()):
            hdf5_file[data_name].resize((hdf5_file[data_name].shape[0] + dataset.shape[0]), axis = 0)
            hdf5_file[data_name][-dataset.shape[0]:] = dataset   
        
def split_data_by_halo(client,frac, halo_props, ptl_data, return_halo=False):
    #TODO implement functionality for multiple sims
    halo_first = halo_props["Halo_first"]
    halo_n = halo_props["Halo_n"]

    num_halos = len(halo_first)
    
    split_halo = int(np.ceil(frac * num_halos))
    
    halo_1 = halo_props.loc[:split_halo]
    halo_2 = halo_props.loc[split_halo:]
    
    halo_2.loc[:,"Halo_first"] = halo_2["Halo_first"] - halo_2["Halo_first"].iloc[0]
    
    num_ptls = halo_n.loc[:split_halo].sum()
    
    ptl_1 = ptl_data.compute().iloc[:num_ptls,:]
    ptl_2 = ptl_data.compute().iloc[num_ptls:,:]

    
    scatter_ptl_1 = client.scatter(ptl_1)
    ptl_1 = dd.from_delayed(scatter_ptl_1)
    
    scatter_ptl_2 = client.scatter(ptl_2)
    ptl_2 = dd.from_delayed(scatter_ptl_2)
    
    if return_halo:
        return ptl_1, ptl_2, halo_1, halo_2
    else:
        return ptl_1, ptl_2
    
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

def find_closest_snap(value, cosmology, snap_loc, snap_dir_format, snap_format):
    tot_num_snaps = get_num_snaps(snap_loc)
    all_times = np.ones(tot_num_snaps) * -1000
    for i in range(tot_num_snaps):
        # Sometimes not all snaps exist
        if os.path.isdir(snap_loc + "snapdir_" + snap_dir_format.format(i)):
            all_times[i] = cosmology.age(readheader(snap_loc + "snapdir_" + snap_dir_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift'))
    idx = (np.abs(all_times - value)).argmin()
    return idx

def conv_halo_id_spid(my_halo_ids, sdata, snapshot):
    sparta_idx = np.zeros(my_halo_ids.shape[0], dtype = np.int32)
    for i, my_id in enumerate(my_halo_ids):
        sparta_idx[i] = int(np.where(my_id == sdata['halos']['id'][:,snapshot])[0])
    return sparta_idx

def get_comp_snap_info(t_dyn, t_dyn_step, cosmol, p_red_shift, all_sparta_z, snap_dir_format, snap_format, snap_path):
    c_snap_dict = {}
    # calculate one dynamical time ago and set that as the comparison snap
    curr_time = cosmol.age(p_red_shift)
    past_time = curr_time - (t_dyn_step * t_dyn)
    c_snap = find_closest_snap(past_time, cosmol, snap_path, snap_dir_format, snap_format)
    
    c_snap_dict["ptl_snap"] = c_snap
    
    # switch to comparison snap
    c_snap_path = snap_path + "/snapdir_" + snap_dir_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap)
        
    # get constants from pygadgetreader
    c_red_shift = readheader(c_snap_path, 'redshift')
    c_sparta_snap = np.abs(all_sparta_z - c_red_shift).argmin()
    c_snap_dict["sparta_snap"] = c_sparta_snap
    
    print("Complementary snapshot:", c_snap, "Complementary redshift:", c_red_shift)
    print("Corresponding SPARTA loc:", c_sparta_snap, "SPARTA redshift:",all_sparta_z[c_sparta_snap])

    c_scale_factor = 1/(1+c_red_shift)
    c_rho_m = cosmol.rho_m(c_red_shift)
    c_hubble_const = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
    
    c_snap_dict["red_shift"] = c_red_shift
    c_snap_dict["scale_factor"] = c_scale_factor
    c_snap_dict["rho_m"] = c_rho_m
    c_snap_dict["hubble_const"] = c_hubble_const

    return c_snap_dict

def split_orb_inf(data, labels):
    infall = data[np.where(labels == 0)[0]]
    orbit = data[np.where(labels == 1)[0]]
    return infall, orbit

def parse_ranges(ranges_str):
    ranges = []
    for part in ranges_str.split(','):
        start, end = map(float, part.split('-'))
        ranges.append((start, end))
    return ranges
def create_nu_string(nu_list):
    return '_'.join('-'.join(map(str, tup)) for tup in nu_list)


