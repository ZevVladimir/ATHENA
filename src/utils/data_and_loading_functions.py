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

##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")

SPARTA_output_path = config["PATHS"]["SPARTA_output_path"]
pickled_path = config["PATHS"]["pickled_path"]

curr_sparta_file = config["MISC"]["curr_sparta_file"]
sim_cosmol = config["MISC"]["sim_cosmol"]

reset_lvl = config.getint("SEARCH","reset")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
##################################################################################################################
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
else:
    sim_pat = r"cbol_l(\d+)_n(\d+)"
match = re.search(sim_pat, curr_sparta_file)
if match:
    sparta_name = match.group(0)

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
    
    print("Finished: %s time: %.5fs, %.2f min" % (txt, time_s, time_min))

def clean_dir(path):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files at location:",path)

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

def load_SPARTA_data(sparta_HDF5_path, param_path_list, sparta_name, snap, halo_ids=None):
    create_directory(pickled_path + str(snap) +  "_" + str(sparta_name) + "/")
    
    reload_sparta = False
    param_dict = {}
    all_save_names = []
    
    for param_path in param_path_list:
        save_name = "_".join(map(str, param_path))
        all_save_names.append(save_name)
        try:
            param = load_pickle(pickled_path + str(snap) + "_" + str(sparta_name) + "/" + save_name + ".pickle")
        except FileNotFoundError:
            if not reload_sparta:
                if halo_ids is None:
                    sparta_output = sparta.load(filename=sparta_HDF5_path, log_level= 0)
                else:
                    #TODO Add functionality of conv_halo_id_spid
                    sparta_output = sparta.load(filename=sparta_HDF5_path, halo_ids=halo_ids, log_level=0)
                        
            param = reduce(lambda dct, key: dct[key], param_path, sparta_output)
            save_pickle(param,pickled_path + str(snap) + "_" + str(sparta_name) +  "/" + save_name + ".pickle")
        
        param_dict[save_name] = param
    
    return param_dict,all_save_names

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
    
def find_closest_z(value,snap_loc,snap_dir_format,snap_format):
    all_z = np.ones(total_num_snaps) * -1000
    for i in range(total_num_snaps):
        # Sometimes not all snaps exist
        if os.path.isdir(snap_loc + "snapdir_" + snap_dir_format.format(i)):
            all_z[i] = readheader(snap_loc + "snapdir_" + snap_dir_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift')

    idx = (np.abs(all_z - value)).argmin()
    return idx, all_z[idx]

def find_closest_snap(value, cosmology, snap_loc, snap_dir_format, snap_format):
    all_times = np.ones(total_num_snaps) * -1000
    for i in range(total_num_snaps):
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

def get_comp_snap(t_dyn, t_dyn_step, snapshot_list, cosmol, p_red_shift, all_red_shifts, snap_dir_format, snap_format, snap_loc):
    # calculate one dynamical time ago and set that as the comparison snap
    curr_time = cosmol.age(p_red_shift)
    past_time = curr_time - (t_dyn_step * t_dyn)
    c_snap = find_closest_snap(past_time, cosmol, snap_loc, snap_dir_format, snap_format)
    snapshot_list.append(c_snap)

    # switch to comparison snap
    c_snap_path = snap_loc + "/snapdir_" + snap_dir_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap)
        
    # get constants from pygadgetreader
    c_red_shift = readheader(c_snap_path, 'redshift')
    c_sparta_snap = np.abs(all_red_shifts - c_red_shift).argmin()
    print("Complementary snapshot:", c_snap, "Complementary redshift:", c_red_shift)
    print("Corresponding SPARTA loc:", c_sparta_snap, "SPARTA redshift:",all_red_shifts[c_sparta_snap])

    c_scale_factor = 1/(1+c_red_shift)
    c_rho_m = cosmol.rho_m(c_red_shift)
    c_hubble_constant = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
    # c_box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
    # c_box_size = c_box_size * 10**3 * c_scale_factor #convert to Kpc/h physical
    # c_box_size = c_box_size + 0.001 # NEED TO MAKE WORK FOR PARTICLES ON THE VERY EDGE

    return c_snap, c_sparta_snap, c_rho_m, c_red_shift, c_scale_factor, c_hubble_constant

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

