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

##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
pickled_path = config["PATHS"]["pickled_path"]

curr_sparta_file = config["MISC"]["curr_sparta_file"]
snap_dir_format = config["MISC"]["snap_dir_format"]
snap_format = config["MISC"]["snap_format"]
pickle_data = config.getboolean("MISC","pickle_data")
debug_gen = config.getboolean("MISC","debug_gen")

reset_lvl = config.getint("SEARCH","reset")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
##################################################################################################################

def create_directory(path):
    os.makedirs(path,exist_ok=True)

# Prints out how long something takes
@contextmanager
def timed(txt):
    print("Starting: " + txt)
    t0 = time.time()
    yield
    t1 = time.time()
    time_s = t1 - t0
    time_min = time_s / 60
    
    print("Finished: %s time: %.5fs, %.2f min" % (txt, time_s, time_min))

# Removes all files in a directory
def clean_dir(path):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files at location:",path)

def save_pickle(data, file_path):
    """
    Save data to a pickle file.
    """
    with open(file_path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)
        
# Loads an inputted particle property after checking if there is a pickled copy
def load_ptl_param(sparta_name, ptl_property, snapshot, snapshot_path):
    # save to folder containing pickled data to be accessed easily later
    file_path = pickled_path + str(snapshot) + "_" + str(sparta_name) + "/" + ptl_property + "_" + str(snapshot) + ".pickle" 
    create_directory(pickled_path + str(snapshot) +  "_" + str(sparta_name) + "/")
    
    # check if the file has already been pickled if so just load it
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            ptl_info = pickle.load(pickle_file)
    # otherwise load the specific information from the particle data and save it as a pickle file
    else:
        ptl_info = readsnap(snapshot_path, ptl_property, 'dm')
        if pickle_data:
            save_pickle(ptl_info,file_path)
    return ptl_info

def load_pickle_or_set_reload(file_path, reload_flag):
    """
    Load a pickle file if it exists, otherwise set reload_flag to True.
    """
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            return pickle.load(pickle_file), reload_flag
    return None, True

def load_SPARTA_data(SPARTA_hdf5_path, sparta_name, scale_factor, snap, sparta_snap):
    """
    Load SPARTA data from pickled files or regenerate from source if not available.
    """
    directory_path = os.path.join(pickled_path, f"{snap}_{sparta_name}")
    create_directory(directory_path)

    # File names and associated variables
    file_vars = {
        "halos_pos.pickle": "halos_pos",
        "halos_last_snap.pickle": "halos_last_snap",
        "halos_r200m.pickle": "halos_r200m",
        "halos_id.pickle": "halos_id",
        "halos_status.pickle": "halos_status",
        "pid.pickle": "pid",
        "ptl_mass.pickle": "ptl_mass",
    }

    data = {}
    reload_sparta = False

    # Attempt to load each file
    for file_name, var_name in file_vars.items():
        file_path = os.path.join(directory_path, file_name)
        data[var_name], reload_sparta = load_pickle_or_set_reload(file_path, reload_sparta)
        if reload_sparta:
            break

    if reload_sparta:
        sparta_output = sparta.load(filename=SPARTA_hdf5_path, log_level=0)

        # Extract and process SPARTA data
        data["halos_pos"] = (
            sparta_output['halos']['position'][:, sparta_snap, :] * 10**3 * scale_factor
        )  # Convert to kpc/h and physical
        data["halos_last_snap"] = sparta_output['halos']['last_snap'][:]
        data["halos_r200m"] = sparta_output['halos']['R200m'][:, sparta_snap]
        data["halos_id"] = sparta_output['halos']['id'][:, sparta_snap]
        data["halos_status"] = sparta_output['halos']['status'][:, sparta_snap]
        data["pid"] = sparta_output['halos']['pid'][:, sparta_snap]
        data["ptl_mass"] = sparta_output["simulation"]["particle_mass"]

        # Save processed data to pickle files
        if pickle_data:
            for file_name, var_name in file_vars.items():
                save_pickle(data[var_name], os.path.join(directory_path, file_name))

    return (
        data["halos_pos"],
        data["halos_r200m"],
        data["halos_id"],
        data["halos_status"],
        data["halos_last_snap"],
        data["pid"],
        data["ptl_mass"],
    )

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

def get_comp_snap(SPARTA_hdf5_path, t_dyn, t_dyn_step, cosmol, p_red_shift, all_red_shifts, snap_dir_format, snap_format, snap_loc):
    # calculate one dynamical time ago and set that as the comparison snap
    curr_time = cosmol.age(p_red_shift)
    past_time = curr_time - (t_dyn_step * t_dyn)
    c_snap = find_closest_snap(past_time, cosmol, snap_loc, snap_dir_format, snap_format)

    # switch to comparison snap
    snap_path = snap_loc + "/snapdir_" + snap_dir_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap)
        
    # get constants from pygadgetreader
    c_red_shift = readheader(snap_path, 'redshift')
    c_sparta_snap = np.abs(snap_path - c_red_shift).argmin()
    if debug_gen:
        print("\nComplementary Snapshot:\nParticle snapshot number:", c_snap, "SPARTA snapshot number:",c_sparta_snap)
        print("Particle redshift:", c_red_shift, "SPARTA redshift:",all_red_shifts[c_sparta_snap],"\n")

    c_scale_factor = 1/(1+c_red_shift)
    c_rho_m = cosmol.rho_m(c_red_shift)
    c_hubble_constant = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
    # c_box_size = readheader(snap_path, 'boxsize') #units Mpc/h comoving
    # c_box_size = c_box_size * 10**3 * c_scale_factor #convert to Kpc/h physical
    # c_box_size = c_box_size + 0.001 # NEED TO MAKE WORK FOR PARTICLES ON THE VERY EDGE
    
    if reset_lvl == 3:
        clean_dir(pickled_path + str(c_snap) + "_" + curr_sparta_file + "/")
    # load particle data and SPARTA data for the comparison snap
    with timed("c_snap ptl load"):
        c_ptls_pid = load_ptl_param(curr_sparta_file, "pid", str(c_snap), snap_path) * 10**3 * c_scale_factor # kpc/h
        c_ptls_vel = load_ptl_param(curr_sparta_file, "vel", str(c_snap), snap_path) # km/s
        c_ptls_pos = load_ptl_param(curr_sparta_file, "pos", str(c_snap), snap_path)
    with timed("c_snap SPARTA load"):
        c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap, c_parent_id, mass = load_SPARTA_data(SPARTA_hdf5_path,curr_sparta_file, c_scale_factor, c_snap, c_sparta_snap)

    return c_snap, c_sparta_snap, c_rho_m, c_red_shift, c_scale_factor, c_hubble_constant, c_ptls_pid, c_ptls_vel, c_ptls_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap

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
