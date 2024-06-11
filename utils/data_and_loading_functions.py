import pickle
import h5py 
import os
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from itertools import repeat
from contextlib import contextmanager
import time
import re

def create_directory(path):
    os.makedirs(path,exist_ok=True)
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
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

snap_format = config["MISC"]["snap_format"]

reset_lvl = config.getint("SEARCH","reset")
global prim_only
prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("XGBOOST","test_halos_ratio")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
num_processes = mp.cpu_count()
##################################################################################################################
import sys
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader # type: ignore
from sparta_tools import sparta # type: ignore
##################################################################################################################
@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    time_s = t1 - t0
    time_min = time_s / 60
    
    print("%s time: %.5fs, %.2f min" % (txt, time_s, time_min))

def clean_dir(path):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files at location:",path)

def check_pickle_exist_gadget(sparta_name, ptl_property, snapshot, snapshot_path, scale_factor):
    # save to folder containing pickled data to be accessed easily later
    file_path = path_to_pickle + str(snapshot) + "_" + str(sparta_name) + "/" + ptl_property + "_" + str(snapshot) + ".pickle" 
    create_directory(path_to_pickle + str(snapshot) +  "_" + str(sparta_name) + "/")
    
    # check if the file has already been pickled if so just load it
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            ptl_info = pickle.load(pickle_file)
    # otherwise load the specific information from the particle data and save it as a pickle file
    else:
        ptl_info = readsnap(snapshot_path, ptl_property, 'dm')
        if ptl_property == "pos":
            ptl_info = ptl_info * 10**3 * scale_factor # convert position to kpc/h and physical
        with open(file_path, "wb") as pickle_file:
            pickle.dump(ptl_info, pickle_file)
    return ptl_info

def load_or_pickle_ptl_data(sparta_name, snapshot, snapshot_path, scale_factor):
    ptl_pid = check_pickle_exist_gadget(sparta_name, "pid", snapshot, snapshot_path, scale_factor=scale_factor)
    ptl_vel = check_pickle_exist_gadget(sparta_name, "vel", snapshot, snapshot_path, scale_factor=scale_factor)
    ptl_pos = check_pickle_exist_gadget(sparta_name, "pos", snapshot, snapshot_path, scale_factor=scale_factor)

    return ptl_pid, ptl_vel, ptl_pos

def load_or_pickle_SPARTA_data(sparta_name, scale_factor, snap, sparta_snap):
    create_directory(path_to_pickle + str(snap) +  "_" + str(sparta_name) + "/")
    reload_sparta = False
    

    if os.path.isfile(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_pos.pickle"):
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_pos.pickle", "rb") as pickle_file:
            halos_pos = pickle.load(pickle_file)
    else:
        reload_sparta = True
    
    if os.path.isfile(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_last_snap.pickle"):
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_last_snap.pickle", "rb") as pickle_file:
            halos_last_snap = pickle.load(pickle_file)
    else:
        reload_sparta = True

    if os.path.isfile(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_r200m.pickle"):
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_r200m.pickle", "rb") as pickle_file:
            halos_r200m = pickle.load(pickle_file)
    else:
        reload_sparta = True

    if os.path.isfile(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_id.pickle"):
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_id.pickle", "rb") as pickle_file:
            halos_id = pickle.load(pickle_file)
    else:
        reload_sparta = True

    if os.path.isfile(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_status.pickle"):
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_status.pickle", "rb") as pickle_file:
            halos_status = pickle.load(pickle_file)
    else:
        reload_sparta = True

    if os.path.isfile(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/parent_id.pickle"):
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/parent_id.pickle", "rb") as pickle_file:
            parent_id = pickle.load(pickle_file)
    else:
        reload_sparta = True
        
    if os.path.isfile(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/ptl_mass.pickle"):
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/ptl_mass.pickle", "rb") as pickle_file:
            ptl_mass = pickle.load(pickle_file)
    else:
        reload_sparta = True
    
    if reload_sparta:
        sparta_output = sparta.load(filename=path_to_hdf5_file, log_level= 0)
        halos_pos = sparta_output['halos']['position'][:,sparta_snap,:] * 10**3 * scale_factor # convert to kpc/h and physical
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_pos.pickle", "wb") as pickle_file:
            pickle.dump(halos_pos, pickle_file)
        halos_last_snap = sparta_output['halos']['last_snap'][:]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_last_snap.pickle", "wb") as pickle_file:
            pickle.dump(halos_last_snap, pickle_file)
        halos_r200m = sparta_output['halos']['R200m'][:,sparta_snap]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_r200m.pickle", "wb") as pickle_file:
            pickle.dump(halos_r200m, pickle_file) 
        halos_id = sparta_output['halos']['id'][:,sparta_snap]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_id.pickle", "wb") as pickle_file:
            pickle.dump(halos_id, pickle_file)
        halos_status = sparta_output['halos']['status'][:,sparta_snap]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_status.pickle", "wb") as pickle_file:
            pickle.dump(halos_status, pickle_file)
        parent_id = sparta_output['halos']['parent_id'][:,sparta_snap]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/parent_id.pickle", "wb") as pickle_file:
            pickle.dump(parent_id, pickle_file)
        ptl_mass = sparta_output["simulation"]["particle_mass"]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/ptl_mass.pickle", "wb") as pickle_file:
            pickle.dump(ptl_mass, pickle_file)

    return halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, parent_id, ptl_mass 

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
        
def choose_halo_split(indices, snap, halo_props, particle_props, num_features):
    start_idxs = halo_props["Halo_start_ind_" + snap].to_numpy()
    num_ptls = halo_props["Halo_num_ptl_" + snap].to_numpy()

    dataset = np.zeros((np.sum(num_ptls[indices]), num_features))
    start = 0
    for idx in indices:
        start_ind = start_idxs[idx]
        curr_num_ptl = num_ptls[idx]
        dataset[start:start+curr_num_ptl] = particle_props[start_ind:start_ind+curr_num_ptl]

        start = start + curr_num_ptl

    return dataset

def find_closest_z(value,snap_path,snap_form):
    all_z = np.ones(total_num_snaps) * -1000
    for i in range(total_num_snaps):
        # Sometimes not all snaps exist
        if os.path.isdir(snap_path + "snapdir_" + snap_format.format(i)):
            all_z[i] = readheader(snap_path + "snapdir_" + snap_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift')

    idx = (np.abs(all_z - value)).argmin()
    return idx, all_z[idx]

def find_closest_snap(value, cosmology, all_red_shifts):
    all_times = np.ones(total_num_snaps) * -1000
    for i in range(total_num_snaps):
        # Sometimes not all snaps exist
        if os.path.isdir(path_to_snaps + "snapdir_" + snap_format.format(i)):
            all_times[i] = cosmology.age(readheader(path_to_snaps + "snapdir_" + snap_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift'))
    idx = (np.abs(all_times - value)).argmin()
    return idx

def conv_halo_id_spid(my_halo_ids, sdata, snapshot):
    sparta_idx = np.zeros(my_halo_ids.shape[0], dtype = np.int32)
    for i, my_id in enumerate(my_halo_ids):
        sparta_idx[i] = int(np.where(my_id == sdata['halos']['id'][:,snapshot])[0])
    return sparta_idx

def get_comp_snap(t_dyn, t_dyn_step, snapshot_list, cosmol, p_red_shift, all_red_shifts, snap_format):
    # calculate one dynamical time ago and set that as the comparison snap
    curr_time = cosmol.age(p_red_shift)
    past_time = curr_time - (t_dyn_step * t_dyn)
    c_snap = find_closest_snap(past_time, cosmol, all_red_shifts)
    snapshot_list.append(c_snap)

    # switch to comparison snap
    snapshot_path = path_to_snaps + "/snapdir_" + snap_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap)
        
    # get constants from pygadgetreader
    c_red_shift = readheader(snapshot_path, 'redshift')
    c_sparta_snap = np.abs(all_red_shifts - c_red_shift).argmin()
    print("Complementary snapshot:", c_snap, "Complementary redshift:", c_red_shift)
    print("Corresponding SPARTA loc:", c_sparta_snap, "SPARTA redshift:",all_red_shifts[c_sparta_snap])

    c_scale_factor = 1/(1+c_red_shift)
    c_rho_m = cosmol.rho_m(c_red_shift)
    c_hubble_constant = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
    # c_box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
    # c_box_size = c_box_size * 10**3 * c_scale_factor #convert to Kpc/h physical
    # c_box_size = c_box_size + 0.001 # NEED TO MAKE WORK FOR PARTICLES ON THE VERY EDGE
    
    if reset_lvl == 3:
        clean_dir(path_to_pickle + str(c_snap) + "_" + str(sparta_name) +  "_" + str(int(search_rad)) + "r200m/")
    # load particle data and SPARTA data for the comparison snap
    with timed("c_snap ptl load"):
        c_particles_pid, c_particles_vel, c_particles_pos = load_or_pickle_ptl_data(curr_sparta_file, str(c_snap), snapshot_path, c_scale_factor)
    with timed("c_snap SPARTA load"):
        c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap, c_parent_id, mass = load_or_pickle_SPARTA_data(curr_sparta_file, c_scale_factor, c_snap, c_sparta_snap)

    return c_snap, c_sparta_snap, c_rho_m, c_red_shift, c_scale_factor, c_hubble_constant, c_particles_pid, c_particles_vel, c_particles_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap
