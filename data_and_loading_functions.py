import pickle
import h5py 
import os
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

def create_directory(path):
    if os.path.exists(path) != True:
        os.makedirs(path)
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("/home/zvladimi/MLOIS/config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
path_to_hdf5_file = path_to_SPARTA_data + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
path_to_xgboost = config["PATHS"]["path_to_xgboost"]
create_directory(path_to_MLOIS)
create_directory(path_to_snaps)
create_directory(path_to_SPARTA_data)
create_directory(path_to_hdf5_file)
create_directory(path_to_pickle)
create_directory(path_to_calc_info)
snap_format = config["MISC"]["snap_format"]
global prim_only
prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
global p_snap
p_snap = config.getint("SEARCH","p_snap")
global c_snap
c_snap = config.getint("XGBOOST", 'c_snap')
global snapshot_list
snapshot_list = [p_snap, c_snap]
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
num_processes = mp.cpu_count()
##################################################################################################################
import sys
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta import sparta
##################################################################################################################

def check_pickle_exist_gadget(sparta_name, ptl_property, snapshot, snapshot_path):
    # save to folder containing pickled data to be accessed easily later
    file_path = path_to_pickle + str(snapshot) + "_" + str(sparta_name) + "/" + ptl_property + "_" + str(snapshot) + ".pickle" 
    create_directory(path_to_pickle + str(snapshot) +  "_" + str(sparta_name) + "/")
    
    # check if the file has already been pickled if so just load it
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            particle_info = pickle.load(pickle_file)
    # otherwise load the specific information from the particle data and save it as a pickle file
    else:
        particle_info = readsnap(snapshot_path, ptl_property, 'dm')
        with open(file_path, "wb") as pickle_file:
            pickle.dump(particle_info, pickle_file)
    return particle_info

def load_or_pickle_ptl_data(sparta_name, snapshot, snapshot_path, scale_factor):
    ptl_pid = check_pickle_exist_gadget(sparta_name, "pid", snapshot, snapshot_path)
    ptl_vel = check_pickle_exist_gadget(sparta_name, "vel", snapshot, snapshot_path)
    ptl_pos = check_pickle_exist_gadget(sparta_name, "pos", snapshot, snapshot_path)
    
    ptl_pos = ptl_pos * 10**3 * scale_factor # convert to kpc/h and physical

    return ptl_pid, ptl_vel, ptl_pos

def load_or_pickle_SPARTA_data(sparta_name, scale_factor, snap):
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
        
    if os.path.isfile(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/ptl_mass.pickle"):
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/ptl_mass.pickle", "rb") as pickle_file:
            ptl_mass = pickle.load(pickle_file)
    else:
        reload_sparta = True
    
    if reload_sparta:
        sparta_output = sparta.load(filename=path_to_hdf5_file, log_level= 0)
        halos_pos = sparta_output['halos']['position'][:,snap,:] * 10**3 * scale_factor # convert to kpc/h and physical
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_pos.pickle", "wb") as pickle_file:
            pickle.dump(halos_pos, pickle_file)
        halos_last_snap = sparta_output['halos']['last_snap'][:]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_last_snap.pickle", "wb") as pickle_file:
            pickle.dump(halos_last_snap, pickle_file)
        halos_r200m = sparta_output['halos']['R200m'][:,snap]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_r200m.pickle", "wb") as pickle_file:
            pickle.dump(halos_r200m, pickle_file) 
        halos_id = sparta_output['halos']['id'][:,snap]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_id.pickle", "wb") as pickle_file:
            pickle.dump(halos_id, pickle_file)
        halos_status = sparta_output['halos']['status'][:,snap]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/halos_status.pickle", "wb") as pickle_file:
            pickle.dump(halos_status, pickle_file)
        ptl_mass = sparta_output["simulation"]["particle_mass"]
        with open(path_to_pickle + str(snap) + "_" + str(sparta_name) + "/ptl_mass.pickle", "wb") as pickle_file:
            pickle.dump(ptl_mass, pickle_file)

    return halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, ptl_mass 

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
                               
def build_ml_dataset(save_path, data_location, sparta_name, dataset_name, snapshot_list):
    save_path = save_path + "datasets/"
    create_directory(save_path)
    dataset_path = save_path + dataset_name + "_dataset" + ".pickle"
    # if the directory for this hdf5 file exists if not make it
    if os.path.exists(save_path) != True:
        os.makedirs(save_path)
    if os.path.exists(dataset_path) != True:
        num_cols = 0
        with h5py.File((data_location + dataset_name + "_all_particle_properties_" + sparta_name + ".hdf5"), 'r') as all_ptl_properties: 
            for key in all_ptl_properties.keys():
                if key != "Halo_first" and key != "Halo_n" and key != "HIPIDS" and key != "Orbit_Infall":
                    if all_ptl_properties[key].ndim > 1:
                        num_cols += all_ptl_properties[key].shape[1]
                    else:
                        num_cols += 1
            num_params_per_snap = num_cols / len(snapshot_list)    

            num_rows = all_ptl_properties[key].shape[0]
            full_dataset = np.zeros((num_rows, num_cols),dtype=np.float32)
            hipids = np.zeros(num_rows, dtype=np.float64)
            labels = np.zeros(num_rows, dtype=np.int8)
            all_keys = np.empty(num_cols,dtype=object)
            curr_col = 0
            for key in all_ptl_properties.keys():
                if key == "HIPIDS":
                    hipids = all_ptl_properties[key][:]
                elif key == "Orbit_Infall":
                    labels = all_ptl_properties[key][:]
                elif key != "Halo_first" and key != "Halo_n" and key != "HIPIDS" and key != "Orbit_Infall":
                    if all_ptl_properties[key].ndim > 1:
                        for row in range(all_ptl_properties[key].ndim):
                            access_col = int((curr_col + (row * num_params_per_snap)))
                            full_dataset[:,access_col] = all_ptl_properties[key][:,row]
                            all_keys[access_col] = (key + str(snapshot_list[row]))
                        curr_col += 1
                    else:
                        full_dataset[:,curr_col] = all_ptl_properties[key]
                        all_keys[curr_col] = (key + str(snapshot_list[0]))
                        curr_col += 1

                
        # once all the halos are gone through save them as pickles for later  
        with open(dataset_path, "wb") as pickle_file:
            pickle.dump(full_dataset, pickle_file)
            
        with open(save_path + dataset_name + "_dataset_all_keys.pickle", "wb") as pickle_file:
            pickle.dump(all_keys, pickle_file)
            
        with open(save_path + dataset_name + "_labels.pickle", "wb") as pickle_file:
            pickle.dump(labels, pickle_file)
        
        with open(save_path + dataset_name + "_hipids.pickle", "wb") as pickle_file:
            pickle.dump(hipids, pickle_file)
            
    # if there are already pickle files just open them
    else:
        with open(dataset_path, "rb") as pickle_file:
            full_dataset = pickle.load(pickle_file)
        with open(save_path + dataset_name + "_dataset_all_keys.pickle", "rb") as pickle_file:
            all_keys = pickle.load(pickle_file)
    return full_dataset, all_keys

def save_to_hdf5(hdf5_file, data_name, dataset, chunk, max_shape, curr_idx, max_num_keys):
    if len(list(hdf5_file.keys())) < (max_num_keys):
        hdf5_file.create_dataset(data_name, data = dataset, chunks = chunk, maxshape = max_shape, dtype=dataset.dtype)
    # with a new file adding on additional data to the datasets
    elif len(list(hdf5_file.keys())) == (max_num_keys):
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

def find_closest_snap(cosmology, time_find, num_snaps, path_to_snap, snap_format):
    closest_time = 0
    closest_snap = 0
    for i in range(num_snaps):
        red_shift = readheader(path_to_snap + "snapdir_" + snap_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift')
        comp_time = cosmology.age(red_shift)
        
        if np.abs(time_find - comp_time) < np.abs(time_find - closest_time):
            closest_time = comp_time
            closest_snap = i
    return closest_snap

def conv_halo_id_spid(my_halo_ids, sdata, snapshot):
    sparta_idx = np.zeros(my_halo_ids.shape[0], dtype = np.int32)
    for i, my_id in enumerate(my_halo_ids):
        sparta_idx[i] = int(np.where(my_id == sdata['halos']['id'][:,snapshot])[0])
    return sparta_idx

def get_comp_snap(t_dyn, t_dyn_step, snapshot_list, cosmol, p_red_shift, total_num_snaps, snap_format):
    # calculate one dynamical time ago and set that as the comparison snap
    curr_time = cosmol.age(p_red_shift)
    past_time = curr_time - (t_dyn_step * t_dyn)
    c_snap = find_closest_snap(cosmol, past_time, num_snaps = total_num_snaps, path_to_snap=path_to_snaps, snap_format = snap_format)
    snapshot_list.append(c_snap)

    # switch to comparison snap
    snapshot_path = path_to_snaps + "/snapdir_" + snap_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap)
        
    # get constants from pygadgetreader
    c_red_shift = readheader(snapshot_path, 'redshift')
    c_scale_factor = 1/(1+c_red_shift)
    c_rho_m = cosmol.rho_m(c_red_shift)
    c_hubble_constant = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
    c_box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
    c_box_size = c_box_size * 10**3 * c_scale_factor #convert to Kpc/h physical
    
    # load particle data and SPARTA data for the comparison snap
    c_particles_pid, c_particles_vel, c_particles_pos = load_or_pickle_ptl_data(curr_sparta_file, str(c_snap), snapshot_path, c_scale_factor)
    c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap, mass = load_or_pickle_SPARTA_data(curr_sparta_file, c_scale_factor, c_snap)

    return c_snap, c_box_size, c_rho_m, c_red_shift, c_hubble_constant, c_particles_pid, c_particles_vel, c_particles_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap
