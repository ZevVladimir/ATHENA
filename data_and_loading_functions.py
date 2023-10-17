import pickle
import h5py 
import os
import numpy as np
import sys
sys.path.insert(0, "/home/zvladimi/MLOIS/pygadgetreader")
sys.path.insert(0,  "/home/zvladimi/MLOIS/sparta/analysis")
from pygadgetreader import readsnap, readheader
from sparta import sparta

def create_directory(path):
    if os.path.exists(path) != True:
        os.makedirs(path)

def check_pickle_exist_gadget(sparta_name, ptl_property, snapshot, snapshot_path, path_dict):
    # save to folder containing pickled data to be accessed easily later
    file_path = path_dict["path_to_pickle"] + str(snapshot) + "_" + str(sparta_name) + "/" + ptl_property + "_" + str(snapshot) + ".pickle" 
    create_directory(path_dict["path_to_pickle"] + str(snapshot) +  "_" + str(sparta_name) + "/")
    
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

def load_or_pickle_ptl_data(sparta_name, snapshot, snapshot_path, scale_factor, little_h, path_dict):
    ptl_pid = check_pickle_exist_gadget(sparta_name, "pid", snapshot, snapshot_path, path_dict)
    ptl_vel = check_pickle_exist_gadget(sparta_name, "vel", snapshot, snapshot_path, path_dict)
    ptl_pos = check_pickle_exist_gadget(sparta_name, "pos", snapshot, snapshot_path, path_dict)
    
    ptl_pos = ptl_pos * 10**3 * scale_factor # convert to kpc/h and physical

    return ptl_pid, ptl_vel, ptl_pos

def load_or_pickle_SPARTA_data(sparta_name, scale_factor, snap, path_dict):
    create_directory(path_dict["path_to_pickle"] + str(snap) +  "_" + str(sparta_name) + "/")
    reload_sparta = False

    if os.path.isfile(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_pos.pickle"):
        with open(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_pos.pickle", "rb") as pickle_file:
            halos_pos = pickle.load(pickle_file)
    else:
        reload_sparta = True
    
    if os.path.isfile(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_vel.pickle"):
        with open(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_vel.pickle", "rb") as pickle_file:
            halos_vel = pickle.load(pickle_file)
    else:
        reload_sparta = True
    
    if os.path.isfile(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_last_snap.pickle"):
        with open(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_last_snap.pickle", "rb") as pickle_file:
            halos_last_snap = pickle.load(pickle_file)
    else:
        reload_sparta = True

    if os.path.isfile(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_r200m.pickle"):
        with open(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_r200m.pickle", "rb") as pickle_file:
            halos_r200m = pickle.load(pickle_file)
    else:
        reload_sparta = True

    if os.path.isfile(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_id.pickle"):
        with open(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_id.pickle", "rb") as pickle_file:
            halos_id = pickle.load(pickle_file)
    else:
        reload_sparta = True

    if os.path.isfile(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_status.pickle"):
        with open(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_halos_status.pickle", "rb") as pickle_file:
            halos_status = pickle.load(pickle_file)
    else:
        reload_sparta = True
        
    if os.path.isfile(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_ptl_mass.pickle"):
        with open(path_dict["path_to_pickle"] + str(snap) + "_" + str(sparta_name) + "_ptl_mass.pickle", "rb") as pickle_file:
            halos_status = pickle.load(pickle_file)
    else:
        reload_sparta = True
    
    if reload_sparta:
        sparta_output = sparta.load(filename=path_dict["path_to_hdf5_file"], log_level= 0)
        halos_pos = sparta_output['halos']['position'][:,snap,:] * 10**3 * scale_factor # convert to kpc/h and physical
        halos_last_snap = sparta_output['halos']['last_snap'][:]
        halos_r200m = sparta_output['halos']['R200m'][:,snap] 
        halos_id = sparta_output['halos']['id'][:,snap]
        halos_status = sparta_output['halos']['status'][:,snap]
        ptl_mass = sparta_output["simulation"]["particle_mass"]

    return halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, ptl_mass

def standardize(values):
    for col in range(values.shape[1]):
        values[:,col] = (values[:,col] - values[:,col].mean())/values[:,col].std()
    return values

def normalize(values):
    for col in range(values.shape[1]):
        values[:,col] = (values[:,col] - values[:,col].min())/(values[:,col].max() - values[:,col].min())
    return values

def build_ml_dataset(save_path, data_location, sparta_name, dataset_name, snapshot_list):
    save_path = save_path + "datasets/"
    create_directory(save_path)
    dataset_path = save_path + dataset_name + "_dataset_" + sparta_name 
    
    for snap in snapshot_list:
        dataset_path = dataset_path + "_" + str(snap)
    dataset_path = dataset_path + ".pickle"
    # if the directory for this hdf5 file exists if not make it
    if os.path.exists(save_path) != True:
        os.makedirs(save_path)
    if os.path.exists(dataset_path) != True:
        num_cols = 0
        with h5py.File((data_location + dataset_name + "_all_particle_properties_" + sparta_name + ".hdf5"), 'r') as all_ptl_properties: 
            for key in all_ptl_properties.keys():
                if all_ptl_properties[key].ndim > 1:
                    num_cols += all_ptl_properties[key].shape[1]
                else:
                    num_cols += 1
            num_params_per_snap = (num_cols - 2) / len(snapshot_list)    
            num_rows = all_ptl_properties[key].shape[0]
            full_dataset = np.zeros((num_rows, num_cols))
            all_keys = np.empty(num_cols,dtype=object)
            curr_col = 0
            for key in all_ptl_properties.keys():
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
    # if there are already pickle files just open them
    else:
        with open(dataset_path, "rb") as pickle_file:
            full_dataset = pickle.load(pickle_file)
        with open(save_path + dataset_name + "_dataset_all_keys.pickle", "rb") as pickle_file:
            all_keys = pickle.load(pickle_file)
    return full_dataset, all_keys

def save_to_hdf5(new_file, hdf5_file, data_name, dataset, chunk, max_shape, curr_idx, max_num_keys):
    if new_file and len(list(hdf5_file.keys())) < (max_num_keys):
        hdf5_file.create_dataset(data_name, data = dataset, chunks = chunk, maxshape = max_shape)
    # with a new file adding on additional data to the datasets
    elif new_file and len(list(hdf5_file.keys())) == (max_num_keys):
        hdf5_file[data_name].resize((hdf5_file[data_name].shape[0] + dataset.shape[0]), axis = 0)
        hdf5_file[data_name][-dataset.shape[0]:] = dataset   

    # if not a new file and same num of particles will just replace the previous information
    if not new_file:
        hdf5_file[data_name][curr_idx:curr_idx + dataset.shape[0]] = dataset
        
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

def get_comp_snap(t_dyn, t_dyn_step, snapshot_list, cosmol, p_red_shift, total_num_snaps, path_dict, snap_format, little_h):
    # calculate one dynamical time ago and set that as the comparison snap
    curr_time = cosmol.age(p_red_shift)
    past_time = curr_time - (t_dyn_step * t_dyn)
    c_snap = find_closest_snap(cosmol, past_time, num_snaps = total_num_snaps, path_to_snap=path_dict['path_to_snaps'], snap_format = snap_format)
    snapshot_list.append(c_snap)

    # switch to comparison snap
    snapshot_path = path_dict['path_to_snaps'] + "/snapdir_" + snap_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap)
        
    # get constants from pygadgetreader
    c_red_shift = readheader(snapshot_path, 'redshift')
    c_scale_factor = 1/(1+c_red_shift)
    c_rho_m = cosmol.rho_m(c_red_shift)
    c_hubble_constant = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
    c_box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
    c_box_size = c_box_size * 10**3 * c_scale_factor #convert to Kpc/h physical
    
    # load particle data and SPARTA data for the comparison snap
    c_particles_pid, c_particles_vel, c_particles_pos = load_or_pickle_ptl_data(path_dict["curr_sparta_file"], str(c_snap), snapshot_path, c_scale_factor, little_h, path_dict)
    c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap, mass = load_or_pickle_SPARTA_data(path_dict["curr_sparta_file"], c_scale_factor, c_snap, path_dict)

    return c_snap, c_box_size, c_rho_m, c_red_shift, c_hubble_constant, c_particles_pid, c_particles_vel, c_particles_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap
