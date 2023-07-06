import pickle
import h5py 
import os
import numpy as np
from pygadgetreader import readsnap, readheader

def check_pickle_exist_gadget(path, ptl_property, snapshot, snapshot_path):
    file_path = path + ptl_property + "_" + snapshot + ".pickle" 
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            particle_info = pickle.load(pickle_file)
    else:
        particle_info = readsnap(snapshot_path, ptl_property, 'dm')
        with open(file_path, "wb") as pickle_file:
            pickle.dump(particle_info, pickle_file)
    return particle_info

def check_pickle_exist_hdf5_prop(path, first_group, second_group, third_group, hdf5_path):
    file_path = path + first_group + "_" + second_group + "_" + third_group + ".pickle" 
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            halo_info = pickle.load(pickle_file)
    else:
        print(hdf5_path)
        with h5py.File(hdf5_path, 'r') as file:
            if third_group != "":
                halo_info = file[first_group][second_group][third_group][:]
            else:
                halo_info = file[first_group][second_group][:]
        with open(file_path, "wb") as pickle_file:
            pickle.dump(halo_info, pickle_file)
    return halo_info

def load_or_pickle_data(path, snapshot, hdf5, snapshot_path):
    if os.path.exists(path) != True:
        os.makedirs(path)
    ptl_pid = check_pickle_exist_gadget(path, "pid", snapshot, snapshot_path)
    ptl_vel = check_pickle_exist_gadget(path, "vel", snapshot, snapshot_path)
    ptl_pos = check_pickle_exist_gadget(path, "pos", snapshot, snapshot_path)
    ptl_mass = check_pickle_exist_gadget(path, "mass", snapshot, snapshot_path)
    
    halo_pos = check_pickle_exist_hdf5_prop(path, "halos", "position", "", hdf5)
    halo_vel = check_pickle_exist_hdf5_prop(path, "halos", "velocity", "", hdf5)
    halo_last_snap = check_pickle_exist_hdf5_prop(path, "halos", "last_snap", "", hdf5)
    halo_r200m = check_pickle_exist_hdf5_prop(path, "halos", "R200m", "", hdf5)
    halo_id = check_pickle_exist_hdf5_prop(path, "halos", "id", "", hdf5)
    halo_status = check_pickle_exist_hdf5_prop(path, "halos", "status", "", hdf5)
    
    density_prf_all = check_pickle_exist_hdf5_prop(path, "anl_prf", "M_all", "", hdf5)
    density_prf_1halo = check_pickle_exist_hdf5_prop(path, "anl_prf", "M_1halo", "", hdf5)
    
    halo_n = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "halo_n", hdf5)
    halo_first = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "halo_first", hdf5)
    num_pericenter = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "n_pericenter", hdf5)
    tracer_id = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "tracer_id", hdf5)
    n_is_lower_limit = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "n_is_lower_limit", hdf5)
    last_pericenter_snap = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "last_pericenter_snap", hdf5)

    return ptl_pid, ptl_vel, ptl_pos, ptl_mass, halo_pos, halo_vel, halo_last_snap, halo_r200m, halo_id, halo_status, num_pericenter, tracer_id, n_is_lower_limit, last_pericenter_snap, density_prf_all, density_prf_1halo, halo_n, halo_first

def standardize(values):
    return (values - values.mean())/values.std()

def normalize(values):
    return (values - values.min())/(values.max() - values.min())

def load_training_data(path, parameter, hdf5, random_indices):
    
    file_path = path + parameter + ".pickle" 
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            particle_info = pickle.load(pickle_file)
    else:
        particle_info = hdf5[parameter][:]
        with open(file_path, "wb") as pickle_file:
            pickle.dump(particle_info, pickle_file)
    
    particle_info = particle_info[random_indices]
    return particle_info

def load_or_pickle_data_ml(path, data_location, curr_split, indices, use_num_particles, snapshot, param_list):
    dataset = np.zeros(use_num_particles)
    if os.path.exists(path) != True:
        os.makedirs(path)
        
    with h5py.File((data_location + "all_particle_properties" + snapshot + ".hdf5"), 'r') as all_particle_properties:
        start_index = curr_split * use_num_particles
        use_random_indices = indices[start_index:start_index+use_num_particles]
        
        for i,parameter in enumerate(param_list):
            curr_dataset = load_training_data(path, parameter, all_particle_properties, use_random_indices)
            if parameter == 'Orbit_Infall' and i != 0:
                dataset[:,0] = curr_dataset
            elif parameter == 'Orbit_Infall' and i == 0:
                dataset = curr_dataset
            else:
                dataset = np.column_stack((dataset, curr_dataset))

    return dataset

def build_ml_dataset(output_path, data_loc_path, curr_split, indices, use_num_particles, snapshot, param_list):
    dataset_path = output_path + "dataset.pickle"
    if os.path.exists(dataset_path) == True:
        with open(dataset_path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    else:
        with open(dataset_path, "wb") as pickle_file:
            dataset = load_or_pickle_data(output_path, data_loc_path, curr_split, indices, use_num_particles, snapshot, param_list)
            pickle.dump(dataset, pickle_file)
            return dataset
