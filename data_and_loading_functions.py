import pickle
import h5py 
import os
import numpy as np
from pygadgetreader import readsnap, readheader

def check_pickle_exist_gadget(path, ptl_property, snapshot, snapshot_path):
    file_path = path + ptl_property + "_" + snapshot + ".pickle" 
    print(file_path)
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            particle_info = pickle.load(pickle_file)
    else:
        particle_info = readsnap(snapshot_path, ptl_property, 'dm')
        with open(file_path, "wb") as pickle_file:
            pickle.dump(particle_info, pickle_file)
    return particle_info

def check_pickle_exist_hdf5_prop(path, first_group, second_group, third_group, hdf5_path, sparta_name):
    file_path = path + first_group + "_" + second_group + "_" + third_group + "_" + sparta_name + ".pickle" 
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            halo_info = pickle.load(pickle_file)
    else:
        with h5py.File(hdf5_path, 'r') as file:
            if third_group != "":
                halo_info = file[first_group][second_group][third_group][:]
            else:
                halo_info = file[first_group][second_group][:]
        with open(file_path, "wb") as pickle_file:
            pickle.dump(halo_info, pickle_file)
    return halo_info

def load_or_pickle_ptl_data(path, snapshot, snapshot_path, scale_factor, little_h):
    if os.path.exists(path) != True:
        os.makedirs(path)
    ptl_pid = check_pickle_exist_gadget(path, "pid", snapshot, snapshot_path)
    ptl_vel = check_pickle_exist_gadget(path, "vel", snapshot, snapshot_path)
    ptl_pos = check_pickle_exist_gadget(path, "pos", snapshot, snapshot_path)
    ptl_mass = check_pickle_exist_gadget(path, "mass", snapshot, snapshot_path)

    ptl_pos = ptl_pos * 10**3 * scale_factor * little_h #convert to kpc and physical
    ptl_mass = ptl_mass[0] * 10**10 #units M_sun/h

    return ptl_pid, ptl_vel, ptl_pos, ptl_mass

def load_or_pickle_SPARTA_data(path, sparta_name, hdf5_path, scale_factor, little_h, snap):
    halos_pos = check_pickle_exist_hdf5_prop(path, "halos", "position", "", hdf5_path, sparta_name)
    halos_vel = check_pickle_exist_hdf5_prop(path, "halos", "velocity", "", hdf5_path, sparta_name)
    halo_last_snap = check_pickle_exist_hdf5_prop(path, "halos", "last_snap", "", hdf5_path, sparta_name)
    halos_r200m = check_pickle_exist_hdf5_prop(path, "halos", "R200m", "", hdf5_path, sparta_name)
    halo_id = check_pickle_exist_hdf5_prop(path, "halos", "id", "", hdf5_path, sparta_name)
    halo_status = check_pickle_exist_hdf5_prop(path, "halos", "status", "", hdf5_path, sparta_name)
    
    density_prf_all = check_pickle_exist_hdf5_prop(path, "anl_prf", "M_all", "", hdf5_path, sparta_name)
    density_prf_1halo = check_pickle_exist_hdf5_prop(path, "anl_prf", "M_1halo", "", hdf5_path, sparta_name)
    
    halos_pos = halos_pos[:,snap,:]
    halos_vel = halos_vel[:,snap,:]
    halos_r200m = halos_r200m[:,snap]
    halo_id = halo_id[:,snap]
    halo_status = halo_status[:,snap]
    density_prf_all = density_prf_all[:,snap,:]
    density_prf_1halo = density_prf_1halo[:,snap,:]
    
    halos_pos = halos_pos * 10**3 * scale_factor * little_h #convert to kpc and physical
    halos_r200m = halos_r200m * little_h # convert to kpc

    return halos_pos, halos_vel, halos_r200m, halo_id, density_prf_all, density_prf_1halo, halo_status, halo_last_snap

def standardize(values):
    return (values - values.mean())/values.std()

def normalize(values):
    return (values - values.min())/(values.max() - values.min())

def build_ml_dataset(path, data_location, sparta_name, param_list, dataset_name, total_num_halos, halo_split_param, snap):
    dataset_path = path + dataset_name + "dataset" + sparta_name + ".pickle"

    # if the directory for this hdf5 file exists if not make it
    if os.path.exists(path) != True:
        os.makedirs(path)

    # if there isn't a pickle for the train_dataset make it 
    if os.path.exists(dataset_path) != True or os.path.exists(path + "halo_prop_ML.pickle") != True:
        # load in the halo properties from the hdf5 file and put them all in one big array
        with h5py.File((data_location + "all_halo_properties" + sparta_name + ".hdf5"), 'r') as all_halo_properties:
            # if there isn't a pickle of the halo properties loop through them and add them all to one big array
            if os.path.exists(path + "halo_prop_ML_" + snap + ".pickle") != True: 
                halo_props = np.zeros(total_num_halos)
                for i,parameter in enumerate(halo_split_param):
                    curr_halo_prop = all_halo_properties[parameter]
                    if i == 0:
                        halo_props = curr_halo_prop
                    else:
                        halo_props = np.column_stack((halo_props, curr_halo_prop))
                with open(path + "halo_prop_ML_" + snap + ".pickle", "wb") as pickle_file:
                    pickle.dump(halo_props, pickle_file)
            else:
                with open(path + "halo_prop_ML_" + snap + ".pickle", "rb") as pickle_file:
                    halo_props = pickle.load(pickle_file)

        with h5py.File((data_location + "all_particle_properties" + sparta_name + ".hdf5"), 'r') as all_particle_properties:
            # get how many parameters there are (since velocities are split into)
            num_cols = 0
            for parameter in param_list:
                param_dset = all_particle_properties[parameter][:]
                if param_dset.ndim > 1:
                    num_cols += param_dset.shape[1]
                else:
                    num_cols += 1

            full_dataset = np.zeros((np.sum(halo_props[:,1]), num_cols))
            # go through each halo
            count = 0
            for j in range(halo_props.shape[0]):
                curr_halo_start = halo_props[j,0]
                curr_halo_num_ptl = halo_props[j,1]

                halo_dataset = np.zeros((curr_halo_num_ptl,1))
                # go through each parameter to load in for this halo
                for k,parameter in enumerate(param_list):
                    curr_dataset = all_particle_properties[parameter][curr_halo_start:curr_halo_start+curr_halo_num_ptl]
                    # have it so orbit_infall assignment is always first column otherwise just stack them
                    if k == 0:
                        halo_dataset = curr_dataset
                    elif parameter == ('Orbit_Infall_' + snap):
                        halo_dataset = np.insert(halo_dataset, curr_dataset, 0, axis = 1)
                    else:
                        halo_dataset = np.column_stack((halo_dataset, curr_dataset))

                full_dataset[count:count + curr_halo_num_ptl] = halo_dataset
    
                count = count + curr_halo_num_ptl

        # once all the halos are gone through save them as pickles for later  
        with open(dataset_path, "wb") as pickle_file:
            pickle.dump(full_dataset, pickle_file)

    # if there are already pickle files just open them
    else:
        with open(dataset_path, "rb") as pickle_file:
            full_dataset = pickle.load(pickle_file)
        with open(path + "halo_prop_ML.pickle", "rb") as pickle_file:
            halo_props = pickle.load(pickle_file)

    return full_dataset, halo_props

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
        print(start_ind)
        print(curr_num_ptl)
        dataset[start:start+curr_num_ptl] = particle_props[start_ind:start_ind+curr_num_ptl]

        start = start + curr_num_ptl

    return dataset

def find_closest_snap(cosmology, time_find):
    closest_time = 0
    closest_snap = 0
    for i in range(193):
        red_shift = readheader("/home/zvladimi/MLOIS/particle_data/snapdir_" + "{:04d}".format(i) + "/snapshot_" + "{:04d}".format(i), 'redshift')
        comp_time = cosmology.age(red_shift)
        
        if np.abs(time_find - comp_time) < np.abs(time_find - closest_time):
            closest_time = comp_time
            closest_snap = i
    return closest_snap