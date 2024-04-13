import numpy as np
import time 
from colossus.cosmology import cosmology
import pickle
import os
import multiprocessing as mp
import h5py
from itertools import repeat
from utils.data_and_loading_functions import check_pickle_exist_gadget, choose_halo_split, create_directory
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("/home/zvladimi/MLOIS/config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
path_to_plotting = config["PATHS"]["path_to_plotting"]
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
create_directory(path_to_xgboost)
snap_format = config["MISC"]["snap_format"]
global prim_only
prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
global p_snap
p_snap = config.getint("XGBOOST","p_snap")
c_snap = config.getint("XGBOOST","c_snap")
snapshot_list = [p_snap, c_snap]
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
training_rad = config.getfloat("XGBOOST","training_rad")
curr_chunk_size = 500
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
num_processes = mp.cpu_count()
##################################################################################################################
# import pygadgetreader and sparta
import sys
sys.path.insert(1, path_to_pygadgetreader)
sys.path.insert(1, path_to_sparta)
sys.path.insert(1, path_to_plotting)
from utils.visualization_functions import *
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta
##################################################################################################################

def take_ptl_within(dataset, labels, scal_rad_loc, max_rad):
    within_rad = np.where(dataset[:,scal_rad_loc] <= max_rad)[0]
    within_rad_dataset = dataset[within_rad]
    within_rad_labels = labels[within_rad]
    return within_rad.shape[0], within_rad_dataset, within_rad_labels

def build_ml_dataset(save_path, data_location, sparta_name, dataset_name, snapshot_list, p_snap, max_rad):
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
                elif key == "Halo_first":
                    all_rad_halo_first = all_ptl_properties[key][:]
                elif key == "Halo_n":
                    all_rad_halo_n = all_ptl_properties[key][:]
                elif key != "Halo_n" and key != "Halo_n" and key != "HIPIDS" and key != "Orbit_Infall":
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

        # find where the primary radius is and then where they are less than the max radius being used
        scal_rad_loc = np.where(all_keys==("Scaled_radii_" + str(p_snap)))
        if (max_rad+0.01) < np.max(full_dataset[:,scal_rad_loc]):
            print(max_rad, np.max(full_dataset[:,scal_rad_loc]))
            curr_num_halos = all_rad_halo_first.shape[0]

            with mp.Pool(processes=num_processes) as p:
                within_rad_halo_n, within_rad_dataset, within_rad_labels = zip(*p.starmap(take_ptl_within,zip((full_dataset[all_rad_halo_first[i]:all_rad_halo_first[i]+all_rad_halo_n[i]] for i in range(curr_num_halos)), 
                                                                                                              (labels[all_rad_halo_first[i]:all_rad_halo_first[i]+all_rad_halo_n[i]] for i in range(curr_num_halos)),
                                                                                                              repeat(scal_rad_loc), repeat(max_rad)),chunksize=100))
            p.join()
            p.close()

            within_rad_dataset = np.concatenate(within_rad_dataset)
            within_rad_labels = np.concatenate(within_rad_labels)
            
            within_rad_halo_n = np.stack(within_rad_halo_n)
            within_rad_halo_first = np.cumsum(within_rad_halo_n)
            within_rad_halo_first = np.insert(within_rad_halo_first,0,0)
            within_rad_halo_first = np.delete(within_rad_halo_first,-1)

            with open(save_path + dataset_name + "_within_rad_halo_n.pickle", "wb") as pickle_file:
                pickle.dump(within_rad_halo_n, pickle_file)
            with open(save_path + dataset_name + "_within_rad_halo_first.pickle", "wb") as pickle_file:
                pickle.dump(within_rad_halo_first, pickle_file)
            with open(save_path + dataset_name + "_within_rad_dataset.pickle", "wb") as pickle_file:
                pickle.dump(within_rad_dataset, pickle_file)
            with open(save_path + dataset_name + "_within_rad_labels.pickle", "wb") as pickle_file:
                pickle.dump(within_rad_labels, pickle_file)
        
        # once all the halos are gone through save them as pickles for later  
        with open(dataset_path, "wb") as pickle_file:
            pickle.dump(full_dataset, pickle_file)
            
        with open(save_path + dataset_name + "_dataset_all_keys.pickle", "wb") as pickle_file:
            pickle.dump(all_keys, pickle_file)
            
        with open(save_path + dataset_name + "_labels.pickle", "wb") as pickle_file:
            pickle.dump(labels, pickle_file)
        
        with open(save_path + dataset_name + "_hipids.pickle", "wb") as pickle_file:
            pickle.dump(hipids, pickle_file)
            
        with open(save_path + dataset_name + "_all_rad_halo_n.pickle", "wb") as pickle_file:
            pickle.dump(all_rad_halo_n, pickle_file)
        
        with open(save_path + dataset_name + "_all_rad_halo_first.pickle", "wb") as pickle_file:
            pickle.dump(all_rad_halo_first, pickle_file)
            
    # if there are already pickle files just open them
    else:
        with open(dataset_path, "rb") as pickle_file:
            full_dataset = pickle.load(pickle_file)
        with open(save_path + dataset_name + "_dataset_all_keys.pickle", "rb") as pickle_file:
            all_keys = pickle.load(pickle_file)
    return full_dataset, all_keys

# set what the paths should be for saving and getting the data
if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"
    
data_location = path_to_calc_info + specific_save
save_location = path_to_xgboost + specific_save

create_directory(save_location)


#TODO change to using sparta
snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(snapshot_list[0]) + "/snapshot_" + snap_format.format(snapshot_list[0])

ptl_mass = check_pickle_exist_gadget(curr_sparta_file, "mass", str(snapshot_list[0]), snapshot_path)
mass = ptl_mass[0] * 10**10 #units M_sun/h
cosmol = cosmology.setCosmology("bolshoi")

np.random.seed(11)

t1 = time.time()
print("Start train dataset creation")
# Create the separate datasets for training and testing
train_dataset, train_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "train", snapshot_list = snapshot_list, p_snap=p_snap, max_rad=training_rad)
t2 = time.time()
print("Time taken:", np.round((t2-t1),2),"seconds")

print("Start test dataset creation")
test_dataset, test_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "test", snapshot_list = snapshot_list, p_snap=p_snap, max_rad=search_rad)
t3 = time.time()
print("Time taken:", np.round((t3-t1),2),"seconds")