import numpy as np
import time 
import pickle
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import peaks
import multiprocessing as mp
import h5py
from itertools import repeat
import os
from data_and_loading_functions import build_ml_dataset, check_pickle_exist_gadget, choose_halo_split, create_directory, split_dataset_by_mass
from visualization_functions import *
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
rand_seed = config.getint("MISC","random_seed")
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
create_directory(path_to_xgboost)
snap_format = config["MISC"]["snap_format"]
global prim_only
prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
global p_snap
p_snap = config.getint("SEARCH","p_snap")
c_snap = config.getint("XGBOOST","c_snap")
snapshot_list = [p_snap, c_snap]
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")
num_nu_split = config.getint("XGBOOST","num_nu_split")

global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")

num_processes = mp.cpu_count()
global n_halo_per
n_halo_per = 1000
##################################################################################################################
# import pygadgetreader and sparta
import sys
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta import sparta
##################################################################################################################

# set what the paths should be for saving and getting the data
if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"
    
data_location = path_to_calc_info + specific_save
save_location = path_to_xgboost + specific_save

create_directory(save_location)

snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(snapshot_list[0]) + "/snapshot_" + snap_format.format(snapshot_list[0])

#TODO change how mass is gotten from SPARTA instead.
ptl_mass = check_pickle_exist_gadget(curr_sparta_file, "mass", str(snapshot_list[0]), snapshot_path)
mass = ptl_mass[0] * 10**10 #units M_sun/h
cosmol = cosmology.setCosmology("bolshoi")

np.random.seed(rand_seed)

t1 = time.time()
print("Start train dataset creation")
# Create the separate datasets for training and testing
with open(data_location + "train_indices.pickle", "rb") as pickle_file:
    train_indices = pickle.load(pickle_file)
with open(data_location + "test_indices.pickle", "rb") as pickle_file:
    test_indices = pickle.load(pickle_file)
p_snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)
p_red_shift = readheader(p_snapshot_path, 'redshift')

with open(path_to_pickle + str(p_snap) + "_" + curr_sparta_file + "/halos_r200m.pickle", "rb") as file:
    halos_r200m = pickle.load(file)
    train_halos_r200m = halos_r200m[train_indices]
    test_halos_r200m = halos_r200m[test_indices]
    train_halos_mass = mass_so.R_to_M(train_halos_r200m, p_red_shift, "200m")
    test_halos_mass = mass_so.R_to_M(test_halos_r200m, p_red_shift, "200m")
    train_peaks = peaks.peakHeight(train_halos_mass, p_red_shift)
    test_peaks = peaks.peakHeight(test_halos_mass, p_red_shift)

if np.max(train_peaks) > np.max(test_peaks):
    max_nu = np.max(train_peaks)
else:
    max_nu = np.max(test_peaks)
    
nus = np.linspace((0.001), (max_nu), num_nu_split)

def split_by_nu(nus, peaks, curr_dataset, test):
    path_to_curr_dataset = path_to_calc_info  + curr_sparta_file + "_" + str(p_snap) + "to" + str(c_snap) + "_" + str(search_rad) + "r200msearch/" + curr_dataset + "_all_particle_properties_" + curr_sparta_file + ".hdf5"

    if os.path.exists(path_to_xgboost + curr_sparta_file + "_" + str(p_snap) + "to" + str(c_snap) + "_" + str(search_rad) + "r200msearch/" + curr_dataset + "_datasets/" + curr_dataset + "_keys.pickle") != True:
        with h5py.File((path_to_curr_dataset), 'r') as all_ptl_properties:
            all_keys = np.empty(0,dtype=object)
            for key in all_ptl_properties.keys():
                if key != "Halo_first" and key != "Halo_n":
                    if all_ptl_properties[key].ndim > 1:
                        for row in range(all_ptl_properties[key].ndim):
                            all_keys = np.append(all_keys, (key + str(snapshot_list[row])))
                    else:
                        all_keys = np.append(all_keys, (key + str(snapshot_list[0])))
        with open(path_to_xgboost + curr_sparta_file + "_" + str(p_snap) + "to" + str(c_snap) + "_" + str(search_rad) + "r200msearch/" + curr_dataset + "_keys.pickle", 'wb') as pickle_file:
            pickle.dump(all_keys, pickle_file)

    for i in range(nus.size - 1):
        use_halos = np.where((peaks > nus[i]) & (peaks < nus[i+1]))[0]
        num_files = int(np.ceil(use_halos.shape[0] / n_halo_per))
        
        with h5py.File((path_to_curr_dataset), 'r') as all_ptl_properties: 
            halo_first = all_ptl_properties["Halo_first"][use_halos]
            halo_n = all_ptl_properties["Halo_n"][use_halos]

        with mp.Pool(processes=num_processes) as p:
            p.starmap(split_dataset_by_mass, zip([halo_first[i*n_halo_per:(i+1)*n_halo_per] for i in range(num_files)], [halo_n[i*n_halo_per:(i+1)*n_halo_per] for i in range(num_files)], 
                                                 repeat(path_to_curr_dataset), repeat(curr_dataset), repeat(np.round(nus[i],2)), repeat(np.round(nus[i+1],2)), np.arange(use_halos.shape[0]), repeat(test)))
        p.close()
        p.join()

split_by_nu(nus, train_peaks, "train", False)
t2 = time.time()
print("Time taken:", np.round((t2-t1),2),"seconds")

print("Start test dataset creation")
split_by_nu(nus, test_peaks, "test", True)
t3 = time.time()
print("Time taken:", np.round((t3-t2),2),"seconds")
