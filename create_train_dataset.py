import numpy as np
import time 
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import peaks
import pickle
import os
import multiprocessing as mp
import h5py
import json
import re
from contextlib import contextmanager
from itertools import repeat
from sklearn.model_selection import train_test_split
from utils.data_and_loading_functions import check_pickle_exist_gadget, choose_halo_split, create_directory
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
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

create_directory(path_to_xgboost)

snap_format = config["MISC"]["snap_format"]
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
p_red_shift = config.getfloat("SEARCH","p_red_shift")

use_sims = json.loads(config.get("DATASET","use_sims"))

model_snapshot_list = json.loads(config.get("XGBOOST","model_snaps"))
test_snapshot_list = json.loads(config.get("XGBOOST","test_snaps"))

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
from utils.visualization_functions import *
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta
##################################################################################################################
@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5fs" % (txt, t1 - t0))
    
def take_ptl_within(dataset, labels, scal_rad_loc, max_rad):
    within_rad = np.where(dataset[:,scal_rad_loc] <= max_rad)[0]
    within_rad_dataset = dataset[within_rad]
    within_rad_labels = labels[within_rad]
    return within_rad.shape[0], within_rad_dataset, within_rad_labels

def transform_string(input_str):
    # Define the regex pattern to match the string parts
    pattern = r"_([\d]+)to([\d]+)"
    
    # Search for the pattern in the input string
    match = re.search(pattern, input_str)
    
    if not match:
        raise ValueError("Input string does not match the expected format.")

    # Extract the parts of the string
    prefix = input_str[:match.start()]
    first_number = match.group(1)
    
    # Construct the new string
    new_string = f"{first_number}_{prefix}"
    
    return new_string

def build_sim_dataset(curr_sim, snapshot_list):
    num_cols = 0
    curr_file = path_to_calc_info + curr_sim + "/all_particle_properties.hdf5"
    with h5py.File((curr_file), 'r') as all_ptl_properties: 
        for key in all_ptl_properties.keys():
            if key != "Halo_first" and key != "Halo_n" and key != "HIPIDS" and key != "Orbit_Infall":
                if all_ptl_properties[key].ndim > 1:
                    num_cols += all_ptl_properties[key].shape[1]
                else:
                    num_cols += 1
        num_params_per_snap = num_cols / len(snapshot_list)    

        num_rows = all_ptl_properties[key].shape[0]
        full_dataset = np.zeros((num_rows, num_cols),dtype=np.float32)
        dataset_keys = np.empty(num_cols,dtype=object)
        curr_col = 0
        for key in all_ptl_properties.keys():
            if key == "HIPIDS":
                hipids = all_ptl_properties[key][:]
            elif key == "Orbit_Infall":
                labels = all_ptl_properties[key][:]
            elif key == "Halo_first":
                halo_first = all_ptl_properties[key][:]
            elif key == "Halo_n":
                halo_n = all_ptl_properties[key][:]
            elif key != "Halo_n" and key != "Halo_n" and key != "HIPIDS" and key != "Orbit_Infall":
                if all_ptl_properties[key].ndim > 1:
                    for row in range(all_ptl_properties[key].ndim):
                        access_col = int((curr_col + (row * num_params_per_snap)))
                        full_dataset[:,access_col] = all_ptl_properties[key][:,row]
                        dataset_keys[access_col] = (key + str(snapshot_list[row]))
                    curr_col += 1
                else:
                    full_dataset[:,curr_col] = all_ptl_properties[key]
                    dataset_keys[curr_col] = (key + str(snapshot_list[0]))
                    curr_col += 1

    return full_dataset, dataset_keys, hipids, labels, halo_first, halo_n

def build_mass_dataset():
    #TODO find some way to do the exact redshift for each simulation
    nus = []
    fig, ax = plt.subplots(1, figsize=(30,15))
    for sim in use_sims:
        with h5py.File(path_to_hdf5_file,"r") as f:
            dic_sim = {}
            grp_sim = f['simulation']
            for f in grp_sim.attrs:
                dic_sim[f] = grp_sim.attrs[f]
        
        all_red_shifts = dic_sim['snap_z']
        p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
        use_z = all_red_shifts[p_sparta_snap]
        
        curr_file = path_to_calc_info + sim + "/all_particle_properties.hdf5"
        
        p_snap_loc = transform_string(sim)
        with open(path_to_pickle + p_snap_loc + "/ptl_mass.pickle", "rb") as pickle_file:
            ptl_mass = pickle.load(pickle_file)
        print(sim, ptl_mass)
        with h5py.File(curr_file,"r") as f:
            nus.append(peaks.peakHeight((f["Halo_n"][:]*ptl_mass), use_z))
   
    
    titlefntsize=26
    axisfntsize=24
    tickfntsize=20
    legendfntsize=22
    ax.hist(nus, bins=50, label=use_sims)
    ax.set_xlabel(r"$\nu$",fontsize=axisfntsize)
    ax.set_ylabel("Number of Halos",fontsize=axisfntsize)
    ax.tick_params(axis='both',which='both',labelsize=tickfntsize)
    ax.set_yscale("log")
    ax.legend(fontsize=legendfntsize)
    fig.savefig("/home/zvladimi/MLOIS/Random_figs/nu_dist.png")
            
cosmol = cosmology.setCosmology("bolshoi")

np.random.seed(11)
build_mass_dataset()

# Create all the datasets for the simulations
combined_name = ""

for i,sim in enumerate(use_sims):
    with timed("Datasets for: "+sim+" created"):
        curr_loc = path_to_xgboost + sim + "/datasets/"
        create_directory(curr_loc)
        
        pattern = r"(\d+)to(\d+)"
        match = re.search(pattern, sim)

        if match:
            curr_snap_list = [match.group(1), match.group(2)] 
            print(f"First number: {match.group(1)}")
            print(f"Second number: {match.group(2)}")
        else:
            print("Pattern not found in the string.")

        if not os.path.exists(curr_loc + "dataset.pickle"):
            curr_dataset,curr_dataset_keys,curr_hipids,curr_labels,curr_halo_first,curr_halo_n = build_sim_dataset(sim,curr_snap_list)
            
            with open(curr_loc + "dataset.pickle", "wb") as pickle_file:
                pickle.dump(curr_dataset, pickle_file)
            with open(curr_loc + "keys.pickle", "wb") as pickle_file:
                pickle.dump(curr_dataset_keys, pickle_file)
            with open(curr_loc + "hipids.pickle", "wb") as pickle_file:
                pickle.dump(curr_hipids, pickle_file)
            with open(curr_loc + "labels.pickle", "wb") as pickle_file:
                pickle.dump(curr_labels, pickle_file)
            with open(curr_loc + "halo_first.pickle", "wb") as pickle_file:
                pickle.dump(curr_halo_first, pickle_file)
            with open(curr_loc + "halo_n.pickle", "wb") as pickle_file:
                pickle.dump(curr_halo_n, pickle_file)    

    parts = sim.split("_")
    combined_name += parts[1] + parts[2] + "s" + parts[4] 
    if i != len(use_sims)-1:
        combined_name += "_"

full_dset_loc = path_to_xgboost + combined_name + "/datasets/"
create_directory(full_dset_loc)

full_dataset = np.array([])
full_dataset_keys = np.array([])
full_hipids = np.array([])
full_labels = np.array([])
full_halo_first = np.array([])
full_halo_n = np.array([])

curr_start=0
for j,sim in enumerate(use_sims):
    with timed("Datasets for: "+sim+" stacked"):
        curr_loc = path_to_xgboost + sim + "/datasets/"

        with open(curr_loc + "dataset.pickle", "rb") as pickle_file:
            curr_dataset = pickle.load(pickle_file)
        with open(curr_loc + "keys.pickle", "rb") as pickle_file:
            curr_dataset_keys = pickle.load(pickle_file)
        with open(curr_loc + "hipids.pickle", "rb") as pickle_file:
            curr_hipids = pickle.load(pickle_file)
        with open(curr_loc + "labels.pickle", "rb") as pickle_file:
            curr_labels = pickle.load(pickle_file)
        with open(curr_loc + "halo_first.pickle", "rb") as pickle_file:
            curr_halo_first = pickle.load(pickle_file)
        with open(curr_loc + "halo_n.pickle", "rb") as pickle_file:
            curr_halo_n = pickle.load(pickle_file)
        
        # ensure that halo indexing works even with stacked simulations
        # TODO ensure that key order is the same for all simulations
        curr_start = curr_halo_first[-1] + curr_halo_n[-1]
        if j == 0:
            full_dataset = curr_dataset
            full_dataset_keys = curr_dataset_keys
            full_hipids = curr_hipids
            full_labels = curr_labels
            full_halo_first = curr_halo_first
            full_halo_n = curr_halo_n
        else:
            curr_halo_first = curr_halo_first + curr_start
            full_dataset = np.vstack((full_dataset,curr_dataset))
            full_hipids = np.vstack((full_hipids,curr_hipids))
            full_labels = np.vstack((full_labels,curr_labels))
            full_halo_first = np.vstack((full_halo_first,curr_halo_first))
            full_halo_n = np.vstack((full_halo_n,curr_halo_n))
    
# Split the dataset by percentage of halos 
halo_splt_idx = int(np.ceil((1-test_halos_ratio) * full_halo_n.size))
train_halo_n = full_halo_n[:halo_splt_idx]
test_halo_n = full_halo_n[halo_splt_idx:]
train_halo_first = full_halo_first[:halo_splt_idx]
test_halo_first = full_halo_first[halo_splt_idx:]

ptl_splt_idx = np.sum(train_halo_n)
train_dataset = full_dataset[:ptl_splt_idx]
test_dataset = full_dataset[ptl_splt_idx:]
train_hipids = full_hipids[:ptl_splt_idx]
test_hipids = full_hipids[ptl_splt_idx:]
train_labels = full_labels[:ptl_splt_idx]
test_labels = full_labels[ptl_splt_idx:]

with open(full_dset_loc + "keys.pickle", "wb") as pickle_file:
    pickle.dump(full_dataset_keys, pickle_file)

with open(full_dset_loc + "train_dataset.pickle", "wb") as pickle_file:
    pickle.dump(train_dataset, pickle_file)
with open(full_dset_loc + "train_hipids.pickle", "wb") as pickle_file:
    pickle.dump(train_hipids, pickle_file)
with open(full_dset_loc + "train_labels.pickle", "wb") as pickle_file:
    pickle.dump(train_labels, pickle_file)
with open(full_dset_loc + "train_halo_first.pickle", "wb") as pickle_file:
    pickle.dump(train_halo_first, pickle_file)
with open(full_dset_loc + "train_halo_n.pickle", "wb") as pickle_file:
    pickle.dump(train_halo_n, pickle_file)    
    
with open(full_dset_loc + "test_dataset.pickle", "wb") as pickle_file:
    pickle.dump(test_dataset, pickle_file)
with open(full_dset_loc + "test_hipids.pickle", "wb") as pickle_file:
    pickle.dump(test_hipids, pickle_file)
with open(full_dset_loc + "test_labels.pickle", "wb") as pickle_file:
    pickle.dump(test_labels, pickle_file)
with open(full_dset_loc + "test_halo_first.pickle", "wb") as pickle_file:
    pickle.dump(test_halo_first, pickle_file)
with open(full_dset_loc + "test_halo_n.pickle", "wb") as pickle_file:
    pickle.dump(test_halo_n, pickle_file)    
    