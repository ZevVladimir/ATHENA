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
from utils.data_and_loading_functions import check_pickle_exist_gadget, choose_halo_split, create_directory, save_to_hdf5
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

model_sims = json.loads(config.get("DATASET","model_sims"))
test_halos_ratio = config.getfloat("DATASET","test_halos_ratio")

global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
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
        raise ValueError("Input string:",input_str, "does not match the expected format.")

    # Extract the parts of the string
    prefix = input_str[:match.start()]
    first_number = match.group(1)
    
    # Construct the new string
    new_string = f"{first_number}_{prefix}"
    
    return new_string

def shorten_sim_name(input_str):
    pattern = r"(\d+)to(\d+)"
    match = re.search(pattern, input_str)

    if match:
        curr_snap_list = [match.group(1), match.group(2)] 
    else:
        raise ValueError("Input string:",input_str, "does not match the expected format.")
    parts = sim.split("_")
    sim_name = parts[1] + parts[2] + "s" + parts[4] 
    
    return sim_name, curr_snap_list

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
                        if row == 0:
                            dataset_keys[access_col] = "p_" + key
                        elif row == 1:
                            dataset_keys[access_col] = "c_" + key
                    curr_col += 1
                else:
                    full_dataset[:,curr_col] = all_ptl_properties[key]
                    dataset_keys[curr_col] = "p_" + key
                    curr_col += 1

    return full_dataset, dataset_keys, hipids, labels, halo_first, halo_n

def plot_nu_dist():
    nus = []
    fig, ax = plt.subplots(1, figsize=(30,15))
    for sim in model_sims:
        sim_pat = r"cbol_l(\d+)_n(\d+)"
        match = re.search(sim_pat, sim)
        if match:
            sparta_name = match.group(0)
        sim_search_pat = sim_pat + r"_(\d+)r200m"
        match = re.search(sim_search_pat, sim)
        if match:
            sparta_search_name = match.group(0)
        with h5py.File(path_to_SPARTA_data + sparta_name + "/" +  sparta_search_name + ".hdf5","r") as f:
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
        with h5py.File(curr_file,"r") as f:
            nus.append(peaks.peakHeight((f["Halo_n"][:]*ptl_mass), use_z))
   
    
    titlefntsize=26
    axisfntsize=24
    tickfntsize=20
    legendfntsize=22
    ax.hist(nus, bins=50, label=model_sims)
    ax.set_xlabel(r"$\nu$",fontsize=axisfntsize)
    ax.set_ylabel("Number of Halos",fontsize=axisfntsize)
    ax.tick_params(axis='both',which='both',labelsize=tickfntsize)
    ax.set_yscale("log")
    ax.legend(fontsize=legendfntsize)
    fig.savefig(path_to_MLOIS + "Random_figs/nu_dist.png")
    print("Finished nu_dist plot")
            
cosmol = cosmology.setCosmology("bolshoi")

np.random.seed(11)
plot_nu_dist()

# Create all the datasets for the simulations
combined_name=""
tot_num_halos=0
tot_num_ptls=0
hdf5_ptl_idx=0
hdf5_halo_idx=0

# Create individual datasets for each model requested
for i,sim in enumerate(model_sims):
    with timed("Datasets for: "+sim+" created"):
        sim_name, curr_snap_list = shorten_sim_name(sim)
        combined_name += sim_name
        if i != len(model_sims)-1:
            combined_name += "_"
            
        curr_loc = path_to_xgboost + sim_name + "/datasets/"
        create_directory(curr_loc)
    
        curr_dataset,curr_keys,curr_hipids,curr_labels,curr_halo_first,curr_halo_n = build_sim_dataset(sim,curr_snap_list)
        curr_num_halos = curr_halo_first.shape[0]
        curr_num_ptls = curr_labels.shape[0]
        tot_num_halos = tot_num_halos + curr_num_halos
        tot_num_ptls = tot_num_ptls + curr_num_ptls
    
        # Don't have to delete file here because we aren't actually iteratively adding to the same file
        with h5py.File((curr_loc + "full_dset.hdf5"), 'a') as curr_dset:
            save_to_hdf5(curr_dset, "Halo_first", dataset = curr_halo_first, chunk = True, max_shape = (curr_num_halos,))
            save_to_hdf5(curr_dset, "Halo_n", dataset = curr_halo_n, chunk = True, max_shape = (curr_num_halos,))
            save_to_hdf5(curr_dset, "HIPIDS", dataset = curr_hipids, chunk = True, max_shape = (curr_num_ptls,))
            save_to_hdf5(curr_dset, "Labels", dataset = curr_labels, chunk = True, max_shape = (curr_num_ptls,))
            save_to_hdf5(curr_dset, "Dataset", dataset = curr_dataset, chunk = True, max_shape = (curr_num_ptls,curr_keys.shape[0]))
        
        with open(curr_loc + "keys.pickle","wb") as file:
            pickle.dump(curr_keys, file)


full_dset_loc = path_to_xgboost + combined_name + "/datasets/"
create_directory(full_dset_loc)

# Now that we know all the simulations have datasets go through all the ones we wanted and combine them into 
# one large dataset that is split into training and testing
for j,sim in enumerate(model_sims):
    with timed("Datasets for: "+sim+" stacked"):
        sim_name, curr_snap_list = shorten_sim_name(sim)
        curr_loc = path_to_xgboost + sim_name + "/datasets/"

        with h5py.File(curr_loc + 'full_dset.hdf5', "r") as file:
            curr_dataset = file["Dataset"][:]
            curr_hipids = file["HIPIDS"][:]
            curr_labels = file["Labels"][:]
            curr_halo_first = file["Halo_first"][:]
            curr_halo_n = file["Halo_n"][:]
        with open(curr_loc + "keys.pickle","rb") as file:
            curr_keys = pickle.load(file)
        
        # TODO ensure that key order is the same for all simulations
        with open(path_to_calc_info + sim + "/all_indices.pickle", "rb") as pickle_file:
            all_idxs = pickle.load(pickle_file)
            
        halo_splt_idx = int(np.ceil((1-test_halos_ratio) * curr_halo_n.size))
        
        # Indexing: when stacking the simulations the halo_first catergory no longer indexes the entire dataset
        # It still corresponds to each individual simulation (indicating where it starts at 0). 
        # In addition when splitting into training and testing datasets we set the halo_first to 0 for the testing
        # set at the beginning as otherwise it will be starting at some random point.
        
        if os.path.isfile(full_dset_loc + "full_dset.hdf5") and i == 0:
            os.remove(full_dset_loc + "full_dset.hdf5")
        with h5py.File((full_dset_loc + "full_dset.hdf5"), 'a') as curr_dset:
            save_to_hdf5(curr_dset, "Halo_first", dataset = curr_halo_first, chunk = True, max_shape = (tot_num_halos,))
            save_to_hdf5(curr_dset, "Halo_n", dataset = curr_halo_n, chunk = True, max_shape = (tot_num_halos,))
            save_to_hdf5(curr_dset, "HIPIDS", dataset = curr_hipids, chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Labels", dataset = curr_labels, chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Halo_Indices", dataset = all_idxs, chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Dataset", dataset = curr_dataset, chunk = True, max_shape = (tot_num_ptls,curr_keys.shape[0]))
        
        ptl_splt_idx = np.sum(curr_halo_n[:halo_splt_idx])
        
        if os.path.isfile(full_dset_loc + "train_dset.hdf5") and i == 0:
            os.remove(full_dset_loc + "train_dset.hdf5")
        with h5py.File((full_dset_loc + "train_dset.hdf5"), 'a') as curr_dset:
            save_to_hdf5(curr_dset, "Halo_first", dataset = curr_halo_first[:halo_splt_idx], chunk = True, max_shape = (tot_num_halos,))
            save_to_hdf5(curr_dset, "Halo_n", dataset = curr_halo_n[:halo_splt_idx], chunk = True, max_shape = (tot_num_halos,))
            save_to_hdf5(curr_dset, "HIPIDS", dataset = curr_hipids[:ptl_splt_idx], chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Labels", dataset = curr_labels[:ptl_splt_idx], chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Halo_Indices", dataset = all_idxs[:halo_splt_idx], chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Dataset", dataset = curr_dataset[:ptl_splt_idx], chunk = True, max_shape = (tot_num_ptls,curr_keys.shape[0]))
        
        if os.path.isfile(full_dset_loc + "test_dset.hdf5") and i == 0:
            os.remove(full_dset_loc + "test_dset.hdf5")
        with h5py.File((full_dset_loc + "test_dset.hdf5"), 'a') as curr_dset:
            save_to_hdf5(curr_dset, "Halo_first", dataset = (curr_halo_first[halo_splt_idx:]- curr_halo_first[halo_splt_idx]), chunk = True, max_shape = (tot_num_halos,))
            save_to_hdf5(curr_dset, "Halo_n", dataset = curr_halo_n[halo_splt_idx:], chunk = True, max_shape = (tot_num_halos,))
            save_to_hdf5(curr_dset, "HIPIDS", dataset = curr_hipids[ptl_splt_idx:], chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Labels", dataset = curr_labels[ptl_splt_idx:], chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Halo_Indices", dataset = all_idxs[halo_splt_idx:], chunk = True, max_shape = (tot_num_ptls,))
            save_to_hdf5(curr_dset, "Dataset", dataset = curr_dataset[ptl_splt_idx:], chunk = True, max_shape = (tot_num_ptls,curr_keys.shape[0]))
            

        