import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
import time 
import pickle
import os
from imblearn import under_sampling, over_sampling
from pairing import depair
from colossus.cosmology import cosmology
from colossus.lss import peaks
from data_and_loading_functions import conv_halo_id_spid, create_directory, load_or_pickle_SPARTA_data, create_directory
from xgboost_model_creator import model_creator, Iterator
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
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
c_snap = config.getint("XGBOOST","c_snap")
model_name = config["XGBOOST"]["model_name"]
radii_splits = config.get("XGBOOST","rad_splits").split(',')
for split in radii_splits:
    model_name = model_name + "_" + str(split)
snapshot_list = [p_snap, c_snap]
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
##################################################################################################################
# import pygadgetreader and sparta
import sys
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta import sparta
from visualization_functions import *
#from train_xgboost_models import model_creator
##################################################################################################################
# set what the paths should be for saving and getting the data
if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"

path_to_datasets = path_to_xgboost + specific_save + "datasets/"
data_location = path_to_calc_info + specific_save
save_location = path_to_xgboost + specific_save + "models/" + model_name + "/"

create_directory(save_location)

p_snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

p_red_shift = readheader(p_snapshot_path, 'redshift')
p_scale_factor = 1/(1+p_red_shift)
halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, ptl_mass = load_or_pickle_SPARTA_data(curr_sparta_file, p_scale_factor, p_snap)
cosmol = cosmology.setCosmology("bolshoi")

with open(path_to_xgboost + specific_save + "test_keys.pickle", "rb") as pickle_file:
    test_all_keys = pickle.load(pickle_file)

num_params_per_snap = (len(test_all_keys) - 2) / len(snapshot_list)

num_bins = 30
    
# Determine where the scaled radii, rad vel, and tang vel are located within the dataset
for i,key in enumerate(test_all_keys[2:]):
    if key == "Scaled_radii_" + str(p_snap):
        scaled_radii_loc = i
    elif key == "Radial_vel_" + str(p_snap):
        rad_vel_loc = i
    elif key == "Tangential_vel_" + str(p_snap):
        tang_vel_loc = i

    
with open(data_location + "test_indices.pickle", "rb") as pickle_file:
    test_indices = pickle.load(pickle_file)
use_halo_ids = halos_id[test_indices]
sparta_output = sparta.load(filename=path_to_hdf5_file, halo_ids=use_halo_ids, log_level=0)
new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_snap) # If the order changed by sparta resort the indices
dens_prf_all = sparta_output['anl_prf']['M_all'][new_idxs,p_snap,:]
dens_prf_1halo = sparta_output['anl_prf']['M_1halo'][new_idxs,p_snap,:]

# test indices are the indices of the match halo idxs used (see find_particle_properties_ML.py to see how test_indices are created)
num_test_halos = test_indices.shape[0]

density_prf_all_within = np.sum(dens_prf_all, axis=0)
density_prf_1halo_within = np.sum(dens_prf_1halo, axis=0)

paths_to_test_data = []
for path, subdirs, files in os.walk(path_to_xgboost + curr_sparta_file + "_" + str(p_snap) + "to" + str(c_snap) + "_" + str(search_rad) + "r200msearch/test_split_datasets/"):
    files.sort()
    for name in files:
        paths_to_test_data.append(os.path.join(path, name))

for i,path in enumerate(paths_to_test_data):
    with open(path, "rb") as file:
        curr_dataset = pickle.load(file)
        curr_labels = curr_dataset[:,1]
    if i == 0:
        actual_labels = curr_labels
    else:
        actual_labels = np.concatenate((actual_labels, curr_labels), axis=0)

test_it = Iterator(paths_to_test_data, inc_labels=False)
test_dataset = xgboost.DMatrix(test_it)

curr_model_location = save_location + "range_all_" + curr_sparta_file + ".json"
model = xgboost.Booster()
model.load_model(curr_model_location)

predicts = model.predict(test_dataset)
predicts = np.round(predicts)

print(classification_report(actual_labels, predicts))

for i,path in enumerate(paths_to_test_data):
    with open(path, "rb") as file:
        curr_dataset = pickle.load(file)
    if i == 0:
        test_dataset = curr_dataset
    else:
        test_dataset = np.concatenate((test_dataset, curr_dataset), axis=0)

bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
bins = np.insert(bins, 0, 0)
compare_density_prf(radii=test_dataset[:,2+scaled_radii_loc], actual_prf_all=density_prf_all_within, actual_prf_1halo=density_prf_1halo_within, mass=ptl_mass, orbit_assn=predicts, prf_bins=bins, title = model_name + " Predicts", show_graph = False, save_graph = True, save_location = save_location)
plot_radius_rad_vel_tang_vel_graphs(predicts, test_dataset[:,2+scaled_radii_loc], test_dataset[:,2+rad_vel_loc], test_dataset[:,2+tang_vel_loc], actual_labels, model_name + " Predicts", num_bins, show = False, save = True, save_location=save_location)
graph_acc_by_bin(predicts, actual_labels, test_dataset[:,2+scaled_radii_loc], num_bins, model_name + " Predicts", plot = False, save = True, save_location = save_location)
