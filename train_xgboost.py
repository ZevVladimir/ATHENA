import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
import time 
import pickle
import os
from imblearn import under_sampling, over_sampling
from data_and_loading_functions import build_ml_dataset, check_pickle_exist_gadget, choose_halo_split, create_directory
from visualization_functions import *
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
create_directory(path_to_xgboost)
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
curr_chunk_size = config.getint("SEARCH","chunk_size")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
##################################################################################################################
# set what the paths should be for saving and getting the data
if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"

save_location = path_to_xgboost + specific_save
path_to_datasets = path_to_xgboost + specific_save + "datasets/"

with open(path_to_datasets + "train_dataset_all_keys.pickle", "rb") as pickle_file:
    train_all_keys = pickle.load(pickle_file)
with open(path_to_datasets + "train_dataset_" + curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(snapshot_list[-1]) + ".pickle", "rb") as pickle_file:
    train_dataset = pickle.load(pickle_file)


create_directory(save_location)

# Determine where the scaled radii, rad vel, and tang vel are located within the dtaset
for i,key in enumerate(train_all_keys[2:]):
    if key == "Scaled_radii_" + str(p_snap):
        radii_loc = i
    elif key == "Radial_vel_" + str(p_snap):
        rad_vel_loc = i 
    elif key == "Tangential_vel_" + str(p_snap):
        tang_vel_loc = i

t0 = time.time()
    
model_save_location = save_location + "models/" + model_name + "/"
create_directory(model_save_location)
print(model_save_location)
if os.path.exists(model_save_location + model_name + "_model.pickle"):
    with open(model_save_location + model_name + "_model.pickle", 'rb') as pickle_file:
        new_model = pickle.load(pickle_file)
else:
    new_model = model_creator(save_location=model_save_location, model_name=model_name, radii_splits=radii_splits, dataset=train_dataset, radii_loc=radii_loc, rad_vel_loc=rad_vel_loc, tang_vel_loc=tang_vel_loc, keys = train_all_keys)
    with open(model_save_location + model_name + "_model.pickle", 'wb') as pickle_file:
        pickle.dump(new_model, pickle_file, pickle.HIGHEST_PROTOCOL)

t3 = time.time()
("Start training model", model_name)
new_model.train_model()
t4 = time.time()
print("Finished training model",model_name,"in",np.round(((t4-t3)/60),2),"minutes")
predicts = new_model.ensemble_predict()
new_model.graph(corr_matrix = False, feat_imp = True)

t1 = time.time()  
print("Total time:", np.round(((t1-t0)/60),2), "minutes")