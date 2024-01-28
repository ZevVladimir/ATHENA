import numpy as np
import time 
from colossus.cosmology import cosmology
from data_and_loading_functions import build_ml_dataset, check_pickle_exist_gadget, choose_halo_split, create_directory
from visualization_functions import *
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