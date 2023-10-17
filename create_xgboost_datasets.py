import numpy as np
import time 
from colossus.cosmology import cosmology
##################################################################################################################
# General params
snapshot_list = [190,184] # SHOULD BE DESCENDING
p_snap = snapshot_list[0]
times_r200m = 6

curr_sparta_file = "sparta_cbol_l0063_n0256"
path_to_hdf5_file = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_sparta_file + ".hdf5"
path_dict = {
    "curr_sparta_file": curr_sparta_file,
    "path_to_MLOIS": "/home/zvladimi/MLOIS/",
    "path_to_snaps": "/home/zvladimi/MLOIS/particle_data/",
    "path_to_hdf5_file": path_to_hdf5_file,
    "path_to_pickle": "/home/zvladimi/MLOIS/pickle_data/",
    "path_to_datasets": "/home/zvladimi/MLOIS/calculated_info/",
    "path_to_model_plots": "/home/zvladimi/MLOIS/xgboost_datasets_plots/"
}

snap_format = "{:04d}" # how are the snapshots formatted with 0s
##################################################################################################################
# import pygadgetreader and sparta
import sys
sys.path.insert(0, path_dict["path_to_MLOIS"] + "pygadgetreader")
sys.path.insert(0, path_dict["path_to_MLOIS"] + "sparta/analysis")
from pygadgetreader import readsnap, readheader
from sparta import sparta
from data_and_loading_functions import build_ml_dataset, check_pickle_exist_gadget, choose_halo_split, create_directory
from visualization_functions import *
##################################################################################################################

# set what the paths should be for saving and getting the data
if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(times_r200m) + "r200msearch/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(times_r200m) + "r200msearch/"
    
data_location = path_dict["path_to_datasets"] + specific_save
save_location = path_dict["path_to_model_plots"] + specific_save

create_directory(save_location)

snapshot_path = path_dict["path_to_snaps"] + "snapdir_" + snap_format.format(snapshot_list[0]) + "/snapshot_" + snap_format.format(snapshot_list[0])

ptl_mass = check_pickle_exist_gadget(path_dict["curr_sparta_file"], "mass", str(snapshot_list[0]), snapshot_path, path_dict=path_dict)
mass = ptl_mass[0] * 10**10 #units M_sun/h
cosmol = cosmology.setCosmology("bolshoi")

np.random.seed(11)

t1 = time.time()
print("Start train dataset creation")
# Create the separate datasets for training and testing
train_dataset, train_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "train", snapshot_list = snapshot_list)
t2 = time.time()
print("Time taken:", np.round((t2-t1),2),"seconds")

print("Start test dataset creation")
test_dataset, test_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "test", snapshot_list = snapshot_list)
t3 = time.time()
print("Time taken:", np.round((t3-t1),2),"seconds")
