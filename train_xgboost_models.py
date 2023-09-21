import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
import time 
import pickle
import os
from imblearn import under_sampling, over_sampling
##################################################################################################################
# General params
snapshot_list = [190,183] # SHOULD BE DESCENDING
p_snap = snapshot_list[0]
times_r200m = 6
curr_sparta_file = "sparta_cbol_l0063_n0256"
path_to_MLOIS = "/home/zvladimi/MLOIS/"
path_to_snapshot = "/home/zvladimi/MLOIS/particle_data/"
path_to_hdf5 = "/home/zvladimi/MLOIS/SPARTA_data/"
path_to_pickle = "/home/zvladimi/MLOIS/pickle_data/"
path_to_model_plots = "/home/zvladimi/MLOIS/xgboost_datasets_plots/"
snap_format = "{:04d}" # how are the snapshots formatted with 0s
##################################################################################################################
from data_and_loading_functions import create_directory
from visualization_functions import *
from xgboost_model_creator import model_creator
##################################################################################################################
# set what the paths should be for saving and getting the data
if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(times_r200m) + "r200msearch/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(times_r200m) + "r200msearch/"

save_location = path_to_model_plots + specific_save
path_to_datasets = path_to_model_plots + specific_save + "datasets/"

with open(path_to_datasets + "train_dataset_all_keys.pickle", "rb") as pickle_file:
    train_all_keys = pickle.load(pickle_file)
with open(path_to_datasets + "train_dataset_" + curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(snapshot_list[-1]) + ".pickle", "rb") as pickle_file:
    train_dataset = pickle.load(pickle_file)

if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(times_r200m) + "r200msearch/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(times_r200m) + "r200msearch/"
    
save_location = path_to_model_plots + specific_save

create_directory(save_location)

# Determine where the scaled radii, rad vel, and tang vel are located within the dtaset
for i,key in enumerate(train_all_keys[2:]):
    if key == "Scaled_radii_" + str(p_snap):
        scaled_radii_loc = i
    elif key == "Radial_vel_" + str(p_snap):
        rad_vel_loc = i
    elif key == "Tangential_vel_" + str(p_snap):
        tang_vel_loc = i

t0 = time.time()
all_models = []
num_params_per_snap = (len(train_all_keys) - 2) / len(snapshot_list)

# Train the models based off the number of snaps (create an object for each snapshot, adding one each loop as to cover all info)
for i in range(len(snapshot_list)):
    curr_dataset = train_dataset[:,:int(2 + (num_params_per_snap * (i+1)))]
    curr_dataset = curr_dataset[np.where(curr_dataset[:,-1] != 0)]
    
    create_directory(save_location + "models/")
    if os.path.exists(save_location + "models/" + str(int(2+num_params_per_snap*(i+1))) + "_param_model.pickle"):
        with open(save_location + "models/" + str(int(2+num_params_per_snap*(i+1))) + "_param_model.pickle", 'rb') as pickle_file:
            new_model = pickle.load(pickle_file)
    else:
        new_model = model_creator(dataset=curr_dataset, keys=train_all_keys[:int(2+num_params_per_snap*(i+1))], snapshot_list=snapshot_list, num_params_per_snap=int((num_params_per_snap * (i+1))+2), save_location=save_location, scaled_radii_loc=scaled_radii_loc, rad_vel_loc=rad_vel_loc, tang_vel_loc=tang_vel_loc, radii_splits=[0.8,1.3], curr_sparta_file=curr_sparta_file)
        new_model.sub_models = new_model.get_sub_models()
        with open(save_location + "models/" + str(int(2+num_params_per_snap*(i+1))) + "_param_model.pickle", 'wb') as pickle_file:
            pickle.dump(new_model, pickle_file, pickle.HIGHEST_PROTOCOL)
    all_models.append(new_model)
    t3 = time.time()
    print("Start training model",i,)
    all_models[i].train_model()
    t4 = time.time()
    print("Finished training model",i,"in",np.round(((t4-t3)/60),2),"minutes")
    all_models[i].predict(1)
    all_models[i].graph(corr_matrix = True, feat_imp = True)

t1 = time.time()  
print("Total time:", np.round(((t1-t0)/60),2), "minutes")