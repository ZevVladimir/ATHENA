import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
import time 
import pickle
import os
from imblearn import under_sampling, over_sampling
from data_and_loading_functions import standardize, build_ml_dataset, check_pickle_exist_gadget, check_pickle_exist_hdf5_prop
from visualization_functions import compare_density_prf, plot_radius_rad_vel_tang_vel_graphs

curr_snapshot = "190"
snapshot_index = int(curr_snapshot)
curr_hdf5_file = "sparta_190.hdf5"

sparta_location = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_hdf5_file
data_location = "/home/zvladimi/MLOIS/calculated_info/" + "calc_from_" + curr_hdf5_file + "/"
save_location = "/home/zvladimi/MLOIS/training_data/" + "data_for_" + curr_hdf5_file + "/"
snapshot_path = "/home/zvladimi/MLOIS/particle_data/snapshot_" + curr_snapshot + "/snapshot_0" + curr_snapshot

ptl_mass = check_pickle_exist_gadget(data_location, "mass", curr_snapshot, snapshot_path)
mass = ptl_mass[0] * 10**10 #units M_sun/h

np.random.seed(11)

num_splits = 1
param_list = ['Orbit_Infall', 'Scaled_radii', 'Radial_vel', 'Radial_vel_magn', 'Tangential_vel', 'Tangential_vel_magn']
num_test_halos = 5
rng = np.random.default_rng()
with h5py.File((data_location + "all_halo_properties" + curr_snapshot + ".hdf5"), 'r') as all_halo_properties:
    total_num_halos = all_halo_properties["Halo_id"][:].shape[0]   
    random_halo_indices = rng.permutation(total_num_halos)

dataset, halo_props = build_ml_dataset(save_location, data_location, random_halo_indices, curr_snapshot, param_list, "full")

num_test_ptl = np.sum(halo_props[:num_test_halos,1])
test_dataset = dataset[:num_test_ptl,:]
train_dataset = dataset[num_test_ptl:,:]

rus = under_sampling.RandomUnderSampler(random_state=0)
ros = over_sampling.RandomOverSampler(random_state=0)
t0 = time.time()
if os.path.exists(save_location + "xgb_model" + curr_snapshot + ".pickle"):
    model = pickle.load(open(save_location + "xgb_model" + curr_snapshot + ".pickle", "rb"))
else:
    model = None
    for i in range(num_splits):
        t1 = time.time()
        print("Split:", (i+1), "/",num_splits)
        
        

        t2 = time.time()
        print("Loaded data", t2 - t1, "seconds")

        X_train, X_val, y_train, y_val = train_test_split(train_dataset[:,1:], train_dataset[:,0], test_size=0.30, random_state=0)

        # standardize has slightly better performance comapared to normalize
        # X_train = standardize(X_train)
        # X_test = standardize(X_test)
        
        #X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
        #X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
        
        t3 = time.time()
        
        model = XGBClassifier(tree_method='gpu_hist')
        model = model.fit(X_train, y_train)

        predicts = model.predict(X_val)

        t4 = time.time()
        print("Fitted model", t4 - t3, "seconds")

        classification = classification_report(y_val, predicts)
        print(classification)

    pickle.dump(model, open(save_location + "xgb_model" + curr_snapshot + ".pickle", "wb"))
t5 = time.time()
print("Total time:", t5-t0, "seconds")

num_bins = 50


with h5py.File((sparta_location), 'r') as hdf5:
    halos_id = check_pickle_exist_hdf5_prop(data_location, "halos", "id", "", hdf5)
    density_prf_all = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_all", "", hdf5)
    density_prf_1halo = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_1halo", "", hdf5)
    halos_id = check_pickle_exist_hdf5_prop(data_location, "halos", "id", "", hdf5)
    halos_status = check_pickle_exist_hdf5_prop(data_location, "halos", "status", "", hdf5)
    halos_last_snap = check_pickle_exist_hdf5_prop(data_location, "halos", "last_snap", "", hdf5)

halos_id = halos_id[:,snapshot_index]
halos_status = halos_status[:,snapshot_index]
density_prf_all = density_prf_all[:,snapshot_index,:]
density_prf_1halo = density_prf_1halo[:,snapshot_index,:]
indices_keep = np.zeros((halos_id.size))
indices_keep = np.where((halos_last_snap >= snapshot_index) & (halos_status == 10))
halos_id = halos_id[indices_keep]
density_prf_all = density_prf_all[indices_keep]
density_prf_1halo = density_prf_1halo[indices_keep]

start = 0
for j in range(num_test_halos):
    curr_halo_num_ptl = halo_props[j,1]
    curr_halo_id = halo_props[j,2]

    curr_test_halo = test_dataset[start:start+curr_halo_num_ptl]

    test_predict = model.predict(curr_test_halo[:,1:])

    halo_idx = np.where(halos_id == curr_halo_id)[0]

    curr_density_prf_all = density_prf_all[halo_idx]
    curr_density_prf_1halo = density_prf_1halo[halo_idx]

    actual_labels = curr_test_halo[:,0]
    classification = classification_report(actual_labels, test_predict)
    print(classification)
    #compare_density_prf(curr_test_halo[:,1], curr_density_prf_all[0], curr_density_prf_1halo[0], mass, test_predict, j, "", "", show_graph = True)
    plot_radius_rad_vel_tang_vel_graphs(test_predict, curr_test_halo[:,1], curr_test_halo[:,5], curr_test_halo[:,9], actual_labels)
    start = start + curr_halo_num_ptl