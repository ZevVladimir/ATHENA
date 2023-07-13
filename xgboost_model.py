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
import shap
from data_and_loading_functions import standardize, build_ml_dataset, check_pickle_exist_gadget, check_pickle_exist_hdf5_prop, choose_halo_split
from visualization_functions import compare_density_prf, plot_radius_rad_vel_tang_vel_graphs, graph_feature_importance, graph_correlation_matrix

# SHOULD BE DESCENDING
snapshot_list = ["190", "189"]
curr_sparta_file = "sparta_cbol_l0063_n0256"

sparta_location = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_sparta_file
data_location = "/home/zvladimi/MLOIS/calculated_info/" + "calc_from_" + curr_sparta_file + "_" + snapshot_list[-1] + "to" + snapshot_list[0] + "/"
save_location = "/home/zvladimi/MLOIS/training_data/" + "data_for_" + curr_sparta_file + "/"
snapshot_path = "/home/zvladimi/MLOIS/particle_data/snapshot_" + snapshot_list[-1] + "/snapshot_0" + snapshot_list[0]

ptl_mass = check_pickle_exist_gadget(data_location, "mass", snapshot_list[0], snapshot_path)
mass = ptl_mass[0] * 10**10 #units M_sun/h

np.random.seed(11)

t1 = time.time()
num_splits = 1



features = ['Orbit_Infall', 'Scaled_radii', 'Radial_vel_magn', 'Tangential_vel_magn', 'PIDS']
halo_prop_names = ['Halo_start_ind','Halo_num_ptl','Halo_id'] # use list to know for sure the order of numpy arrays
dataset_df = pd.DataFrame()
halo_prop_df = pd.DataFrame()
all_feature_names = np.array([])
all_halo_prop_names = np.array([])

if os.path.exists(save_location + curr_sparta_file + "full_dataset_" + snapshot_list[-1] + "_to_" + snapshot_list[0]) != True:
    for i, snap in enumerate(snapshot_list):
        print(snap)
        snap_feature_names = np.array([])
        snap_halo_prop_names = np.array([])
        for feature in features:
            if snap != snapshot_list[0] and feature == "Orbit_Infall":
                
                continue
            else:
                all_feature_names= np.append(all_feature_names, (feature + "_" + snap))
                snap_feature_names = np.append(snap_feature_names, (feature + "_" + snap))
        for prop in halo_prop_names:
            all_halo_prop_names = np.append(all_halo_prop_names, (prop + "_" + snap))
            snap_halo_prop_names = np.append(snap_halo_prop_names, (prop + "_" + snap))
            
        with h5py.File((data_location + "all_halo_properties" + curr_sparta_file + ".hdf5"), 'r') as all_halo_properties:
            total_num_halos = all_halo_properties["Halo_id_" + snap][:].shape[0]   

        dataset, halo_props = build_ml_dataset(save_location, data_location, curr_sparta_file, snap_feature_names, snap, total_num_halos, snap_halo_prop_names, snap)
        if snap == snapshot_list[0]:
            start_pids = dataset[:,4]
            for col,prop in enumerate(snap_halo_prop_names):
                halo_prop_df[prop] = halo_props[:,col]
        else:
            #TODO make it so it automatically chooses the right col index based off features list
            
            compare_pids = dataset[:,3]
            matching_elements, match_pids_start, match_pids_compare = np.intersect1d(start_pids, compare_pids, return_indices=True)

            dataset = dataset[match_pids_compare]

        if i == 0:
            start_dataset = dataset
        elif i == 1:
            start_dataset = start_dataset[match_pids_start]
            dataset = np.column_stack((start_dataset,dataset))
            for col,feature in enumerate(all_feature_names):
                dataset_df[feature] = dataset[:,col]
        else:
            for col,feature in enumerate(snap_feature_names):
                print(feature)
                dataset_df[feature] = dataset[:,col]

    dataset_df = dataset_df.drop(["PIDS_190", "PIDS_189"], axis = 1)

    pickle.dump(dataset_df, open(save_location + curr_sparta_file + "full_dataset_" + snapshot_list[-1] + "_to_" + snapshot_list[0], "wb"))
else:
    dataset_df = pickle.load(open(save_location + curr_sparta_file + "full_dataset_" + snapshot_list[-1] + "_to_" + snapshot_list[0], "rb"))

print(dataset_df)
rng = np.random.default_rng()
print(dataset_df.keys())
random_halo_indices = rng.permutation(halo_prop_df["Halo_id_" + snapshot_list[0]].to_numpy().shape[0])
num_test_halos = 10
num_validation_halos= 10

#graph_correlation_matrix(dataset_df, all_feature_names)
t2 = time.time()
print("Loaded data", t2 - t1, "seconds")

num_test_ptl = np.sum(halo_props[:num_test_halos,1])
# test_dataset = choose_halo_split(random_halo_indices[:num_test_halos], snapshot_list[0], halo_prop_df, dataset_df.to_numpy(), all_feature_names.size)
# validation_dataset = choose_halo_split(random_halo_indices[num_test_halos:num_test_halos+num_validation_halos], snapshot_list[0], halo_prop_df, dataset_df.to_numpy(), all_feature_names.size)
# training_dataset = choose_halo_split(random_halo_indices[num_test_halos+num_validation_halos:], snapshot_list[0], halo_prop_df, dataset_df.to_numpy(), all_feature_names.size)

# rus = under_sampling.RandomUnderSampler(random_state=0)
# ros = over_sampling.RandomOverSampler(random_state=0)
t0 = time.time()
if os.path.exists(save_location + "xgb_model" + curr_sparta_file + ".pickle"):
    model = pickle.load(open(save_location + "xgb_model" + curr_sparta_file + ".pickle", "rb"))
else:
    model = None
    for i in range(num_splits):
        
        print("Split:", (i+1), "/",num_splits)
       
        X_train, X_val, y_train, y_val = train_test_split(dataset_df.to_numpy()[:,1:], dataset_df.to_numpy()[:,0], test_size=0.30, random_state=0)

        # standardize has slightly better performance comapared to normalize
        # X_train = standardize(X_train)
        # X_test = standardize(X_test)
        
        #X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
        #X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
        
        t3 = time.time()
        
        model = XGBClassifier(tree_method='gpu_hist', n_estimators = 100)
        model = model.fit(X_train, y_train)

        t4 = time.time()
        print("Fitted model", t4 - t3, "seconds")

    pickle.dump(model, open(save_location + "xgb_model" + curr_sparta_file + ".pickle", "wb"))
t5 = time.time()
print("Total time:", t5-t0, "seconds")

graph_feature_importance(dataset_df.columns(), model.feature_importances_)

predicts = model.predict(X_val)
classification = classification_report(y_val, predicts)
print(classification)



num_bins = 50

# with h5py.File((sparta_location), 'r') as hdf5:
#     halos_id = check_pickle_exist_hdf5_prop(data_location, "halos", "id", "", hdf5)
#     density_prf_all = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_all", "", hdf5)
#     density_prf_1halo = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_1halo", "", hdf5)
#     halos_id = check_pickle_exist_hdf5_prop(data_location, "halos", "id", "", hdf5)
#     halos_status = check_pickle_exist_hdf5_prop(data_location, "halos", "status", "", hdf5)
#     halos_last_snap = check_pickle_exist_hdf5_prop(data_location, "halos", "last_snap", "", hdf5)

# halos_id = halos_id[:,snapshot_index]
# halos_status = halos_status[:,snapshot_index]
# density_prf_all = density_prf_all[:,snapshot_index,:]
# density_prf_1halo = density_prf_1halo[:,snapshot_index,:]
# indices_keep = np.zeros((halos_id.size))
# indices_keep = np.where((halos_last_snap >= snapshot_index) & (halos_status == 10))
# halos_id = halos_id[indices_keep]
# density_prf_all = density_prf_all[indices_keep]
# density_prf_1halo = density_prf_1halo[indices_keep]

# start = 0
# for j in range(num_test_halos):
#     curr_halo_num_ptl = halo_props[j,1]
#     curr_halo_id = halo_props[j,2]

#     curr_test_halo = test_dataset[start:start+curr_halo_num_ptl]

#     test_predict = model.predict(curr_test_halo[:,1:])

#     halo_idx = np.where(halos_id == curr_halo_id)[0]

#     curr_density_prf_all = density_prf_all[halo_idx]
#     curr_density_prf_1halo = density_prf_1halo[halo_idx]

#     actual_labels = curr_test_halo[:,0]
#     classification = classification_report(actual_labels, test_predict)
#     print(classification)
#     #compare_density_prf(curr_test_halo[:,1], curr_density_prf_all[0], curr_density_prf_1halo[0], mass, test_predict, j, "", "", show_graph = True, save_graph = False)
#     plot_radius_rad_vel_tang_vel_graphs(test_predict, curr_test_halo[:,1], curr_test_halo[:,5], curr_test_halo[:,9], actual_labels)
#     start = start + curr_halo_num_ptl