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
from pairing import depair
from data_and_loading_functions import standardize, build_ml_dataset, check_pickle_exist_gadget, check_pickle_exist_hdf5_prop, choose_halo_split
from visualization_functions import compare_density_prf, plot_radius_rad_vel_tang_vel_graphs, graph_feature_importance, graph_correlation_matrix

# SHOULD BE DESCENDING
snapshot_list = [190, 176]
curr_sparta_file = "sparta_cbol_l0063_n0256"

sparta_location = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_sparta_file + ".hdf5"
data_location = "/home/zvladimi/MLOIS/calculated_info/" + "calc_from_" + curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[1]) + "/"
save_location = "/home/zvladimi/MLOIS/training_data/" + "data_for_" + curr_sparta_file + "/"
snapshot_path = "/home/zvladimi/MLOIS/particle_data/snapdir_" + "{:04d}".format(snapshot_list[0]) + "/snapshot_" + "{:04d}".format(snapshot_list[1])

print(snapshot_path)
ptl_mass = check_pickle_exist_gadget(data_location, "mass", str(snapshot_list[0]), snapshot_path)
mass = ptl_mass[0] * 10**10 #units M_sun/h

np.random.seed(11)

t1 = time.time()
num_splits = 1

train_dataset, all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "train", snapshot_list = snapshot_list)
test_dataset, all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "test", snapshot_list = snapshot_list)
print(train_dataset.shape)
print(test_dataset.shape)
print(all_keys)

dataset_df = pd.DataFrame(train_dataset[:,2:], columns = all_keys[2:])
print(dataset_df)
#graph_correlation_matrix(dataset_df, np.array(all_keys[2:]))
t2 = time.time()
print("Loaded data", t2 - t1, "seconds")

# rus = under_sampling.RandomUnderSampler(random_state=0)
# ros = over_sampling.RandomOverSampler(random_state=0)
t0 = time.time()

X_train, X_val, y_train, y_val = train_test_split(train_dataset[:,2:], train_dataset[:,1], test_size=0.30, random_state=0)

if os.path.exists(save_location + "xgb_model" + curr_sparta_file + ".pickle"):
    with open(save_location + "xgb_model" + curr_sparta_file + ".pickle", "rb") as pickle_file:
        model = pickle.load(pickle_file)
else:
    model = None
    for i in range(num_splits):
        
        print("Split:", (i+1), "/",num_splits)
        

        # standardize has slightly better performance comapared to normalize
        # X_train = standardize(X_train)
        # X_test = standardize(X_test)
        
        #X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
        #X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
        
        t3 = time.time()
        
        model = XGBClassifier(tree_method='gpu_hist', eta = 0.01, n_estimators = 100)
        model = model.fit(X_train, y_train)

        t4 = time.time()
        print("Fitted model", t4 - t3, "seconds")

    pickle.dump(model, open(save_location + "xgb_model" + curr_sparta_file + ".pickle", "wb"))
    t5 = time.time()
    print("Total time:", t5-t0, "seconds")

#graph_feature_importance(np.array(all_keys[2:]), model.feature_importances_)

predicts = model.predict(X_val)
classification = classification_report(y_val, predicts)
print(classification)

num_bins = 50
snapshot_index = int(snapshot_list[0])

curr_sparta_file = "sparta_cbol_l0063_n0256"
hdf5_file_path = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_sparta_file + ".hdf5"
density_prf_all = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_all", "", hdf5_file_path, curr_sparta_file)
density_prf_1halo = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_1halo", "", hdf5_file_path, curr_sparta_file)
halos_id = check_pickle_exist_hdf5_prop(data_location, "halos", "id", "", hdf5_file_path, curr_sparta_file)
halos_status = check_pickle_exist_hdf5_prop(data_location, "halos", "status", "", hdf5_file_path, curr_sparta_file)
halos_last_snap = check_pickle_exist_hdf5_prop(data_location, "halos", "last_snap", "", hdf5_file_path, curr_sparta_file)

p_halos_status = halos_status[:,snapshot_list[0]]
c_halos_status = halos_status[:,snapshot_list[1]]
density_prf_all = density_prf_all[:,snapshot_index,:]
density_prf_1halo = density_prf_1halo[:,snapshot_index,:]

match_halo_idxs = np.where((p_halos_status == 10) & (halos_last_snap >= 190) & (c_halos_status > 0) & (halos_last_snap >= snapshot_list[1]))[0]
#[ 321 6238  927 5728  599 3315 4395 4458 2159 7281]
test_indices = [321, 6238, 5728, 599, 3315, 4395, 4458, 2159, 7281]
num_test_halos = len(test_indices)
test_indices = match_halo_idxs[test_indices] 
test_density_prf_all = density_prf_all[test_indices]
test_density_prf_1halo = density_prf_1halo[test_indices]

halo_idxs = np.zeros(test_dataset.shape[0])
for i,id in enumerate(test_dataset[:,0]):
    halo_idxs[i] = depair(id)[1]

start = 0
for j in range(num_test_halos):
    print(j)
    curr_halo_idx = test_indices[j]
    curr_test_halo = test_dataset[np.where(halo_idxs == curr_halo_idx)]
    test_predict = model.predict(curr_test_halo[:,2:])

    curr_density_prf_all = test_density_prf_all[j]
    curr_density_prf_1halo = test_density_prf_1halo[j]
    actual_labels = curr_test_halo[:,1]

    classification = classification_report(actual_labels, test_predict)
    print(classification)
    #compare_density_prf(curr_test_halo[:,1], curr_density_prf_all, curr_density_prf_1halo, mass, test_predict, j, "", "", show_graph = True, save_graph = False)
    plot_radius_rad_vel_tang_vel_graphs(test_predict, curr_test_halo[:,4], curr_test_halo[:,2], curr_test_halo[:,6], actual_labels, "ML Predictions")
    plot_radius_rad_vel_tang_vel_graphs(actual_labels, curr_test_halo[:,4], curr_test_halo[:,2], curr_test_halo[:,6], actual_labels, "Actual Labels")
    plt.show()
