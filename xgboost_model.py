import numpy as np
import pandas as pd
import matplotlib as mpl
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
from pygadgetreader import readsnap, readheader
from colossus.cosmology import cosmology
from colossus.lss import peaks
from data_and_loading_functions import build_ml_dataset, check_pickle_exist_gadget, check_pickle_exist_hdf5_prop, choose_halo_split, create_directory
from visualization_functions import *

cosmol = cosmology.setCosmology("bolshoi")

# SHOULD BE DESCENDING
snapshot_list = [190]
p_snap = snapshot_list[0]
curr_sparta_file = "sparta_cbol_l0063_n0256"

if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "/"
    
sparta_location = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_sparta_file + ".hdf5"
if len(snapshot_list) > 1:
    data_location = "/home/zvladimi/MLOIS/calculated_info/" + specific_save
else:
    data_location = "/home/zvladimi/MLOIS/calculated_info/" + specific_save

if len(snapshot_list) > 1:
    save_location = "/home/zvladimi/MLOIS/xgboost_datasets_plots/" + specific_save
else:
    save_location = "/home/zvladimi/MLOIS/xgboost_datasets_plots/" + specific_save
create_directory(save_location)

snapshot_path = "/home/zvladimi/MLOIS/particle_data/snapdir_" + "{:04d}".format(snapshot_list[0]) + "/snapshot_" + "{:04d}".format(snapshot_list[0])

ptl_mass = check_pickle_exist_gadget(data_location, "mass", str(snapshot_list[0]), snapshot_path)
mass = ptl_mass[0] * 10**10 #units M_sun/h

np.random.seed(11)

t1 = time.time()
num_splits = 1

train_dataset, train_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "train", snapshot_list = snapshot_list)
test_dataset, test_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "test", snapshot_list = snapshot_list)
print(test_all_keys)
dataset_df = pd.DataFrame(train_dataset[:,2:], columns = train_all_keys[2:])
test_dataset_df = pd.DataFrame(test_dataset[:,2:], columns = test_all_keys[2:])
print(dataset_df)
print(test_dataset_df)
#graph_correlation_matrix(dataset_df, np.array(all_keys[2:]))
t2 = time.time()
print("Loaded data", t2 - t1, "seconds")

# rus = under_sampling.RandomUnderSampler(random_state=0)
# ros = over_sampling.RandomOverSampler(random_state=0)
t0 = time.time()

feature_dist(train_dataset[:,2:], train_all_keys[2:], "orig_data", False, True, save_location)
# train_dataset[:,2:] = normalize(train_dataset[:,2:])
# test_dataset[:,2:] = normalize(test_dataset[:,2:])
# feature_dist(train_dataset[:,2:], train_all_keys[2:], "norm_data", False, True)

X_train, X_val, y_train, y_val = train_test_split(train_dataset[:,2:], train_dataset[:,1], test_size=0.20, random_state=0)

for i,key in enumerate(train_all_keys[2:]):
    if key == "Scaled_radii_" + str(p_snap):
        scaled_radii_loc = i
    elif key == "Radial_vel_" + str(p_snap):
        rad_vel_loc = i
    elif key == "Tangential_vel_" + str(p_snap):
        tang_vel_loc = i

low_cut_off = 0.8
high_cutoff = 3
X_train_below_r200m = X_train[np.where(X_train[:,scaled_radii_loc] < low_cut_off)[0]]
y_train_below_r200m = y_train[np.where(X_train[:,scaled_radii_loc] < low_cut_off)[0]]

X_train_r200m = X_train[np.where((X_train[:,scaled_radii_loc] >= low_cut_off) & (X_train[:,scaled_radii_loc] < high_cutoff))[0]]
y_train_r200m = y_train[np.where((X_train[:,scaled_radii_loc] >= low_cut_off) & (X_train[:,scaled_radii_loc] < high_cutoff))[0]]

X_train_great_r200m = X_train[np.where(X_train[:,scaled_radii_loc] > high_cutoff)[0]]
y_train_great_r200m = y_train[np.where(X_train[:,scaled_radii_loc] > high_cutoff)[0]]


def train_model(X, y, rad_range, num_params, snapshots, graph_feat_imp, save_location):
    create_directory(save_location + "models/")
    model_location = save_location + "models/" + "xgb_model"
    for snap in snapshots:
        model_location = model_location + "_" + str(snap)
    model_location = model_location + "_" + rad_range + "_" + str(num_params) + "_" + curr_sparta_file + ".pickle"

    if os.path.exists(model_location):
        with open(model_location, "rb") as pickle_file:
            model = pickle.load(pickle_file)
    else:
        model = None

        t3 = time.time()
        
        model = XGBClassifier(tree_method='gpu_hist', eta = 0.01, n_estimators = 100)
        model = model.fit(X, y)

        t4 = time.time()
        print("Fitted model", t4 - t3, "seconds")

        pickle.dump(model, open(model_location, "wb"))
        t5 = time.time()
        print("Total time:", t5-t0, "seconds")
    if graph_feat_imp:
        graph_feature_importance(np.array(train_all_keys[2:]), model.feature_importances_, rad_range, False, True, save_location)
       
    return model

def det_class(below, at, beyond):
    predicts = np.zeros(below.shape[0])
    prob_inf = (below[:,0] + at[:,0] + beyond[:,0])/3
    prob_orb = (below[:,1] + at[:,1] + beyond[:,1])/3

    predicts[np.where(prob_inf <= prob_orb)] = 1
    
    return predicts

model_below_r200m = train_model(X_train_below_r200m, y_train_below_r200m, "below_r200m", len(train_all_keys), snapshot_list, True, save_location)
model_at_r200m = train_model(X_train_r200m, y_train_r200m, "at_r200m", len(train_all_keys), snapshot_list, True, save_location)
model_beyond_r200m = train_model(X_train_great_r200m, y_train_great_r200m, "beyond_r200m", len(train_all_keys), snapshot_list, True, save_location)

predicts_below = model_below_r200m.predict_proba(X_val)
predicts_at = model_at_r200m.predict_proba(X_val)
predicts_beyond = model_beyond_r200m.predict_proba(X_val)

final_predicts = det_class(predicts_below, predicts_at, predicts_beyond)

classification = classification_report(y_val, final_predicts)
print(classification)

num_bins = 30

hdf5_file_path = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_sparta_file + ".hdf5"
density_prf_all = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_all", "", hdf5_file_path, curr_sparta_file)
density_prf_1halo = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_1halo", "", hdf5_file_path, curr_sparta_file)
halos_id = check_pickle_exist_hdf5_prop(data_location, "halos", "id", "", hdf5_file_path, curr_sparta_file)
halos_status = check_pickle_exist_hdf5_prop(data_location, "halos", "status", "", hdf5_file_path, curr_sparta_file)
halos_last_snap = check_pickle_exist_hdf5_prop(data_location, "halos", "last_snap", "", hdf5_file_path, curr_sparta_file)

p_halos_status = halos_status[:,p_snap]
density_prf_all = density_prf_all[:,p_snap,:]
density_prf_1halo = density_prf_1halo[:,p_snap,:]

if len(snapshot_list) > 1:
    c_halos_status = halos_status[:,snapshot_list[1]]
    match_halo_idxs = np.where((p_halos_status == 10) & (halos_last_snap >= p_snap) & (c_halos_status > 0) & (halos_last_snap >= snapshot_list[1]))[0]
else:
    match_halo_idxs = np.where((p_halos_status == 10) & (halos_last_snap >= p_snap))[0]
    
with open(data_location + "test_indices.pickle", "rb") as pickle_file:
    test_indices = pickle.load(pickle_file)

num_test_halos = test_indices.shape[0]
test_indices = match_halo_idxs[test_indices] 
test_density_prf_all = density_prf_all[test_indices]
test_density_prf_1halo = density_prf_1halo[test_indices]

test_halo_idxs = np.zeros(test_dataset.shape[0])
for i,id in enumerate(test_dataset[:,0]):
    test_halo_idxs[i] = depair(id)[1]

halo_masses = np.zeros(num_test_halos)

start = 0
all_accuracy = []
for j in range(num_test_halos):
    curr_halo_idx = test_indices[j]
    curr_test_halo = test_dataset[np.where(test_halo_idxs == curr_halo_idx)]
    halo_masses[j] = curr_test_halo.shape[0] * mass

snapshot_path = "/home/zvladimi/MLOIS/particle_data/snapdir_" + "{:04d}".format(p_snap) + "/snapshot_" + "{:04d}".format(p_snap)
p_red_shift = readheader(snapshot_path, 'redshift')
peak_heights = peaks.peakHeight(halo_masses, p_red_shift)

start_nu = 0 
nu_step = 0.5
num_iter = 7

for k in range(num_iter):
    end_nu = start_nu + nu_step
    
    idx_within_nu = np.where((peak_heights >= start_nu) & (peak_heights < end_nu))[0]
    curr_test_halo_idxs = test_indices[idx_within_nu]
    print(start_nu, "to", end_nu, ":", idx_within_nu.shape, "halos")

    #curr_halo_idx = test_indices[k]
    if curr_test_halo_idxs.shape[0] != 0:
        for l, idx in enumerate(curr_test_halo_idxs):
            print(idx)
            if l == 0:
                test_halos_within = test_dataset[np.where(test_halo_idxs == idx)]
                break
            else:
                curr_test_halo = test_dataset[np.where(test_halo_idxs == idx)]
                test_halos_within = np.row_stack((test_halos_within, curr_test_halo))
        
        print(test_halos_within)
        predicts_below = model_below_r200m.predict_proba(test_halos_within[:,2:])
        predicts_at = model_at_r200m.predict_proba(test_halos_within[:,2:])
        predicts_beyond = model_beyond_r200m.predict_proba(test_halos_within[:,2:])
        test_predict = det_class(predicts_below, predicts_at, predicts_below)
        #curr_density_prf_all = test_density_prf_all[k]
        #curr_density_prf_1halo = test_density_prf_1halo[k]

        actual_labels = test_halos_within[:,1]
        print(actual_labels)
        print(test_predict)
        classification = classification_report(actual_labels, test_predict, output_dict=True)

        all_accuracy.append(classification["accuracy"])
        #compare_density_prf(curr_test_halo[:,1], curr_density_prf_all, curr_density_prf_1halo, mass, test_predict, k, "", "", show_graph = True, save_graph = False)
        # plot_radius_rad_vel_tang_vel_graphs(test_predict, curr_test_halo[idx_around_r200m,scaled_radii_loc], curr_test_halo[idx_around_r200m,2], curr_test_halo[idx_around_r200m,6], actual_labels, "ML Predictions", num_bins)
        # plot_radius_rad_vel_tang_vel_graphs(actual_labels, curr_test_halo[idx_around_r200m,scaled_radii_loc], curr_test_halo[idx_around_r200m,2], curr_test_halo[idx_around_r200m,6], actual_labels, "Actual Labels", num_bins)
        plot_radius_rad_vel_tang_vel_graphs(test_predict, test_halos_within[:,2+scaled_radii_loc], test_halos_within[:,2+rad_vel_loc], test_halos_within[:,2+tang_vel_loc], actual_labels, "ML Predictions", num_bins, start_nu, end_nu, plot = False, save = True, save_location=save_location)
        plot_radius_rad_vel_tang_vel_graphs(actual_labels, test_halos_within[:,2+scaled_radii_loc], test_halos_within[:,2+rad_vel_loc], test_halos_within[:,2+tang_vel_loc], actual_labels, "Actual Labels", num_bins, start_nu, end_nu, plot = False, save = True, save_location=save_location)
        #plt.show()
        graph_err_by_bin(test_predict, actual_labels, test_halos_within[:,2+scaled_radii_loc], num_bins, start_nu, end_nu, plot = False, save = True, save_location = save_location)
    
    start_nu = end_nu
    
print(all_accuracy)
fig, ax = plt.subplots(1,1)
ax.set_title("Number of halos at each accuracy level")
ax.set_xlabel("Accuracy")
ax.set_ylabel("Num halos")
ax.hist(all_accuracy)
fig.savefig(save_location + "all_test_halo_acc.png")