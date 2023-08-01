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
from data_and_loading_functions import standardize, build_ml_dataset, check_pickle_exist_gadget, check_pickle_exist_hdf5_prop, choose_halo_split
from visualization_functions import *

cosmol = cosmology.setCosmology("bolshoi")

# SHOULD BE DESCENDING
snapshot_list = [190, 176]
p_snap = snapshot_list[0]
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

train_dataset, train_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "train", snapshot_list = snapshot_list)
test_dataset, test_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "test", snapshot_list = snapshot_list)
print(train_dataset.shape)
print(test_dataset.shape)
print(train_all_keys)

dataset_df = pd.DataFrame(train_dataset[:,2:], columns = train_all_keys[2:])
print(dataset_df)
#graph_correlation_matrix(dataset_df, np.array(all_keys[2:]))
t2 = time.time()
print("Loaded data", t2 - t1, "seconds")

# rus = under_sampling.RandomUnderSampler(random_state=0)
# ros = over_sampling.RandomOverSampler(random_state=0)
t0 = time.time()

feature_dist(train_dataset[:,2:], train_all_keys[2:], False, True)

X_train, X_val, y_train, y_val = train_test_split(train_dataset[:,2:], train_dataset[:,1], test_size=0.20, random_state=0)

low_cut_off = 0.8
high_cutoff = 3
X_train_below_r200m = X_train[np.where(X_train[:,4] < low_cut_off)[0]]
y_train_below_r200m = y_train[np.where(X_train[:,4] < low_cut_off)[0]]

X_train_r200m = X_train[np.where((X_train[:,4] >= low_cut_off) & (X_train[:,4] < high_cutoff))[0]]
y_train_r200m = y_train[np.where((X_train[:,4] >= low_cut_off) & (X_train[:,4] < high_cutoff))[0]]

X_train_great_r200m = X_train[np.where(X_train[:,4] > high_cutoff)[0]]
y_train_great_r200m = y_train[np.where(X_train[:,4] > high_cutoff)[0]]

def train_model(X, y, save_name, graph_feat_imp):
    if os.path.exists(save_location + "/models/" + "xgb_model_" + save_name + "_" + curr_sparta_file + ".pickle"):
        with open(save_location + "/models/" + "xgb_model_" + save_name + "_" + curr_sparta_file + ".pickle", "rb") as pickle_file:
            model = pickle.load(pickle_file)
    else:
        model = None

        t3 = time.time()
        
        model = XGBClassifier(tree_method='gpu_hist', eta = 0.01, n_estimators = 100)
        model = model.fit(X, y)

        t4 = time.time()
        print("Fitted model", t4 - t3, "seconds")

        pickle.dump(model, open(save_location + "/models/" + "xgb_model_" + save_name + "_" + curr_sparta_file + ".pickle", "wb"))
        t5 = time.time()
        print("Total time:", t5-t0, "seconds")
    if graph_feat_imp:
        graph_feature_importance(np.array(train_all_keys[2:]), model.feature_importances_, save_name, False, True)
       
    return model

def det_class(below, at, beyond):
    predicts = np.zeros(below.shape[0])
    prob_inf = (below[:,0] + at[:,0] + beyond[:,0])/3
    prob_orb = (below[:,1] + at[:,1] + beyond[:,1])/3

    predicts[np.where(prob_inf <= prob_orb)] = 1
    
    return predicts

model_below_r200m = train_model(X_train_below_r200m, y_train_below_r200m, "below_r200m", True)
model_at_r200m = train_model(X_train_r200m, y_train_r200m, "at_r200m", True)
model_beyond_r200m = train_model(X_train_great_r200m, y_train_great_r200m, "beyond_r200m", True)

predicts_below = model_below_r200m.predict_proba(X_val)
predicts_at = model_at_r200m.predict_proba(X_val)
predicts_beyond = model_beyond_r200m.predict_proba(X_val)

final_predicts = det_class(predicts_below, predicts_at, predicts_beyond)

classification = classification_report(y_val, final_predicts)
print(classification)

num_bins = 30
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
            if l == 0:
                test_halos_within = test_dataset[np.where(test_halo_idxs == idx)]
            else:
                curr_test_halo = test_dataset[np.where(test_halo_idxs == idx)]
                test_halos_within = np.row_stack((test_halos_within, curr_test_halo))
        print(test_halos_within.shape)
        # idx_around_r200m = np.where((curr_test_halo[:,4] > 0.9) & (curr_test_halo[:,4] < 1.1))[0]
        # test_predict = model.predict(curr_test_halo[idx_around_r200m,2:])\
            
        predicts_below = model_below_r200m.predict_proba(test_halos_within[:,2:])
        predicts_at = model_at_r200m.predict_proba(test_halos_within[:,2:])
        predicts_beyond = model_beyond_r200m.predict_proba(test_halos_within[:,2:])
        test_predict = det_class(predicts_below, predicts_at, predicts_below)
        #curr_density_prf_all = test_density_prf_all[k]
        #curr_density_prf_1halo = test_density_prf_1halo[k]
        # actual_labels = curr_test_halo[idx_around_r200m,1]
        actual_labels = test_halos_within[:,1]
        classification = classification_report(actual_labels, test_predict, output_dict=True)

        all_accuracy.append(classification["accuracy"])
        #compare_density_prf(curr_test_halo[:,1], curr_density_prf_all, curr_density_prf_1halo, mass, test_predict, k, "", "", show_graph = True, save_graph = False)
        # plot_radius_rad_vel_tang_vel_graphs(test_predict, curr_test_halo[idx_around_r200m,4], curr_test_halo[idx_around_r200m,2], curr_test_halo[idx_around_r200m,6], actual_labels, "ML Predictions", num_bins)
        # plot_radius_rad_vel_tang_vel_graphs(actual_labels, curr_test_halo[idx_around_r200m,4], curr_test_halo[idx_around_r200m,2], curr_test_halo[idx_around_r200m,6], actual_labels, "Actual Labels", num_bins)
        plot_radius_rad_vel_tang_vel_graphs(test_predict, test_halos_within[:,4], test_halos_within[:,2], test_halos_within[:,6], actual_labels, "ML Predictions", num_bins, start_nu, end_nu, plot = False, save = True)
        plot_radius_rad_vel_tang_vel_graphs(actual_labels, test_halos_within[:,4], test_halos_within[:,2], test_halos_within[:,6], actual_labels, "Actual Labels", num_bins, start_nu, end_nu, plot = False, save = True)
        #plt.show()
        graph_err_by_bin(test_predict, actual_labels, test_halos_within[:,4], num_bins, start_nu, end_nu, plot = False, save = True)
    
    start_nu = end_nu
    
print(all_accuracy)
fig, ax = plt.subplots(1,1)
ax.set_title("Number of halos at each accuracy level")
ax.set_xlabel("Accuracy")
ax.set_ylabel("Num halos")
ax.hist(all_accuracy)
fig.savefig("/home/zvladimi/MLOIS/Random_figures/all_test_halo_acc.png")