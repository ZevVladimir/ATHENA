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
snapshot_list = [190,160]
times_r200m = 6
p_snap = snapshot_list[0]
curr_sparta_file = "sparta_cbol_l0063_n0256"

if len(snapshot_list) > 1:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(times_r200m) + "r200msearch/"
else:
    specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(times_r200m) + "r200msearch/"
    
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

ptl_mass = check_pickle_exist_gadget("mass", str(snapshot_list[0]), snapshot_path)
mass = ptl_mass[0] * 10**10 #units M_sun/h

np.random.seed(11)

t1 = time.time()


train_dataset, train_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "train", snapshot_list = snapshot_list)
test_dataset, test_all_keys = build_ml_dataset(save_path = save_location, data_location = data_location, sparta_name = curr_sparta_file, dataset_name = "test", snapshot_list = snapshot_list)
print(test_all_keys)
print(train_all_keys)
for i,key in enumerate(train_all_keys[2:]):
    if key == "Scaled_radii_" + str(p_snap):
        scaled_radii_loc = i
    elif key == "Radial_vel_" + str(p_snap):
        rad_vel_loc = i
    elif key == "Tangential_vel_" + str(p_snap):
        tang_vel_loc = i
print(scaled_radii_loc)

class model_creator:
    def __init__(self, dataset, keys, snapshot_list, num_params_per_snap, save_location, scaled_radii_loc, rad_vel_loc, tang_vel_loc, radii_splits):
        self.dataset = dataset
        self.keys = keys
        self.snapshot_list = snapshot_list
        self.num_params = num_params_per_snap
        self.save_location = save_location
        self.scaled_radii_loc = scaled_radii_loc
        self.rad_vel_loc = rad_vel_loc
        self.tang_vel_loc = tang_vel_loc
        self.radii_splits = radii_splits
        print(self.keys)
        self.dataset_df = pd.DataFrame(dataset[:,2:], columns = keys[2:])
        self.train_val_split()

    def train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.dataset[:,2:], self.dataset[:,1], test_size=0.20, random_state=0)
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.mix_data()
    
    def mix_data(self):
        train_rand_idxs = np.random.permutation(self.X_train.shape[0])
        val_rand_idxs = np.random.permutation(self.X_val.shape[0])
        self.X_train = self.X_train[train_rand_idxs]
        self.y_train = self.y_train[train_rand_idxs]
        self.X_val = self.X_val[val_rand_idxs]
        self.y_val = self.y_val[val_rand_idxs]

    def over_sample():
        return
    
    def under_sample():
        return
    
    def normalize():
        return
    
    def standardisze():
        return
    
    def split_by_dist(self, low_cutoff, high_cutoff):
        X_train_within = self.X_train[np.where((self.X_train[:,scaled_radii_loc] > low_cutoff) & (self.X_train[:,scaled_radii_loc] < high_cutoff))[0]]
        y_train_within = self.y_train[np.where((self.X_train[:,scaled_radii_loc] > low_cutoff) & (self.X_train[:,scaled_radii_loc] < high_cutoff))[0]]

        return X_train_within, y_train_within

    def train_model(self):
        create_directory(self.save_location + "models/")
        model_location = self.save_location + "models/" + "xgb_model"

        sub_models = []

        for snap in self.snapshot_list:
            model_location = model_location + "_" + str(snap)

        for i in range(len(self.radii_splits) + 1):
            if i == 0:
                low_cutoff = 0
            else:
                low_cutoff = self.radii_splits[i - 1]
            if i == len(self.radii_splits):
                high_cutoff = np.max(self.X_train[:,self.scaled_radii_loc])
            else:
                high_cutoff = self.radii_splits[i]
            
            X, y = self.split_by_dist(low_cutoff, high_cutoff)
        
            curr_model_location = model_location + "_range_" + str(low_cutoff) + "_" + str(np.round(high_cutoff,2)) + "_params_" + str(self.num_params) + "_" + curr_sparta_file + ".pickle"

            if os.path.exists(curr_model_location):
                with open(curr_model_location, "rb") as pickle_file:
                    model = pickle.load(pickle_file)
            else:
                model = None

                t3 = time.time()
                
                model = XGBClassifier(tree_method='gpu_hist', eta = 0.01, n_estimators = 100)
                model = model.fit(X, y)

                t4 = time.time()
                print("Fitted model", t4 - t3, "seconds")

                pickle.dump(model, open(curr_model_location, "wb"))
            
            sub_models.append(model)
        # if graph_feat_imp:
        #     graph_feature_importance(np.array(self.keys), model.feature_importances_, rad_range, False, True, self.save_location)
        
        self.sub_models = sub_models

    def predict(self, dataset):
        if np.all(dataset):
            use_dataset = self.X_val
            use_labels = self.y_val
        else:
            use_dataset = dataset[:,2:]
            use_labels = dataset[:,1]

        all_predicts = np.zeros((use_dataset.shape[0],(len(self.sub_models)*2)))
        for i,model in enumerate(self.sub_models):
            all_predicts[:,2*i:(2*i+2)] = model.predict_proba(use_dataset)

        self.det_class(all_predicts)
        print(classification_report(use_labels, self.predicts))
    
    def det_class(self, predicts):
        pred_loc = np.argmax(predicts, axis = 1)

        final_predicts = np.zeros(predicts.shape[0])
        final_predicts[np.where((pred_loc % 2) != 0)] = 1

        self.predicts = final_predicts

    def graph(self, corr_matrix = False,):
        if corr_matrix:
            graph_correlation_matrix(self.dataset_df, self.save_location, show = False, save = True)
        
    def get_predicts(self):
        return self.predicts

t0 = time.time()
all_models = []
num_params_per_snap = (len(train_all_keys) - 2) / len(snapshot_list)
print(train_dataset.shape)
for i in range(len(snapshot_list)):
    curr_dataset = train_dataset[:,:int(2 + (num_params_per_snap * (i+1)))]
    curr_dataset = curr_dataset[np.where(curr_dataset[:,-1] != 0)]
    all_models.append(model_creator(dataset=curr_dataset, keys=train_all_keys[:int(2+num_params_per_snap*(i+1))], snapshot_list=snapshot_list, num_params_per_snap=int((num_params_per_snap * (i+1))+2), save_location=save_location, scaled_radii_loc=scaled_radii_loc, rad_vel_loc=rad_vel_loc, tang_vel_loc=tang_vel_loc, radii_splits=[0.8,1.3]))
    all_models[i].train_model()
    all_models[i].predict(1)

t5 = time.time()  
print("Total time:", t5-t0, "seconds")

num_bins = 30

# load halo information for comparison
hdf5_file_path = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_sparta_file + ".hdf5"
density_prf_all = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_all", "", hdf5_file_path, curr_sparta_file)
density_prf_1halo = check_pickle_exist_hdf5_prop(data_location, "anl_prf", "M_1halo", "", hdf5_file_path, curr_sparta_file)
halos_status = check_pickle_exist_hdf5_prop(data_location, "halos", "status", "", hdf5_file_path, curr_sparta_file)
halos_last_snap = check_pickle_exist_hdf5_prop(data_location, "halos", "last_snap", "", hdf5_file_path, curr_sparta_file)

# choose the halos for the primary snap to compare against
p_halos_status = halos_status[:,p_snap]
density_prf_all = density_prf_all[:,p_snap,:]
density_prf_1halo = density_prf_1halo[:,p_snap,:]

    
with open(data_location + "test_indices.pickle", "rb") as pickle_file:
    test_indices = pickle.load(pickle_file)

# test indices are the indices of the match halo idxs used (see find_particle_properties_ML.py to see how test_indices are created)
num_test_halos = test_indices.shape[0]

# for every halo idx and pid paired in the test dataset get the halo idxs
test_halo_idxs = np.zeros(test_dataset.shape[0])
for i,id in enumerate(test_dataset[:,0]):
    test_halo_idxs[i] = depair(id)[1]

halo_masses = np.zeros(num_test_halos)

start = 0
all_accuracy = []
# get the mases for each halo
for j in range(num_test_halos):
    curr_halo_idx = test_indices[j]
    curr_test_halo = test_dataset[np.where(test_halo_idxs == curr_halo_idx)]
    halo_masses[j] = curr_test_halo.shape[0] * mass

snapshot_path = "/home/zvladimi/MLOIS/particle_data/snapdir_" + "{:04d}".format(p_snap) + "/snapshot_" + "{:04d}".format(p_snap)
p_red_shift = readheader(snapshot_path, 'redshift')
peak_heights = peaks.peakHeight(halo_masses, p_red_shift)

# for i,idx in enumerate(test_indices):
#     curr_test_halo = test_dataset[np.where(test_halo_idxs == idx)]

#     test_predict = np.ones(curr_test_halo.shape[0]) * -1
#     for i in range(len(snapshot_list)):
#         curr_dataset = curr_test_halo[:,:int(2 + (num_params_per_snap * (i+1)))]

#         if i == (len(snapshot_list)-1):
#             use_ptls = np.where(curr_test_halo[:,-1]!= 0)[0]
#         else:
#             use_ptls = np.where(curr_test_halo[:,int(2 + (num_params_per_snap * (i+1)))] == 0)[0]

#         curr_dataset = curr_dataset[use_ptls]
#         all_models[i].predict(curr_dataset)
#         test_predict[use_ptls] = all_models[i].get_predicts()

#     compare_density_prf(curr_test_halo[:,2+scaled_radii_loc], density_prf_all[idx], density_prf_1halo[idx], mass, test_predict, title = str(idx), show_graph = True, save_graph = True, save_location = save_location)
                


start_nu = 0 
nu_step = 0.5
num_iter = 7

for i in range(num_iter):
    end_nu = start_nu + nu_step

    idx_within_nu = np.where((peak_heights >= start_nu) & (peak_heights < end_nu))[0]
    curr_test_halo_idxs = test_indices[idx_within_nu]
    print(start_nu, "to", end_nu, ":", idx_within_nu.shape, "halos")

    if curr_test_halo_idxs.shape[0] != 0:
        density_prf_all_within = np.zeros(density_prf_all.shape[1])
        density_prf_1halo_within = np.zeros(density_prf_1halo.shape[1])
        for j, idx in enumerate(curr_test_halo_idxs):
            density_prf_all_within = density_prf_all_within + density_prf_all[curr_test_halo_idxs[j]]
            density_prf_1halo_within = density_prf_1halo_within + density_prf_1halo[curr_test_halo_idxs[j]]
            if j == 0:
                test_halos_within = test_dataset[np.where(test_halo_idxs == idx)]
                continue
            else:
                curr_test_halo = test_dataset[np.where(test_halo_idxs == idx)]
                test_halos_within = np.row_stack((test_halos_within, curr_test_halo))

        test_predict = np.ones(test_halos_within.shape[0]) * -1

        for i in range(len(snapshot_list)):
            curr_dataset = test_halos_within[:,:int(2 + (num_params_per_snap * (i+1)))]

            if i == (len(snapshot_list)-1):
                use_ptls = np.where(test_halos_within[:,-1]!= 0)[0]
            else:
                use_ptls = np.where(test_halos_within[:,int(2 + (num_params_per_snap * (i+1)))] == 0)[0]

            curr_dataset = curr_dataset[use_ptls]
            all_models[i].predict(curr_dataset)
            test_predict[use_ptls] = all_models[i].get_predicts()

        actual_labels = test_halos_within[:,1]

        classification = classification_report(actual_labels, test_predict, output_dict=True)

        all_accuracy.append(classification["accuracy"])
        compare_density_prf(test_halos_within[:,2+scaled_radii_loc], density_prf_all_within, density_prf_1halo_within, mass, test_predict, title = str(start_nu) + "-" + str(end_nu), show_graph = True, save_graph = True, save_location = save_location)
        plot_radius_rad_vel_tang_vel_graphs(test_predict, test_halos_within[:,2+scaled_radii_loc], test_halos_within[:,2+rad_vel_loc], test_halos_within[:,2+tang_vel_loc], actual_labels, "ML Predictions", num_bins, start_nu, end_nu, show = False, save = True, save_location=save_location)
        #graph_acc_by_bin(test_predict, actual_labels, test_halos_within[:,2+scaled_radii_loc], num_bins, start_nu, end_nu, plot = False, save = True, save_location = save_location)
    
    start_nu = end_nu
    
print(all_accuracy)
fig, ax = plt.subplots(1,1)
ax.set_title("Number of halos at each accuracy level")
ax.set_xlabel("Accuracy")
ax.set_ylabel("Num halos")
ax.hist(all_accuracy)
fig.savefig(save_location + "all_test_halo_acc.png")