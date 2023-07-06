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
from data_and_loading_functions import standardize, build_ml_dataset

curr_snapshot = "190"
curr_hdf5_file = "sparta_190.hdf5"

data_location = "/home/zvladimi/MLOIS/calculated_info/" + "calc_from_" + curr_hdf5_file + "/"
save_location = "/home/zvladimi/MLOIS/training_data/" + "data_for_" + curr_hdf5_file + "/"

np.random.seed(11)

num_splits = 10
param_list = ['Orbit_Infall', 'Scaled_radii', 'Radial_vel', 'Tangential_vel']
with h5py.File((data_location + "all_particle_properties" + curr_snapshot + ".hdf5"), 'r') as all_particle_properties:
    total_num_particles = all_particle_properties["PIDS"][:].shape[0]    
    random_indices = np.random.choice(total_num_particles, total_num_particles)
    use_num_particles = int(np.floor(total_num_particles/num_splits))
    
model=None

def choose_class(predictions):
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    return predictions

rus = under_sampling.RandomUnderSampler(random_state=0)
ros = over_sampling.RandomOverSampler(random_state=0)
t0 = time.time()
for i in range(num_splits):
    t1 = time.time()
    print("Split:", (i+1), "/",num_splits)
    
    dataset = build_ml_dataset(save_location, data_location, i, random_indices, use_num_particles, curr_snapshot, param_list)

    t2 = time.time()
    print("Loaded data", t2 - t1, "seconds")

    X_train, X_test, y_train, y_test = train_test_split(dataset[:,1:], dataset[:,0], test_size=0.30, random_state=0)

    # standardize has slightly better performance comapared to normalize
    # X_train = standardize(X_train)
    # X_test = standardize(X_test)
    
    #X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    #X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    
    t3 = time.time()
    
    model = XGBClassifier(tree_method='gpu_hist')
    model = model.fit(X_train, y_train)


    predicts = model.predict(X_test)

    t4 = time.time()
    print("Fitted model", t4 - t3, "seconds")

    predicts = choose_class((predicts))

    classification = classification_report(y_test, predicts)
    print(classification)

model.save_model(save_location + "xgb_model" + curr_snapshot + ".model")
t5 = time.time()
print("Total time:", t5-t0, "seconds")
