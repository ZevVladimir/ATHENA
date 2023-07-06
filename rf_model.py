import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
from sklearn.ensemble import RandomForestClassifier
import time 
import pickle
import os
import multiprocessing as mp
from imblearn import under_sampling
from data_and_loading_functions import build_ml_dataset

curr_snapshot = "190"
curr_hdf5_file = "sparta_190.hdf5"

data_location = "/home/zvladimi/MLOIS/calculated_info/" + "calc_from_" + curr_hdf5_file + "/"
save_location = "/home/zvladimi/MLOIS/training_data/" + "data_for_" + curr_hdf5_file + "/"

np.random.seed(11)

num_splits = 20
param_list = ['Orbit_Infall', 'Scaled_radii', 'Radial_vel', 'Tangential_vel']
with h5py.File((data_location + "all_particle_properties" + curr_snapshot + ".hdf5"), 'r') as all_particle_properties:
    total_num_particles = all_particle_properties["PIDS"][:].shape[0]    
    random_indices = np.random.choice(total_num_particles, total_num_particles)
    use_num_particles = int(np.floor(total_num_particles/num_splits))
    
rus = under_sampling.RandomUnderSampler(random_state=0)
t0 = time.time()
for i in range(1):
    t1 = time.time()
    print("Split:", (i + 1), "/",num_splits)
    
    dataset = build_ml_dataset(save_location, data_location, i, random_indices, use_num_particles, curr_snapshot, param_list)

    X_train, X_test, y_train, y_test = train_test_split(dataset[:,1:], dataset[:,0], test_size=0.30, random_state=0)
    
    total_num_particles = y_train.size
    print(f'{total_num_particles:,}')
    print(f'{np.where(y_train == 0)[0].size:,}')
    print(f'{np.where(y_train == 1)[0].size:,}')
    print(np.where(y_train == 0)[0].size / total_num_particles)
    print(np.where(y_train == 1)[0].size / total_num_particles)
    
    X_train_rs, y_train_rs = rus.fit_resample(X_train, y_train)
    
    total_num_particles = y_train_rs.size
    print(f'{total_num_particles:,}')
    print(f'{np.where(y_train_rs == 0)[0].size:,}')
    print(f'{np.where(y_train_rs == 1)[0].size:,}')
    print(np.where(y_train_rs == 0)[0].size / total_num_particles)
    print(np.where(y_train_rs == 1)[0].size / total_num_particles)
    
    model_params = {
        'n_estimators': 25,
        'max_depth': 5,
        'n_jobs': 6,
    }
    
    t2 = time.time()
    print("Loaded data", t2 - t1, "seconds")
    
    t3 = time.time()
    
    model = RandomForestClassifier(**model_params)
    model.fit(X_train_rs, y_train_rs)

    t4 = time.time()
    print("Fitted model", t4 - t3, "seconds")
    
    predicts = model.predict(X_test)

    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

# model.save_model(save_location + "rnn_model" + curr_snapshot + ".model")
# t5 = time.time()
# print("Total time:", t5-t0, "seconds")
