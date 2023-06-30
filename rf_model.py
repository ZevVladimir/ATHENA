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

curr_snapshot = "190"
curr_hdf5_file = "sparta_190.hdf5"

data_location = "/home/zvladimi/MLOIS/calculated_info/" + "calc_from_" + curr_hdf5_file + "/"
save_location = "/home/zvladimi/MLOIS/training_data/" + "data_for_" + curr_hdf5_file + "/"

np.random.seed(11)


def check_pickle_exist_hdf5_prop(path, first_group, hdf5, random_indices):    
    file_path = path + first_group + ".pickle" 
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            particle_info = pickle.load(pickle_file)
    else:
        particle_info = hdf5[first_group][:]
        with open(file_path, "wb") as pickle_file:
            pickle.dump(particle_info, pickle_file)
    
    particle_info = particle_info[random_indices]
    return particle_info

def load_or_pickle_data(path, curr_split, indices, use_num_particles, snapshot):
    if os.path.exists(path) != True:
        os.makedirs(path)
        
    with h5py.File((data_location + "all_particle_properties" + snapshot + ".hdf5"), 'r') as all_particle_properties:
        start_index = curr_split * use_num_particles
        use_random_indices = indices[start_index:start_index+use_num_particles]

        orbit_infall = check_pickle_exist_hdf5_prop(path, "Orbit_Infall", all_particle_properties, use_random_indices)
        #ptl_pids = check_pickle_exist_hdf5_prop(path, "PIDS", all_particle_properties, use_random_indices)
        scaled_radii = check_pickle_exist_hdf5_prop(path, "Scaled_radii", all_particle_properties, use_random_indices)
        radial_vel = check_pickle_exist_hdf5_prop(path, "Radial_vel", all_particle_properties, use_random_indices)
        tang_vel = check_pickle_exist_hdf5_prop(path, "Tangential_vel", all_particle_properties, use_random_indices)

    return orbit_infall, scaled_radii, radial_vel, tang_vel

num_splits = 20
with h5py.File((data_location + "all_particle_properties" + curr_snapshot + ".hdf5"), 'r') as all_particle_properties:
    total_num_particles = all_particle_properties["PIDS"][:].shape[0]    
    random_indices = np.random.choice(total_num_particles, total_num_particles)
    use_num_particles = int(np.floor(total_num_particles/num_splits))
    
rus = under_sampling.RandomUnderSampler(random_state=0)
t0 = time.time()
for i in range(1):
    t1 = time.time()
    print("Split:", (i + 1), "/",num_splits)
    
    orbit_infall, scaled_radii, radial_vel, tang_vel = load_or_pickle_data(save_location, i, random_indices, use_num_particles, curr_snapshot)

    features = np.zeros((scaled_radii.size, 7))
    features[:,0] = scaled_radii
    features[:,1:4] = radial_vel
    features[:,4:] = tang_vel

    X_train, X_test, y_train, y_test = train_test_split(features, orbit_infall, test_size=0.30, random_state=0)
    
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
