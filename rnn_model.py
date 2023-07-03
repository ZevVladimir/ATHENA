import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_and_loading_functions import standardize, build_ml_dataset

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

for i in range(num_splits):
    t1 = time.time()
    print("Split:", (i+1), "/",num_splits)
    
    dataset = build_ml_dataset(save_location, data_location, i, random_indices, use_num_particles, curr_snapshot, param_list)

    t2 = time.time()
    print("Loaded data", t2 - t1, "seconds")

    X_train, X_test, y_train, y_test = train_test_split(dataset[:,1:], dataset[:,0], test_size=0.30, random_state=0)
