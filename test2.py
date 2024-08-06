from utils.ML_support import print_model_prop,load_data
import numpy as np
from utils.visualization_functions import plot_log_vel
import pandas as pd
import os
import h5py

arr = [0,1,2,3,4,5]
print(arr[:-1])
print(arr[:0])

# folder_path = "/home/zvladimi/MLOIS/calculated_info/cbol_l0063_n0256_4r200m_1-5v200m_190to166/Train/ptl_info/"

# all_dfs = []

# for file in os.listdir(folder_path):
#     all_dfs.append(pd.read_hdf(folder_path + file))
    
    
# ptl_df = pd.concat(all_dfs)

# plot_log_vel(ptl_df["p_phys_vel"].values,ptl_df["p_Scaled_radii"].values,ptl_df["Orbit_infall"].values,"/home/zvladimi/MLOIS/Random_figs/",1.5)
