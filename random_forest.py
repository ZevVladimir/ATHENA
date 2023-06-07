import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
import seaborn as sns

save_location = "/home/zvladimi/ML_orbit_infall_project/calculated_info/"
#halo properties: scaled radius, scaled radial vel x, scaled radial vel y, scaled radial vel z, scaled tangential vel x, scaled tangential vel y, scaled tangential vel z

halo_0_5_to_1 = pd.DataFrame(np.load(save_location + "all_part_prop_0.5_to_1.0.npy"), columns= ["radius", "rad_vel_x", "rad_vel_y", "rad_vel_z", "rad_tang_x", "rad_tang_y", "rad_tang_z"])
halo_1_to_1_5 = pd.DataFrame(np.load(save_location + "all_part_prop_1.0_to_1.5.npy"), columns= ["radius", "rad_vel_x", "rad_vel_y", "rad_vel_z", "rad_tang_x", "rad_tang_y", "rad_tang_z"])
halo_1_5_to_2 = pd.DataFrame(np.load(save_location + "all_part_prop_1.5_to_2.0.npy"), columns= ["radius", "rad_vel_x", "rad_vel_y", "rad_vel_z", "rad_tang_x", "rad_tang_y", "rad_tang_z"])
halo_2_to_2_5 = pd.DataFrame(np.load(save_location + "all_part_prop_2.0_to_2.5.npy"), columns= ["radius", "rad_vel_x", "rad_vel_y", "rad_vel_z", "rad_tang_x", "rad_tang_y", "rad_tang_z"])
halo_2_5_to_3 = pd.DataFrame(np.load(save_location + "all_part_prop_2.5_to_3.0.npy"), columns= ["radius", "rad_vel_x", "rad_vel_y", "rad_vel_z", "rad_tang_x", "rad_tang_y", "rad_tang_z"])
halo_3_to_3_5 = pd.DataFrame(np.load(save_location + "all_part_prop_3.0_to_3.5.npy"), columns= ["radius", "rad_vel_x", "rad_vel_y", "rad_vel_z", "rad_tang_x", "rad_tang_y", "rad_tang_z"])

fig, axes = plt.subplots(2,3)
axes[0,0].set_title("halo_0_5_to_1")
axes[0,1].set_title("halo_1_to_1_5")
axes[0,2].set_title("halo_1_5_to_2")
axes[1,0].set_title("halo_2_to_2_5")
axes[1,1].set_title("halo_2_5_to_3")
axes[1,2].set_title("halo_3_to_3_5")

sns.heatmap(halo_0_5_to_1.corr(), xticklabels = halo_0_5_to_1.columns, yticklabels = halo_0_5_to_1.columns, ax = axes[0,0])
sns.heatmap(halo_1_to_1_5.corr(), xticklabels = halo_0_5_to_1.columns, yticklabels = halo_0_5_to_1.columns, ax = axes[0,1])
sns.heatmap(halo_1_5_to_2.corr(), xticklabels = halo_0_5_to_1.columns, yticklabels = halo_0_5_to_1.columns, ax = axes[0,2])
sns.heatmap(halo_2_to_2_5.corr(), xticklabels = halo_0_5_to_1.columns, yticklabels = halo_0_5_to_1.columns, ax = axes[1,0])
sns.heatmap(halo_2_5_to_3.corr(), xticklabels = halo_0_5_to_1.columns, yticklabels = halo_0_5_to_1.columns, ax = axes[1,1])
sns.heatmap(halo_3_to_3_5.corr(), xticklabels = halo_0_5_to_1.columns, yticklabels = halo_0_5_to_1.columns, ax = axes[1,2])
print(halo_0_5_to_1.shape[0] + halo_1_5_to_2.shape[0] + halo_1_to_1_5.shape[0] + halo_2_5_to_3.shape[0] + halo_3_to_3_5.shape[0] + halo_2_to_2_5.shape[0])
plt.show()