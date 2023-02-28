import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from pygadgetreader import *


original_data_file_location = "/home/zeevvladimir/ML_orbit_infall_project/Original_Data/sparta_cbol_l0063_n0256_strd1_v2.hdf5"
save_location = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"

snapshot_path = "/home/zeevvladimir/ML_orbit_infall_project/Original_Data/snapshot_190/snapshot_0190"

array = np.empty(0)

original_data_file = h5py.File(original_data_file_location)

# def allkeys(obj):
#     "Recursively find all keys in an h5py.Group."
#     keys = (obj.name,)
#     if isinstance(obj, h5py.Group):
#         for key, value in obj.items():
#             if isinstance(value, h5py.Group):
#                 keys = keys + allkeys(value)
#             else:
#                 keys = keys + (value.name,)
#     return keys

# print(allkeys(original_data_file))


with h5py.File(original_data_file_location, "r") as f: 
    # halo_last_snap = np.empty(f["/halos/last_snap"].shape, f["/halos/last_snap"].dtype)
    # halo_last_snap = np.array(list(f["/halos/last_snap"]))
    # np.save(save_location + "halo_last_snap", halo_last_snap)

#     #save halo R200, velocity, position, halo_id as .np files
#     halo_R200m = np.empty(f["/halos/R200m"].shape, f["/halos/R200m"].dtype)
#     halo_R200m = np.array(list(f["/halos/R200m"]))
#     np.save(save_location + "halo_R200m", halo_R200m)

#     halo_velocity = np.empty(f["/halos/velocity"].shape, f["/halos/velocity"].dtype)
#     halo_velocity = np.array(list(f["/halos/velocity"]))
#     np.save(save_location + "halo_velocity", halo_velocity)

#     
#     halo_position = np.empty(f["/halos/position"].shape, f["/halos/position"].dtype)
#     halo_position = np.array(list(f["/halos/position"]))
#     np.save(save_location + "halo_position", halo_position)

    # halo_id = np.empty(f["/halos/id"].shape, f["/halos/id"].dtype)
    # halo_id = np.array(list(f["/halos/id"]))
    # np.save(save_location + "halo_id", halo_id)

# save info from res_ifl

    halo_first = np.empty(f["/tcr_ptl/res_ifl/halo_first"].shape, f["/tcr_ptl/res_ifl/halo_first"].dtype)
    halo_first = np.array(list(f["/tcr_ptl/res_ifl/halo_first"]))
    np.save(save_location + "halo_first", halo_first)

    halo_n = np.empty(f["/tcr_ptl/res_ifl/halo_n"].shape, f["/tcr_ptl/res_ifl/halo_n"].dtype)
    halo_n = np.array(list(f["/tcr_ptl/res_ifl/halo_n"]))
    np.save(save_location + "halo_n", halo_n)



#save info from all snapshots

# particles_pos = readsnap(snapshot_path, 'pos', 'dm')
# np.save(save_location + "particle_pos", particles_pos)

# particles_vel = readsnap(snapshot_path, 'vel', 'dm')
# np.save(save_location + "particle_vel", particles_vel)

# particles_pid = readsnap(snapshot_path, 'pid', 'dm')
# np.save(save_location + "particle_pid", particles_pid)

# particle_mass = readsnap(snapshot_path, 'mass', 'dm')
# np.save(save_location + "all_particle_mass", particle_mass)




