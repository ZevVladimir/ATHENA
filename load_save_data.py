import numpy as np
import matplotlib.pyplot as plt
import h5py
from pygadgetreader import *


original_data_file_location = "/home/zeevvladimir/ML_orbit_infall_project/Original_Data/sparta_cbol_l0063_n0256_strd1_v2.hdf5"
save_location = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"

snapshot = "/home/zeevvladimir/ML_orbit_infall_project/Original_Data/snapshot_0190."

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
    #save halo R200, velocity, position as .np files
    halo_R200m = np.empty(f["/halos/R200m"].shape, f["/halos/R200m"].dtype)
    halo_R200m = np.array(list(f["/halos/R200m"]))
    np.save(save_location + "halo_R200m", halo_R200m)

    halo_velocity = np.empty(f["/halos/velocity"].shape, f["/halos/velocity"].dtype)
    halo_velocity = np.array(list(f["/halos/velocity"]))
    np.save(save_location + "halo_velocity", halo_velocity)

    #save info from res_oct
    halo_position = np.empty(f["/halos/position"].shape, f["/halos/position"].dtype)
    halo_position = np.array(list(f["/halos/position"]))
    np.save(save_location + "halo_position", halo_position)

    halo_first = np.empty(f["/tcr_ptl/res_oct/halo_first"].shape, f["/tcr_ptl/res_oct/halo_first"].dtype)
    halo_first = np.array(list(f["/tcr_ptl/res_oct/halo_first"]))
    np.save(save_location + "halo_first", halo_first)

    halo_n = np.empty(f["/tcr_ptl/res_oct/halo_n"].shape, f["/tcr_ptl/res_oct/halo_n"].dtype)
    halo_n = np.array(list(f["/tcr_ptl/res_oct/halo_n"]))
    np.save(save_location + "halo_n", halo_n)

    last_pericenter_snap = np.empty(f["/tcr_ptl/res_oct/last_pericenter_snap"].shape, f["/tcr_ptl/res_oct/last_pericenter_snap"].dtype)
    last_pericenter_snap = np.array(list(f["/tcr_ptl/res_oct/last_pericenter_snap"]))
    np.save(save_location + "last_pericenter_snap", last_pericenter_snap)
    
    n_is_lower_limit = np.empty(f["/tcr_ptl/res_oct/n_is_lower_limit"].shape, f["/tcr_ptl/res_oct/n_is_lower_limit"].dtype)
    n_is_lower_limit = np.array(list(f["/tcr_ptl/res_oct/n_is_lower_limit"]))
    np.save(save_location + "n_is_lower_limit", n_is_lower_limit)
    
    n_pericenter = np.empty(f["/tcr_ptl/res_oct/n_pericenter"].shape, f["/tcr_ptl/res_oct/n_pericenter"].dtype)
    n_pericenter = np.array(list(f["/tcr_ptl/res_oct/n_pericenter"]))
    np.save(save_location + "n_pericenter", n_pericenter)

    tracer_id = np.empty(f["/tcr_ptl/res_oct/tracer_id"].shape, f["/tcr_ptl/res_oct/tracer_id"].dtype)
    tracer_id = np.array(list(f["/ tcr_ptl/res_oct/tracer_id"]))
    np.save(save_location + "tracer_id", tracer_id)

count = 0

#save info from ll snapshots
while(count < 16):
    particles_pos = readsnap(snapshot + str(count), 'pos', 'dm')
    np.save(save_location + "particles_pos_" + str(count), particles_pos)
    particles_vel = readsnap(snapshot + str(count), 'vel', 'dm')
    np.save(save_location + "particles_vel_" + str(count), particles_vel)
    particles_pid = readsnap(snapshot + str(count), 'pid', 'dm')
    np.save(save_location + "particles_pid_" + str(count), particles_pid)
    count+=1

