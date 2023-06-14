import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from pygadgetreader import readsnap


original_data_file_location = "/home/zvladimi/ML_orbit_infall_project/SPARTA_data/sparta_cbol_l0063_n0256_strd1_v2.hdf5"
orbit_count_hdf5 = "/home/zvladimi/ML_orbit_infall_project/SPARTA_data/sparta_190.hdf5"
save_location = "/home/zvladimi/ML_orbit_infall_project/calculated_info/"

snapshot_path = "/home/zvladimi/ML_orbit_infall_project/particle_data/snapshot_192/snapshot_0192"

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


# with h5py.File(original_data_file_location, "r") as f: 
#     halo_last_snap = np.empty(f["/halos/last_snap"].shape, f["/halos/last_snap"].dtype)
#     halo_last_snap = np.array(list(f["/halos/last_snap"]))
#     np.save(save_location + "halo_last_snap", halo_last_snap)

#     #save halo R200, velocity, position, halo_id as .np files
#     halo_R200m = np.empty(f["/halos/R200m"].shape, f["/halos/R200m"].dtype)
#     halo_R200m = np.array(list(f["/halos/R200m"]))
#     np.save(save_location + "halo_R200m", halo_R200m)

#     halo_velocity = np.empty(f["/halos/velocity"].shape, f["/halos/velocity"].dtype)
#     halo_velocity = np.array(list(f["/halos/velocity"]))
#     np.save(save_location + "halo_velocity", halo_velocity)

    
#     halo_position = np.empty(f["/halos/position"].shape, f["/halos/position"].dtype)
#     halo_position = np.array(list(f["/halos/position"]))
#     np.save(save_location + "halo_position", halo_position)

#     halo_id = np.empty(f["/halos/id"].shape, f["/halos/id"].dtype)
#     halo_id = np.array(list(f["/halos/id"]))
#     np.save(save_location + "halo_id", halo_id)
#     halo_status = np.empty(f["/halos/status"].shape, f["/halos/status"].dtype)
#     halo_status = np.array(list(f["/halos/status"]))
#     np.save(save_location + "halo_status", halo_status)
# # save info from res_ifl

#     halo_first = np.empty(f["/tcr_ptl/res_ifl/halo_first"].shape, f["/tcr_ptl/res_ifl/halo_first"].dtype)
#     halo_first = np.array(list(f["/tcr_ptl/res_ifl/halo_first"]))
#     np.save(save_location + "halo_first", halo_first)

#     halo_n = np.empty(f["/tcr_ptl/res_ifl/halo_n"].shape, f["/tcr_ptl/res_ifl/halo_n"].dtype)
#     halo_n = np.array(list(f["/tcr_ptl/res_ifl/halo_n"]))
#     np.save(save_location + "halo_n", halo_n)
    

with h5py.File(orbit_count_hdf5, "r") as f: 
    num_pericenters = f['tcr_ptl']['res_oct']['n_pericenter'][:]
    num_pericenters[num_pericenters > 0] = 1

    pids = f['tcr_ptl']['res_oct']['tracer_id'][:]
    orbit_ptl_pid = np.column_stack((pids, num_pericenters))
    print(orbit_ptl_pid)
    np.save(save_location + "orbit_ptl_pid", orbit_ptl_pid)




#save info from all snapshots

# particles_pos = readsnap(snapshot_path, 'pos', 'dm')
# np.save(save_location + "particle_pos_192", particles_pos)

# particles_vel = readsnap(snapshot_path, 'vel', 'dm')
# np.save(save_location + "particle_vel_192", particles_vel)

# particles_pid = readsnap(snapshot_path, 'pid', 'dm')
# np.save(save_location + "particle_pid_192", particles_pid)

# particle_mass = readsnap(snapshot_path, 'mass', 'dm')
# np.save(save_location + "all_particle_mass_192", particle_mass)

