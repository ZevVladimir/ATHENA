##################################################################################################################
# CONFIG PARAMETERS

# Loading params
curr_sparta_file = "sparta_cbol_l0063_n0256"
path_to_hdf5_file = "/home/zvladimi/MLOIS/SPARTA_data" + curr_sparta_file + ".hdf5"

path_dict = {
    "curr_sparta_file": curr_sparta_file,
    "path_to_MLOIS": "/home/zvladimi/MLOIS/",
    "path_to_snaps": "/home/zvladimi/MLOIS/particle_data/",
    "path_to_hdf5_file": path_to_hdf5_file,
    "path_to_pickle": "/home/zvladimi/MLOIS/pickle_data/"
}

snap_format = "{:04d}" # how are the snapshots formatted with 0s

# Search params
prim_only = False
t_dyn_step = 0.5
p_snap = 190
times_r200m = 6
total_num_snaps = 193
num_halo_per_split = 2500
num_print_per_split = 4
test_halos_ratio = 0.25

# Save information
num_save_ptl_params = 5 # don't change unless editing code
save_location = path_dict["path_to_MLOIS"] + "calculated_info/"
##################################################################################################################
import sys
sys.path.insert(0, path_dict["path_to_MLOIS"] + "pygadgetreader")
sys.path.insert(0, path_dict["path_to_MLOIS"] + "sparta/analysis")
from pygadgetreader import readsnap, readheader
from sparta import sparta
import numexpr as ne
import numpy as np
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import peaks
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time
import h5py
import pickle
import os
import multiprocessing as mp
import logging
import ctypes
from contextlib import closing
from data_and_loading_functions import load_or_pickle_SPARTA_data, load_or_pickle_ptl_data, save_to_hdf5, find_closest_snap, conv_halo_id_spid, get_comp_snap
from visualization_functions import compare_density_prf
from calculation_functions import *
##################################################################################################################

def to_np_arr(mp_arr):
    return np.frombuffer(mp_arr.get_obj())
def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_ # must be inherited, not passed as an argument

def search_halos(ptl_tree, sparta_output, scale_factor,):
    
    curr_save_idx = 0

def halo_loop(indices, p_halo_ids, p_snap, p_scale_factor, p_num_ptls_per_halo):
    num_iter = int(np.ceil(indices.shape[0] / num_halo_per_split))

    for i in range(num_iter):
        logger = mp.log_to_stderr()
        logger.setLevel(logging.INFO)


        if i < (num_iter - 1):
            use_indices = indices[i * num_halo_per_split: (i+1) * num_halo_per_split]
        else:
            use_indices = indices[i * num_halo_per_split:]

        use_halo_ids = p_halo_ids[use_indices]
        sparta_output = sparta.load(file_name = path_dict["path_to_hdf5_file"], halo_ids=use_halo_ids)
        new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_snap)
        use_halo_idxs = use_halo_idxs[new_idxs]
        
        p_use_num_ptls = p_num_ptls_per_halo[use_halo_idxs]
        p_tot_num_use_ptls = np.sum(p_use_num_ptls)

        halos_pos = sparta_output['halos']['position'][:,p_snap,:] * 10**3 * p_scale_factor * little_h
        halos_vel = sparta_output['halos']['velocity'][:,p_snap,:]
        halos_r200m = sparta_output['halos']['R200m'][:,p_snap] * little_h
        dens_prf_all = sparta_output['anl_prf']['M_all'][:,p_snap,:]
        dens_prf_1halo = sparta_output['anl_prf']['M_1halo'][:,p_snap,:]

        share_calc_rad_vel = mp.Array(ctypes.c_float, p_tot_num_use_ptls)
        share_calc_tang_vel  = mp.Array(ctypes.c_float, p_tot_num_use_ptls)
        share_all_scal_radii = mp.Array(ctypes.c_float, p_tot_num_use_ptls)
        share_all_orb_assn = mp.Array(ctypes.c_long, (p_tot_num_use_ptls,2))
        calc_rad_vel = to_np_arr(share_calc_rad_vel)
        calc_tang_vel = to_np_arr(share_calc_tang_vel)
        all_scal_radii = to_np_arr(share_all_scal_radii)
        all_orb_assn = to_np_arr(share_all_orb_assn)

        with closing(mp.Pool(processes=20, initializer=init, initargs=(share_calc_rad_vel,share_calc_tang_vel,share_all_scal_radii,share_all_orb_assn))) as p:
            p.map_async(search_halos, [slice(np.sum(p_use_num_ptls[:i]), np.sum(p_use_num_ptls[:i+1])) for i in range(num_halo_per_split)])

# Set constants
cosmol = cosmology.setCosmology("bolshoi")
little_h = cosmol.h 
p_snapshot_path = path_dict["path_to_snaps"] + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)
p_red_shift = readheader(p_snapshot_path, 'redshift')
p_scale_factor = 1/(1+p_red_shift)
p_rho_m = cosmol.rho_m(p_red_shift)
p_hubble_constant = cosmol.Hz(p_red_shift) * 0.001 # convert to units km/s/kpc
p_box_size = readheader(p_snapshot_path, 'boxsize') #units Mpc/h comoving
p_box_size = p_box_size * 10**3 * p_scale_factor * little_h #convert to Kpc physical

p_snapshot_path = + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

# load all information needed for the primary snap
p_particles_pid, p_particles_vel, p_particles_pos, mass = load_or_pickle_ptl_data(str(p_snap), p_snapshot_path, p_scale_factor, little_h, path_dict["path_to_pickle"])
p_halos_pos, p_halos_r200m, p_halos_id, p_halos_status, p_halos_last_snap = load_or_pickle_SPARTA_data(curr_sparta_file, path_dict["path_to_hdf5_file"], p_scale_factor, little_h, p_snap, path_dict["path_to_pickle"])
p_particle_tree = cKDTree(data = p_particles_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size) # construct search trees for primary snap

p_num_particles_per_halo, p_halo_masses, t_dyn = initial_search(p_halos_pos, times_r200m, p_halos_r200m, p_particle_tree, p_red_shift)

c_snap, c_box_size, c_rho_m, c_red_shift, c_hubble_constant, c_particles_pid, c_particles_vel, c_particles_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap = get_comp_snap(t_dyn=t_dyn, t_dyn_step=t_dyn_step, snapshot_list=[p_snap], cosmol = cosmol, p_red_shift=p_red_shift, total_num_snaps=total_num_snaps, path_dict=path_dict, snap_format=snap_format,little_h=little_h)
snapshot_list = [p_snap, c_snap]

if prim_only:
    save_location =  path_dict["path_to_MLOIS"] +  "calculated_info/" + curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(times_r200m) + "r200msearch/"
else:
    save_location =  path_dict["path_to_MLOIS"] + "calculated_info/" + curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[1]) + "_" + str(times_r200m) + "r200msearch/"

if os.path.exists(save_location) != True:
    os.makedirs(save_location)

# only take halos that are hosts in primary snap and exist past the p_snap and exist in some form at the comparison snap
if prim_only:
    match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_snap))[0]
else:
    match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_snap) & (c_halos_status > 0) & (c_halos_last_snap >= c_snap))[0]
    
rng = np.random.default_rng(11)    
total_num_halos = match_halo_idxs.shape[0]
rng.shuffle(match_halo_idxs)
# split all indices into train and test groups
train_indices, test_indices = np.split(match_halo_idxs, [int((1-test_halos_ratio) * total_num_halos)])
# need to sort indices otherwise sparta.load breaks...
train_indices = np.sort(train_indices)
test_indices = np.sort(test_indices)

with open(save_location + "test_indices.pickle", "wb") as pickle_file:
    pickle.dump(test_indices, pickle_file)
with open(save_location + "train_indices.pickle", "wb") as pickle_file:
    pickle.dump(train_indices, pickle_file)

print("Total num halos:", total_num_halos)
print("Num train halos:", train_indices.shape[0])
print("Num test halos:", test_indices.shape[0])


