import numexpr as ne
import numpy as np
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import peaks
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import time
import h5py
import pickle
import os
from multiprocessing import shared_memory
import multiprocessing as mp
import logging
import ctypes
from itertools import repeat
from contextlib import closing
import pandas as pd
from data_and_loading_functions import load_or_pickle_SPARTA_data, load_or_pickle_ptl_data, save_to_hdf5, conv_halo_id_spid, get_comp_snap, create_directory
from visualization_functions import compare_density_prf
from calculation_functions import *
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
rand_seed = config.getint("MISC","random_seed")
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
path_to_hdf5_file = path_to_SPARTA_data + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
create_directory(path_to_MLOIS)
create_directory(path_to_snaps)
create_directory(path_to_SPARTA_data)
create_directory(path_to_hdf5_file)
create_directory(path_to_pickle)
create_directory(path_to_calc_info)
snap_format = config["MISC"]["snap_format"]
global prim_only
prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
global p_snap
p_snap = config.getint("SEARCH","p_snap")
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
num_processes = mp.cpu_count()
curr_chunk_size = config.getint("SEARCH","chunk_size")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
##################################################################################################################
import sys
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta import sparta
##################################################################################################################

def to_np_arr(mp_arr, dtype):
    return np.frombuffer(mp_arr.get_obj(), dtype = dtype)

def initial_search(halo_positions, halo_r200m, comp_snap, find_mass = False, find_ptl_indices = False):
    global search_rad
    if comp_snap:
        tree = c_ptl_tree
    else:
        tree = p_ptl_tree
    
    if halo_r200m > 0:
        #find how many particles we are finding
        indices = tree.query_ball_point(halo_positions, r = search_rad * halo_r200m)
        indices = np.array(indices)
        # how many new particles being added and correspondingly how massive the halo is
        num_new_particles = indices.shape[0]

        if find_mass:
            halo_mass = num_new_particles * mass
        if find_ptl_indices:
            all_ptl_indices = indices
    else:
        num_new_particles = 0
    
    if find_mass == False and find_ptl_indices == False:
        return num_new_particles
    elif find_mass == True and find_ptl_indices == False:
        return num_new_particles, halo_mass
    elif find_mass == False and find_ptl_indices == True:
        return num_new_particles, all_ptl_indices
    else:
        return num_new_particles, halo_mass, all_ptl_indices
    
def search_halos(halo_slice, comp_snap, curr_halo_idx, curr_sparta_idx):
    # Doing this this way as otherwise will have to generate super large arrays for input from multiprocessing
    if comp_snap:
        global c_snap
        global c_red_shift
        global c_scale_factor
        global c_hubble_constant
        global c_box_size
        global c_ptls_pid
        global c_ptls_pos
        global c_ptls_vel
        global c_curr_ptl_indices
        snap = c_snap
        red_shift = c_red_shift
        scale_factor = c_scale_factor
        hubble_const = c_hubble_constant
        box_size = c_box_size
        ptl_pids = c_ptls_pid
        ptl_pos = c_ptls_pos
        ptl_vel = c_ptls_vel
        ptl_indices = c_curr_ptl_indices[curr_sparta_idx]
    else:
        global p_snap
        global p_red_shift
        global p_scale_factor
        global p_hubble_constant
        global p_box_size
        global p_ptls_pid
        global p_ptls_pos
        global p_ptls_vel
        global p_curr_ptl_indices
        snap = p_snap
        red_shift = p_red_shift
        scale_factor = p_scale_factor
        hubble_const = p_hubble_constant
        box_size = p_box_size
        ptl_pids = p_ptls_pid
        ptl_pos = p_ptls_pos
        ptl_vel = p_ptls_vel
        ptl_indices = p_curr_ptl_indices[curr_sparta_idx]

    # get all the information for this specific halo
    halo_pos = sparta_output['halos']['position'][curr_sparta_idx,snap,:] * 10**3 * scale_factor
    halo_vel = sparta_output['halos']['velocity'][curr_sparta_idx,snap,:]
    halo_r200m = sparta_output['halos']['R200m'][curr_sparta_idx,snap]
    dens_prf_all = sparta_output['anl_prf']['M_all'][curr_sparta_idx,snap,:]
    dens_prf_1halo = sparta_output['anl_prf']['M_1halo'][curr_sparta_idx,snap,:]
    prf_status = sparta_output['anl_prf']['status'][curr_sparta_idx,snap]

    num_new_ptls = ptl_indices.shape[0]
    curr_ptl_pids = ptl_pids[ptl_indices]
    curr_ptl_pos = ptl_pos[ptl_indices]
    curr_ptl_vel = ptl_vel[ptl_indices]

    curr_ptl_pids = curr_ptl_pids.astype(np.int64) # otherwise ne.evaluate doesn't work
    fnd_HIPIDs = ne.evaluate("0.5 * (curr_ptl_pids + curr_halo_idx) * (curr_ptl_pids + curr_halo_idx + 1) + curr_halo_idx")
    
    #calculate the radii of each particle based on the distance formula
    ptl_rad, coord_dist = calculate_distance(halo_pos[0], halo_pos[1], halo_pos[2], curr_ptl_pos[:,0], curr_ptl_pos[:,1], curr_ptl_pos[:,2], num_new_ptls, box_size)         
    
    if comp_snap == False:
        # Get the range of indices for the SPARTA particles for this halo
        curr_halo_first = sparta_output['halos']['ptl_oct_first'][curr_sparta_idx]
        curr_halo_n = sparta_output['halos']['ptl_oct_n'][curr_sparta_idx]
        
        # Get the particle information for this halo
        sparta_last_pericenter_snap = sparta_output['tcr_ptl']['res_oct']['last_pericenter_snap'][curr_halo_first:curr_halo_first+curr_halo_n]
        sparta_n_pericenter = sparta_output['tcr_ptl']['res_oct']['n_pericenter'][curr_halo_first:curr_halo_first+curr_halo_n]
        sparta_tracer_ids = sparta_output['tcr_ptl']['res_oct']['tracer_id'][curr_halo_first:curr_halo_first+curr_halo_n]
        sparta_n_is_lower_limit = sparta_output['tcr_ptl']['res_oct']['n_is_lower_limit'][curr_halo_first:curr_halo_first+curr_halo_n]
        
        compare_sparta_assn = np.zeros((sparta_tracer_ids.shape[0]))
        curr_orb_assn = np.zeros((num_new_ptls))
         # Anywhere sparta_last_pericenter is greater than the current snap then that is in the future so set to 0
        future_peri = np.where(sparta_last_pericenter_snap > snap)[0]
        adj_sparta_n_pericenter = sparta_n_pericenter
        adj_sparta_n_pericenter[future_peri] = 0
        adj_sparta_n_is_lower_limit = sparta_n_is_lower_limit
        adj_sparta_n_is_lower_limit[future_peri] = 0
        # If a particle has a pericenter of the lower limit is 1 then it is orbiting
        compare_sparta_assn[np.where((adj_sparta_n_pericenter >= 1) | (adj_sparta_n_is_lower_limit == 1))[0]] = 1
        # Compare the ids between SPARTA and the found prtl ids and match the SPARTA results
        matched_ids = np.intersect1d(curr_ptl_pids, sparta_tracer_ids, return_indices = True)
        curr_orb_assn[matched_ids[1]] = compare_sparta_assn[matched_ids[2]]

    # calculate peculiar, radial, and tangential velocity
    pec_vel = calc_pec_vel(curr_ptl_vel, halo_vel)
    fnd_rad_vel, curr_v200m, physical_vel, rhat = calc_rad_vel(pec_vel, ptl_rad, coord_dist, halo_r200m, red_shift, hubble_const)
    fnd_tang_vel_comp = calc_tang_vel(fnd_rad_vel, physical_vel, rhat)/curr_v200m
    fnd_tang_vel = np.linalg.norm(fnd_tang_vel_comp, axis = 1)
    
    scaled_rad_vel = fnd_rad_vel / curr_v200m
    scaled_tang_vel = fnd_tang_vel
    scaled_radii = (ptl_rad / halo_r200m)

    if comp_snap == False:
        return fnd_HIPIDs, curr_orb_assn, scaled_rad_vel, scaled_tang_vel, scaled_radii
    else:
        return fnd_HIPIDs, scaled_rad_vel, scaled_tang_vel, scaled_radii

def halo_loop(train, indices, tot_num_ptls, p_halo_ids, p_snap, p_scale_factor, p_ptl_tree, c_snap, c_scale_factor, c_ptl_tree):
    num_iter = int(np.ceil(indices.shape[0] / num_halo_per_split))
    print("Num halo per", num_iter, "splits:", num_halo_per_split)
    hdf5_ptl_idx = 0
    hdf5_halo_idx = 0
    
    for i in range(num_iter):
        t3 = time.time()

        # Get the indices corresponding to where we are in the number of iterations (0:num_halo_persplit) -> (num_halo_persplit:2*num_halo_persplit) etc
        if i < (num_iter - 1):
            use_indices = indices[i * num_halo_per_split: (i+1) * num_halo_per_split]
        else:
            use_indices = indices[i * num_halo_per_split:]
        
        curr_num_halos = use_indices.shape[0]
        use_halo_ids = p_halo_ids[use_indices]

        # Load the halo information for the ids within this range
        global sparta_output
        sparta_output = sparta.load(filename = path_to_hdf5_file, halo_ids=use_halo_ids, log_level=0)
        new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_snap) # If the order changed by sparta resort the indices
        use_halo_idxs = use_indices[new_idxs]

        # Search around these halos and get the number of particles and the corresponding ptl indices for them
        p_use_halos_pos = sparta_output['halos']['position'][:,p_snap] * 10**3 * p_scale_factor 
        p_use_halos_r200m = sparta_output['halos']['R200m'][:,p_snap]
        global p_curr_ptl_indices

        with mp.Pool(processes=num_processes) as p:
            # halo position, halo r200m, if comparison snap, want mass?, want indices?
            p_use_num_ptls, p_curr_ptl_indices = zip(*p.starmap(initial_search, zip(p_use_halos_pos, p_use_halos_r200m, repeat(False), repeat(False), repeat(True)), chunksize=curr_chunk_size))
        p.close()
        p.join() 

        # Remove halos with 0 ptls around them
        p_use_num_ptls = np.array(p_use_num_ptls)
        p_curr_ptl_indices = np.array(p_curr_ptl_indices, dtype=object)
        has_ptls = np.where(p_use_num_ptls > 0)
        p_use_num_ptls = p_use_num_ptls[has_ptls]
        p_curr_ptl_indices = p_curr_ptl_indices[has_ptls]
        use_halo_idxs = use_halo_idxs[has_ptls]
       
        p_start_num_ptls = [np.sum(p_use_num_ptls[0:i+1]) for i in range(p_use_num_ptls.shape[0])]
        p_start_num_ptls = np.insert(p_start_num_ptls, 0, 0)
        p_start_num_ptls = np.delete(p_start_num_ptls, -1)
        p_start_num_ptls += hdf5_ptl_idx # scale to where we are in the hdf5 file
        
        p_tot_num_use_ptls = int(np.sum(p_use_num_ptls))

        # Create an array of the indices for each halo's particels
        halo_split = np.zeros((curr_num_halos,2),dtype = np.int32)
        for j in range(curr_num_halos):
            if j == curr_num_halos - 1:
                halo_split[j] = np.array([int(np.sum(p_use_num_ptls[:j])),int(np.sum(p_use_num_ptls))])
            else:
                halo_split[j] = np.array([int(np.sum(p_use_num_ptls[:j])),int(np.sum(p_use_num_ptls[:(j+1)]))])
        
        # Use multiprocessing to search multiple halos at the same time and add information to shared arrays
        with mp.Pool(processes=num_processes) as p:
            p_all_HIPIDs, p_all_orb_assn, p_all_rad_vel, p_all_tang_vel, p_all_scal_rad = zip(*p.starmap(search_halos, zip(halo_split, repeat(False), use_halo_idxs, np.arange(curr_num_halos)), chunksize=curr_chunk_size))
        p.close()
        p.join()
        
        p_all_HIPIDs = np.concatenate(p_all_HIPIDs, axis = 0)
        p_all_orb_assn = np.concatenate(p_all_orb_assn, axis = 0)
        p_all_rad_vel = np.concatenate(p_all_rad_vel, axis = 0)
        p_all_tang_vel = np.concatenate(p_all_tang_vel, axis = 0)
        p_all_scal_rad = np.concatenate(p_all_scal_rad, axis = 0)
        
        # If multiple snaps also search the comparison snaps in the same manner as with the primary snap
        if prim_only == False:
            c_use_halos_pos = sparta_output['halos']['position'][:,c_snap] * 10**3 * c_scale_factor
            c_use_halos_r200m = sparta_output['halos']['R200m'][:,c_snap] 
            global c_curr_ptl_indices

            with mp.Pool(processes=num_processes) as p:
                # halo position, halo r200m, if comparison snap, if train dataset, want mass?, want indices?
                c_use_num_ptls, c_curr_ptl_indices = zip(*p.starmap(initial_search, zip(c_use_halos_pos, c_use_halos_r200m, repeat(True), repeat(False), repeat(True)), chunksize=curr_chunk_size))
            p.close()
            p.join() 

            c_tot_num_use_ptls = int(np.sum(c_use_num_ptls))
            
            halo_split = np.zeros((curr_num_halos,2),dtype = np.int32)
            for j in range(curr_num_halos):
                if j == curr_num_halos - 1:
                    halo_split[j] = np.array([int(np.sum(c_use_num_ptls[:j])),int(np.sum(c_use_num_ptls))])
                else:
                    halo_split[j] = np.array([int(np.sum(c_use_num_ptls[:j])),int(np.sum(c_use_num_ptls[:(j+1)]))])
            
            with mp.Pool(processes=num_processes) as p:
                c_all_HIPIDs, c_all_rad_vel, c_all_tang_vel, c_all_scal_rad = zip(*p.starmap(search_halos, zip(halo_split, repeat(True), use_halo_idxs, np.arange(curr_num_halos)), chunksize=curr_chunk_size))
            p.close()
            p.join()
            
            c_all_HIPIDs = np.concatenate(c_all_HIPIDs, axis = 0)
            c_all_rad_vel = np.concatenate(c_all_rad_vel, axis = 0)
            c_all_tang_vel = np.concatenate(c_all_tang_vel, axis = 0)
            c_all_scal_rad = np.concatenate(c_all_scal_rad, axis = 0)
            
            use_max_shape = (tot_num_ptls,2)                  
            save_scale_radii = np.zeros((p_tot_num_use_ptls,2))
            save_rad_vel = np.zeros((p_tot_num_use_ptls,2))
            save_tang_vel = np.zeros((p_tot_num_use_ptls,2))

            # Match the PIDs from primary snap to the secondary snap
            # If they don't have a match set those as np.NaN for xgboost 
            match_hipid_idx = np.intersect1d(p_all_HIPIDs, c_all_HIPIDs, return_indices=True)
            save_scale_radii[:,0] = p_all_scal_rad 
            save_scale_radii[match_hipid_idx[1],1] = c_all_scal_rad[match_hipid_idx[2]]
            save_rad_vel[:,0] = p_all_rad_vel
            save_rad_vel[match_hipid_idx[1],1] = c_all_rad_vel[match_hipid_idx[2]]
            save_tang_vel[:,0] = p_all_tang_vel
            save_tang_vel[match_hipid_idx[1],1] = c_all_tang_vel[match_hipid_idx[2]]
            
            save_scale_radii[save_scale_radii[:,1] == 0, 1] = np.NaN
            save_rad_vel[save_rad_vel[:,1] == 0, 1] = np.NaN
            save_tang_vel[save_tang_vel[:,1] == 0, 1] = np.NaN
        
        else:
            use_max_shape = (tot_num_ptls)  
            save_scale_radii = np.zeros(p_tot_num_use_ptls)
            save_rad_vel = np.zeros(p_tot_num_use_ptls)
            save_tang_vel = np.zeros(p_tot_num_use_ptls)
            
            save_scale_radii = p_all_scal_rad
            save_rad_vel = p_all_rad_vel
            save_tang_vel = p_all_tang_vel
        
        # Save all data in hdf5 file depending on if training or testing halos]
        save_cols = ["Halo_first", "Halo_n", "HIPIDS", "Orbit_Infall", "Svaled_radii_", "Radial_vel_", "Tangential_vel_"]
        if train:
            if os.path.isfile(save_location + "train_all_particle_properties_" + curr_sparta_file + ".hdf5") and i == 0:
                os.remove(save_location + "train_all_particle_properties_" + curr_sparta_file + ".hdf5")
            with h5py.File((save_location + "train_all_particle_properties_" + curr_sparta_file + ".hdf5"), 'a') as all_particle_properties:
                save_location + "train_all_particle_properties_" + curr_sparta_file + ".hdf5"
                save_to_hdf5(all_particle_properties, "Halo_first", dataset = p_start_num_ptls, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_halo_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Halo_n", dataset = p_use_num_ptls, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_halo_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "HIPIDS", dataset = p_all_HIPIDs, chunk = True, max_shape = (tot_num_ptls,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Orbit_Infall", dataset = p_all_orb_assn, chunk = True, max_shape = (tot_num_ptls,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Scaled_radii_", dataset = save_scale_radii, chunk = True, max_shape = use_max_shape, curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Radial_vel_", dataset = save_rad_vel, chunk = True, max_shape = use_max_shape, curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Tangential_vel_", dataset = save_tang_vel, chunk = True, max_shape = use_max_shape, curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
        else:
            if os.path.isfile(save_location + "test_all_particle_properties_" + curr_sparta_file + ".hdf5") and i == 0:
                os.remove(save_location + "test_all_particle_properties_" + curr_sparta_file + ".hdf5")
            with h5py.File((save_location + "test_all_particle_properties_" + curr_sparta_file + ".hdf5"), 'a') as all_particle_properties:
                save_to_hdf5(all_particle_properties, "Halo_first", dataset = p_start_num_ptls, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_halo_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Halo_n", dataset = p_use_num_ptls, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_halo_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "HIPIDS", dataset = p_all_HIPIDs, chunk = True, max_shape = (tot_num_ptls,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Orbit_Infall", dataset = p_all_orb_assn, chunk = True, max_shape = (tot_num_ptls,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Scaled_radii_", dataset = save_scale_radii, chunk = True, max_shape = use_max_shape, curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Radial_vel_", dataset = save_rad_vel, chunk = True, max_shape = use_max_shape, curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(all_particle_properties, "Tangential_vel_", dataset = save_tang_vel, chunk = True, max_shape = use_max_shape, curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
        hdf5_ptl_idx += p_tot_num_use_ptls
        hdf5_halo_idx += p_start_num_ptls.shape[0]
        t4 = time.time()
        print("Bin", (i+1),"/",num_iter,"complete:",t4-t3,"sec")      
                
t1 = time.time()
# Set constants
cosmol = cosmology.setCosmology("bolshoi") 
p_snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)
global p_red_shift
global p_scale_factor
global p_hubble_constant
global p_box_size
global p_ptl_tree
p_red_shift = readheader(p_snapshot_path, 'redshift')
p_scale_factor = 1/(1+p_red_shift)
p_rho_m = cosmol.rho_m(p_red_shift)
p_hubble_constant = cosmol.Hz(p_red_shift) * 0.001 # convert to units km/s/kpc
p_box_size = readheader(p_snapshot_path, 'boxsize') #units Mpc/h comoving
p_box_size = p_box_size * 10**3 * p_scale_factor #convert to Kpc/h physical

p_snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

# load all information needed for the primary snap
p_ptls_pid, p_ptls_vel, p_ptls_pos = load_or_pickle_ptl_data(curr_sparta_file, str(p_snap), p_snapshot_path, p_scale_factor)

global mass
p_halos_pos, p_halos_r200m, p_halos_id, p_halos_status, p_halos_last_snap, mass = load_or_pickle_SPARTA_data(curr_sparta_file, p_scale_factor, p_snap)

t_dyn = calc_t_dyn(p_halos_r200m[np.where(p_halos_r200m > 0)[0][0]], p_red_shift)

global c_snap
global c_box_size
global c_red_shift
global c_hubble_constant
global c_scale_factor

c_snap, c_box_size, c_rho_m, c_red_shift, c_hubble_constant, c_ptls_pid, c_ptls_vel, c_ptls_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap = get_comp_snap(t_dyn=t_dyn, t_dyn_step=t_dyn_step, snapshot_list=[p_snap], cosmol = cosmol, p_red_shift=p_red_shift, total_num_snaps=total_num_snaps, snap_format=snap_format)

c_scale_factor = 1/(1+c_red_shift)
snapshot_list = [p_snap, c_snap]

if prim_only:
    save_location =  path_to_MLOIS +  "calculated_info/" + curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"
else:
    save_location =  path_to_MLOIS + "calculated_info/" + curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[1]) + "_" + str(search_rad) + "r200msearch/"

if os.path.exists(save_location) != True:
    os.makedirs(save_location)

if os.path.isfile(save_location + "p_ptl_tree.pickle"):
        with open(save_location + "p_ptl_tree.pickle", "rb") as pickle_file:
            p_ptl_tree = pickle.load(pickle_file)
else:
    p_ptl_tree = cKDTree(data = p_ptls_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size) # construct search trees for primary snap
    with open(save_location + "p_ptl_tree.pickle", "wb") as pickle_file:
        pickle.dump(p_ptl_tree, pickle_file)
        
if os.path.isfile(save_location + "c_ptl_tree.pickle"):
        with open(save_location + "c_ptl_tree.pickle", "rb") as pickle_file:
            c_ptl_tree = pickle.load(pickle_file)
else:
    c_ptl_tree = cKDTree(data = c_ptls_pos, leafsize = 3, balanced_tree = False, boxsize = c_box_size)
    with open(save_location + "c_ptl_tree.pickle", "wb") as pickle_file:
        pickle.dump(c_ptl_tree, pickle_file)

# only take halos that are hosts in primary snap and exist past the p_snap and exist in some form at the comparison snap
if prim_only:
    match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_snap))[0]
else:
    match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_snap) & (c_halos_status > 0) & (c_halos_last_snap >= c_snap))[0]
    
rng = np.random.default_rng(rand_seed)    
total_num_halos = match_halo_idxs.shape[0]
rng.shuffle(match_halo_idxs)
# split all indices into train and test groups
train_indices, test_indices = np.split(match_halo_idxs, [int((1-test_halos_ratio) * total_num_halos)])
# need to sort indices otherwise sparta.load breaks...
train_indices = np.sort(train_indices)
test_indices = np.sort(test_indices)

num_halo_per_split = int(np.ceil(per_n_halo_per_split * total_num_halos))

with open(save_location + "test_indices.pickle", "wb") as pickle_file:
    pickle.dump(test_indices, pickle_file)
with open(save_location + "train_indices.pickle", "wb") as pickle_file:
    pickle.dump(train_indices, pickle_file)

with mp.Pool(processes=num_processes) as p:
    # halo position, halo r200m, if comparison snap, want mass?, want indices?
    train_num_ptls = p.starmap(initial_search, zip(p_halos_pos[train_indices], p_halos_r200m[train_indices], repeat(False), repeat(False), repeat(False)), chunksize=curr_chunk_size)
p.close()
p.join() 

with mp.Pool(processes=num_processes) as p:
    # halo position, halo r200m, if comparison snap, want mass?, want indices?
    test_num_ptls = p.starmap(initial_search, zip(p_halos_pos[test_indices], p_halos_r200m[test_indices], repeat(False), repeat(False), repeat(False)), chunksize=curr_chunk_size)
p.close()
p.join() 

train_tot_num_ptls = np.sum(train_num_ptls)
test_tot_num_ptls = np.sum(test_num_ptls)

print("Total num halos:", total_num_halos)
print("Num train halos:", train_indices.shape[0])
print("Num test halos:", test_indices.shape[0])
print("train num ptls:", train_tot_num_ptls)
print("test num ptls:", test_tot_num_ptls)

t2 = time.time()
print("Start up finished in:",t2-t1,"seconds", (t2-t1)/60, "minutes")
halo_loop(train=True, indices=train_indices, tot_num_ptls=train_tot_num_ptls, p_halo_ids=p_halos_id, p_snap=p_snap, p_scale_factor=p_scale_factor, p_ptl_tree=p_ptl_tree, c_snap=c_snap, c_scale_factor=c_scale_factor, c_ptl_tree=c_ptl_tree)
halo_loop(train=False, indices=test_indices, tot_num_ptls=test_tot_num_ptls, p_halo_ids=p_halos_id, p_snap=p_snap, p_scale_factor=p_scale_factor, p_ptl_tree=p_ptl_tree, c_snap=c_snap, c_scale_factor=c_scale_factor, c_ptl_tree=c_ptl_tree)
t3 = time.time()
print("Finished in:",t3-t1,"seconds", (t3-t1)/60, "minutes")