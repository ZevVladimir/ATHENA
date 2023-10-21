##################################################################################################################
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
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import time
import h5py
import pickle
import os
import multiprocessing as mp
import logging
import ctypes
from itertools import repeat
from contextlib import closing
from data_and_loading_functions import load_or_pickle_SPARTA_data, load_or_pickle_ptl_data, save_to_hdf5, conv_halo_id_spid, get_comp_snap, create_directory
from visualization_functions import compare_density_prf
from calculation_functions import *
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
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
num_halo_per_split = config.getint("SEARCH","num_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
#num_processes = config.getint("SEARCH","num_processes")
num_processes = int(os.getenv('SLURM_CPUS_PER_TASK'))
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
##################################################################################################################
import sys
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
##################################################################################################################

def to_np_arr(mp_arr):
    return np.frombuffer(mp_arr.get_obj(), dtype ="float32")

def search_halos(slice, comp_snap, curr_halo_idx, curr_sparta_idx):
    # Doing this this way as otherwise will have to generate super large arrays for input from multiprocessing
    if comp_snap:
        snap = c_snap
        red_shift = c_red_shift
        scale_factor = c_scale_factor
        hubble_const = c_hubble_constant
        box_size = c_box_size
        ptl_pids = c_ptls_pid
        ptl_pos = c_ptls_pos
        ptl_vel = c_ptls_vel
        curr_ptl_indices = c_curr_ptl_indices
        share_calc_rad_vel = c_share_calc_rad_vel
        share_calc_tang_vel = c_share_calc_tang_vel
        share_all_scal_rad = c_share_all_scal_rad
        share_all_HPIDs = c_share_all_HPIDs
    else:
        snap = p_snap
        red_shift = p_red_shift
        scale_factor = p_scale_factor
        hubble_const = p_hubble_constant
        box_size = p_box_size
        ptl_pids = p_ptls_pid
        ptl_pos = p_ptls_pos
        ptl_vel = p_ptls_vel
        curr_ptl_indices = p_curr_ptl_indices
        share_calc_rad_vel = p_share_calc_rad_vel
        share_calc_tang_vel = p_share_calc_tang_vel
        share_all_scal_rad = p_share_all_scal_rad
        share_all_HPIDs = p_share_all_HPIDs
        share_all_orb_assn = p_share_all_orb_assn

    # get all the information for this specific halo
    halo_pos = sparta_output['halos']['position'][curr_sparta_idx,snap,:] * 10**3 * scale_factor
    halo_vel = sparta_output['halos']['velocity'][curr_sparta_idx,snap,:]
    halo_r200m = sparta_output['halos']['R200m'][curr_sparta_idx,snap]
    dens_prf_all = sparta_output['anl_prf']['M_all'][curr_sparta_idx,snap,:]
    dens_prf_1halo = sparta_output['anl_prf']['M_1halo'][curr_sparta_idx,snap,:]
    prf_status = sparta_output['anl_prf']['status'][curr_sparta_idx,snap]

    ptl_indices = curr_ptl_indices[slice[0]:slice[1]]

    num_new_ptls = ptl_indices.shape[0]
    curr_ptl_pids = ptl_pids[ptl_indices]
    curr_ptl_pos = ptl_pos[ptl_indices]
    curr_ptl_vel = ptl_vel[ptl_indices]
    
    curr_ptl_pids = curr_ptl_pids.astype(np.int64) # otherwise ne.evaluate doesn't work
    fnd_HPIDs = ne.evaluate("0.5 * (curr_ptl_pids + curr_halo_idx) * (curr_ptl_pids + curr_halo_idx + 1) + curr_halo_idx")
    
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
          
    if curr_halo_idx == 9512 and comp_snap == False:
        bins = sparta_output["config"]['anl_prf']["r_bins_lin"]
        bins = np.insert(bins, 0, 0)
        diff_inf, diff_orb, diff_tot, graph_bins = compare_density_prf(radii = ptl_rad / halo_r200m, actual_prf_all=dens_prf_all, actual_prf_1halo=dens_prf_1halo, mass=mass, orbit_assn=curr_orb_assn, prf_bins=bins, title = str(curr_halo_idx), save_location="/home/zvladimi/MLOIS/Random_figures/", show_graph=False, save_graph=True)
        diff_inf = np.round(diff_inf)
        diff_orb = np.round(diff_orb)
        diff_tot = np.round(diff_tot)
        
        wrong_bin_idx = np.where(diff_orb != 0)[0][-1]
        print(wrong_bin_idx)
        wrong_pids = curr_ptl_pids[np.where(((ptl_rad / halo_r200m) >= bins[wrong_bin_idx]) & ((ptl_rad / halo_r200m) < bins[wrong_bin_idx + 1]))[0]]
        new_match = np.intersect1d(wrong_pids, sparta_tracer_ids, return_indices = True)
        print(repr(sparta_tracer_ids[new_match[2]]))
        # print("infall")
        # print("where inf wrong", np.where(diff_inf != 0), diff_inf[np.where(diff_inf != 0)])
        # print("num inf ptls wrong / tot ptls", np.max(diff_inf)/ptl_rad.shape[0])
        # for idx in np.where(diff_inf != 0)[0]:
        #     print("bin:", idx, bins[idx], "to", bins[idx+1])
        #     wrong_pids_loc = np.where(((ptl_rad / halo_r200m) >= bins[idx]) & ((ptl_rad / halo_r200m) < bins[idx + 1]))[0]
        #     wrong_pids = curr_ptl_pids[wrong_pids_loc]
        #     new_match_ids = np.intersect1d(wrong_pids, sparta_tracer_ids, return_indices = True)
        #     for i in range(new_match_ids[0].shape[0]):
        #         print("last_pericenter_snap:",sparta_last_pericenter_snap[new_match_ids[2][i]], "n_pericenter", adj_sparta_n_pericenter[new_match_ids[2][i]], 
        #               "n_is_lower_limit", adj_sparta_n_is_lower_limit[new_match_ids[2][i]], "orb_assn", curr_orb_assn[wrong_pids_loc[new_match_ids[1][i]]],
        #               "radius", ptl_rad[wrong_pids_loc[new_match_ids[1][i]]], "radius/r200m", ptl_rad[wrong_pids_loc[new_match_ids[1][i]]]/halo_r200m)
        # print("orbit")
        # print("where orb wrong", np.where(diff_orb != 0), diff_orb[np.where(diff_orb != 0)])
        # print("num orb ptls wrong / tot ptls", np.max(diff_orb)/ptl_rad.shape[0])
        # for idx in np.where(diff_orb != 0)[0]:
        #     print("bin:", idx, bins[idx], "to", bins[idx+1])
        #     wrong_pids_loc = np.where(((ptl_rad / halo_r200m) >= bins[idx]) & ((ptl_rad / halo_r200m) < bins[idx + 1]))[0]
        #     wrong_pids = curr_ptl_pids[wrong_pids_loc]
        #     new_match_ids = np.intersect1d(wrong_pids, sparta_tracer_ids, return_indices = True)
        #     for i in range(new_match_ids[0].shape[0]):
        #         print("last_pericenter_snap:",sparta_last_pericenter_snap[new_match_ids[2][i]], "n_pericenter", adj_sparta_n_pericenter[new_match_ids[2][i]], 
        #               "n_is_lower_limit", adj_sparta_n_is_lower_limit[new_match_ids[2][i]], "orb_assn", curr_orb_assn[wrong_pids_loc[new_match_ids[1][i]]],
        #               "radius", ptl_rad[wrong_pids_loc[new_match_ids[1][i]]], "radius/r200m", ptl_rad[wrong_pids_loc[new_match_ids[1][i]]]/halo_r200m)
        
    #     # print("total")
    #     # print("where tot wrong", np.where(diff_tot != 0), diff_tot[np.where(diff_tot != 0)])
    #     # print("num tot ptls wrong / tot ptls", np.max(diff_tot)/ptl_rad.shape[0])
    #     # for idx in np.where(diff_tot != 0)[0]:
    #     #     print("bin:", idx, bins[idx], "to", bins[idx+1])
    #     #     wrong_pids_loc = np.where(((ptl_rad / halo_r200m) >= bins[idx]) & ((ptl_rad / halo_r200m) < bins[idx + 1]))[0]
    #     #     wrong_pids = curr_ptl_pids[wrong_pids_loc]
    #     #     new_match_ids = np.intersect1d(wrong_pids, sparta_tracer_ids, return_indices = True)
    #     #     for i in range(new_match_ids[0].shape[0]):
    #     #         print("last_pericenter_snap:",sparta_last_pericenter_snap[new_match_ids[2][i]], "n_pericenter", adj_sparta_n_pericenter[new_match_ids[2][i]], 
    #     #               "n_is_lower_limit", adj_sparta_n_is_lower_limit[new_match_ids[2][i]], "orb_assn", curr_orb_assn[wrong_pids_loc[new_match_ids[1][i]]],
    #     #               "radius", ptl_rad[wrong_pids_loc[new_match_ids[1][i]]], "radius/r200m", ptl_rad[wrong_pids_loc[new_match_ids[1][i]]]/halo_r200m)

        fig,ax = plt.subplots(1, layout = "tight")
        ax.scatter(graph_bins,diff_inf, c = 'g', alpha = 0.5, label = "infall ptls")
        ax.scatter(graph_bins,diff_orb, c = 'b', alpha = 0.5, label = "orb ptls")
        ax.scatter(graph_bins,diff_tot, c = 'r', alpha = 0.5, label = "tot ptls")
        ax.set_title("Num of Ptls SPARTA - Num Ptls Code; Halo idx: " + str(curr_halo_idx))
        ax.set_xlabel("radius $r/R_{200m}$")
        ax.set_ylabel("Number of ptls")
        ax.set_xscale("log")
        ax.legend()
        fig.savefig("/home/zvladimi/MLOIS/Random_figures/" + str(curr_halo_idx) + "_num_ptls_wrong_class.png")

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(halo_pos[0],halo_pos[1],halo_pos[2],s = 10, marker="*")
        # ax.scatter(curr_ptl_pos[matched_ids[1],0], curr_ptl_pos[matched_ids[1],1],curr_ptl_pos[matched_ids[1],2], s = 2, marker = ".")
        # fig.savefig("/home/zvladimi/MLOIS/Random_figures/" + str(curr_halo_idx) + "sparta_ptls.png")

    # calculate peculiar, radial, and tangential velocity
    pec_vel = calc_pec_vel(curr_ptl_vel, halo_vel)
    fnd_rad_vel, curr_v200m, physical_vel, rhat = calc_rad_vel(pec_vel, ptl_rad, coord_dist, halo_r200m, red_shift, hubble_const)
    fnd_tang_vel_comp = calc_tang_vel(fnd_rad_vel, physical_vel, rhat)/curr_v200m
    fnd_tang_vel = np.linalg.norm(fnd_tang_vel_comp, axis = 1)
    
    all_rad_vel = to_np_arr(share_calc_rad_vel)
    all_tang_vel = to_np_arr(share_calc_tang_vel)
    all_scal_radii = to_np_arr(share_all_scal_rad)
    all_HPIDS = to_np_arr(share_all_HPIDs)
    
    all_rad_vel[slice[0]:slice[1]] = fnd_rad_vel / curr_v200m
    all_tang_vel[slice[0]:slice[1]] = fnd_tang_vel
    all_scal_radii[slice[0]:slice[1]] = (ptl_rad / halo_r200m)
    all_HPIDS[slice[0]:slice[1]] = fnd_HPIDs

    if comp_snap == False:
        all_orb_assn = to_np_arr(share_all_orb_assn)
        all_orb_assn[slice[0]:slice[1]] = curr_orb_assn

def halo_loop(train, indices, tot_num_ptls, p_halo_ids, p_snap, p_scale_factor, p_ptl_tree, c_snap, c_scale_factor, c_ptl_tree):
    num_halos = indices.shape[0]
    num_iter = int(np.ceil(indices.shape[0] / num_halo_per_split))
    all_start_idx = 0
    count_num_ptls = 0
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
        use_halos_pos = sparta_output['halos']['position'][:,p_snap] * 10**3 * p_scale_factor 
        use_halos_r200m = sparta_output['halos']['R200m'][:,p_snap]
        global p_curr_ptl_indices
 
        p_use_num_ptls, p_all_halo_mass, p_curr_ptl_indices = initial_search(use_halos_pos, search_rad, use_halos_r200m, p_ptl_tree, mass, find_ptl_indices=True)
        
        p_tot_num_use_ptls = int(np.sum(p_use_num_ptls))
        
        # Set up arrays that can be accessed by all processes
        global p_share_calc_rad_vel
        global p_share_calc_tang_vel
        global p_share_all_scal_rad
        global p_share_all_HPIDs
        global p_share_all_orb_assn
        p_share_calc_rad_vel = mp.Array(ctypes.c_float,p_tot_num_use_ptls)
        p_share_calc_tang_vel  = mp.Array(ctypes.c_float,p_tot_num_use_ptls)
        p_share_all_scal_rad = mp.Array(ctypes.c_float,p_tot_num_use_ptls)
        p_share_all_HPIDs = mp.Array(ctypes.c_uint,p_tot_num_use_ptls)
        p_share_all_orb_assn = mp.Array(ctypes.c_float,p_tot_num_use_ptls)
        
        # Create an array of the indices for each halo's particels
        halo_split = np.zeros((curr_num_halos,2),dtype = np.int32)
        for j in range(curr_num_halos):
            if j == curr_num_halos - 1:
                halo_split[j] = np.array([int(np.sum(p_use_num_ptls[:j])),int(np.sum(p_use_num_ptls))])
            else:
                halo_split[j] = np.array([int(np.sum(p_use_num_ptls[:j])),int(np.sum(p_use_num_ptls[:(j+1)]))])
        
        # Use multiprocessing to search multiple halos at the same time and add information to shared arrays
        with mp.Pool(processes=num_processes) as p:
            results = p.starmap(search_halos, zip(halo_split,repeat(False), use_halo_idxs, np.arange(curr_num_halos)))

        p.join()
        p.close()
        
        p_all_rad_vel = to_np_arr(p_share_calc_rad_vel)
        p_all_tang_vel = to_np_arr(p_share_calc_tang_vel)
        p_all_scal_rad = to_np_arr(p_share_all_scal_rad)
        p_all_HPIDs = to_np_arr(p_share_all_HPIDs)
        p_all_orb_assn = to_np_arr(p_share_all_orb_assn) 
        
        # If multiple snaps also search the comparison snaps in the same manner as with the primary snap
        if prim_only == False:
            use_halos_pos = sparta_output['halos']['position'][:,c_snap] * 10**3 * p_scale_factor
            use_halos_r200m = sparta_output['halos']['R200m'][:,c_snap] 
            global c_curr_ptl_indices
            c_use_num_ptls, c_all_halo_mass, c_curr_ptl_indices = initial_search(use_halos_pos, search_rad, use_halos_r200m, c_ptl_tree, mass, find_ptl_indices=True)

            c_tot_num_use_ptls = int(np.sum(c_use_num_ptls))

            global c_share_calc_rad_vel
            global c_share_calc_tang_vel
            global c_share_all_scal_rad
            global c_share_all_HPIDs
            c_share_calc_rad_vel = mp.Array(ctypes.c_float,c_tot_num_use_ptls)
            c_share_calc_tang_vel  = mp.Array(ctypes.c_float,c_tot_num_use_ptls)
            c_share_all_scal_rad = mp.Array(ctypes.c_float,c_tot_num_use_ptls)
            c_share_all_HPIDs = mp.Array(ctypes.c_uint,c_tot_num_use_ptls)
            
            halo_split = np.zeros((curr_num_halos,2),dtype = np.int32)
            for j in range(curr_num_halos):
                if j == curr_num_halos - 1:
                    halo_split[j] = np.array([int(np.sum(c_use_num_ptls[:j])),int(np.sum(c_use_num_ptls))])
                else:
                    halo_split[j] = np.array([int(np.sum(c_use_num_ptls[:j])),int(np.sum(c_use_num_ptls[:(j+1)]))])
            
            with mp.Pool(processes=num_processes) as p:
                p.starmap(search_halos, zip(halo_split, repeat(True), use_halo_idxs, np.arange(curr_num_halos)))
            p.join()
            p.close()

            c_all_rad_vel = to_np_arr(c_share_calc_rad_vel)
            c_all_tang_vel = to_np_arr(c_share_calc_tang_vel)
            c_all_scal_rad = to_np_arr(c_share_all_scal_rad)
            c_all_HPIDs = to_np_arr(c_share_all_HPIDs)
                
            use_max_shape = (tot_num_ptls,2)                  
            save_scale_radii = np.zeros((p_tot_num_use_ptls,2))
            save_rad_vel = np.zeros((p_tot_num_use_ptls,2))
            save_tang_vel = np.zeros((p_tot_num_use_ptls,2))
            
            # Match the PIDs from primary snap to the secondary snap
            # If they don't have a match just leave those as 0s
            match_pidh_idx = np.intersect1d(p_all_HPIDs, c_all_HPIDs, return_indices=True)
            save_scale_radii[:,0] = p_all_scal_rad 
            save_scale_radii[match_pidh_idx[1],1] = c_all_scal_rad[match_pidh_idx[2]]
            save_rad_vel[:,0] = p_all_rad_vel
            save_rad_vel[match_pidh_idx[1],1] = c_all_rad_vel[match_pidh_idx[2]]
            save_tang_vel[:,0] = p_all_tang_vel
            save_tang_vel[match_pidh_idx[1],1] = c_all_tang_vel[match_pidh_idx[2]]
        
        else:
            use_max_shape = (tot_num_ptls)  
            save_scale_radii = np.zeros(p_tot_num_use_ptls)
            save_rad_vel = np.zeros(p_tot_num_use_ptls)
            save_tang_vel = np.zeros(p_tot_num_use_ptls)
            p_snap

        count_num_ptls = count_num_ptls + int(p_all_HPIDs.shape[0])
        new_file = True
        all_start_idx += p_tot_num_use_ptls
        # Save all data in hdf5 file depending on if training or testing halos
        if train:
            with h5py.File((save_location + "train_all_particle_properties_" + curr_sparta_file + ".hdf5"), 'a') as all_particle_properties:
                save_location + "train_all_particle_properties_" + curr_sparta_file + ".hdf5"
                save_to_hdf5(new_file, all_particle_properties, "HPIDS", dataset = p_all_HPIDs, chunk = True, max_shape = (tot_num_ptls,), curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Orbit_Infall", dataset = p_all_orb_assn, chunk = True, max_shape = (tot_num_ptls,), curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Scaled_radii_", dataset = save_scale_radii, chunk = True, max_shape = use_max_shape, curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Radial_vel_", dataset = save_rad_vel, chunk = True, max_shape = use_max_shape, curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Tangential_vel_", dataset = save_tang_vel, chunk = True, max_shape = use_max_shape, curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
        else:
            with h5py.File((save_location + "test_all_particle_properties_" + curr_sparta_file + ".hdf5"), 'a') as all_particle_properties:
                save_to_hdf5(new_file, all_particle_properties, "HPIDS", dataset = p_all_HPIDs, chunk = True, max_shape = (tot_num_ptls,), curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Orbit_Infall", dataset = p_all_orb_assn, chunk = True, max_shape = (tot_num_ptls,), curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Scaled_radii_", dataset = save_scale_radii, chunk = True, max_shape = use_max_shape, curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Radial_vel_", dataset = save_rad_vel, chunk = True, max_shape = use_max_shape, curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Tangential_vel_", dataset = save_tang_vel, chunk = True, max_shape = use_max_shape, curr_idx = all_start_idx, max_num_keys = num_save_ptl_params)
    
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

p_ptls_pid, p_ptls_vel, p_ptls_pos = load_or_pickle_ptl_data(p_snap, p_scale_factor)
global mass
p_halos_pos, p_halos_r200m, p_halos_id, p_halos_status, p_halos_last_snap, mass = load_or_pickle_SPARTA_data(p_snap, p_scale_factor)

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
    save_location =  path_to_calc_info + curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"
else:
    save_location =  path_to_calc_info + curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[1]) + "_" + str(search_rad) + "r200msearch/"

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

if os.path.isfile(save_location + "train_use_num_ptls.pickle") and os.path.isfile(save_location + "train_all_halo_mass.pickle"):
        with open(save_location + "train_use_num_ptls.pickle", "rb") as pickle_file:
            train_use_num_ptls = pickle.load(pickle_file)
        with open(save_location + "train_all_halo_mass.pickle", "rb") as pickle_file:
            train_all_halo_mass = pickle.load(pickle_file)
else:
    train_use_num_ptls, train_all_halo_mass = initial_search(p_halos_pos[train_indices], search_rad, p_halos_r200m[train_indices], p_ptl_tree, mass, find_ptl_indices=False)
    with open(save_location + "train_use_num_ptls.pickle", "wb") as pickle_file:
        pickle.dump(train_use_num_ptls, pickle_file)
    with open(save_location + "train_all_halo_mass.pickle", "wb") as pickle_file:
        pickle.dump(train_all_halo_mass, pickle_file)

if os.path.isfile(save_location + "test_use_num_ptls.pickle") and os.path.isfile(save_location + "test_all_halo_mass.pickle"):
        with open(save_location + "test_use_num_ptls.pickle", "rb") as pickle_file:
            test_use_num_ptls = pickle.load(pickle_file)
        with open(save_location + "test_all_halo_mass.pickle", "rb") as pickle_file:
            test_all_halo_mass = pickle.load(pickle_file)
else:
    test_use_num_ptls, test_all_halo_mass = initial_search(p_halos_pos[test_indices], search_rad, p_halos_r200m[test_indices], p_ptl_tree, mass, find_ptl_indices=False)
    with open(save_location + "test_use_num_ptls.pickle", "wb") as pickle_file:
        pickle.dump(train_use_num_ptls, pickle_file)
    with open(save_location + "test_all_halo_mass.pickle", "wb") as pickle_file:
        pickle.dump(train_all_halo_mass, pickle_file)

train_tot_num_ptls = np.sum(train_use_num_ptls)
test_tot_num_ptls = np.sum(test_use_num_ptls)

print("Total num halos:", total_num_halos)
print("Num train halos:", train_indices.shape[0])
print("Num test halos:", test_indices.shape[0])

halo_loop(train=True, indices=train_indices, tot_num_ptls=train_tot_num_ptls, p_halo_ids=p_halos_id, p_snap=p_snap, p_scale_factor=p_scale_factor, p_ptl_tree=p_ptl_tree, c_snap=c_snap, c_scale_factor=c_scale_factor, c_ptl_tree=c_ptl_tree)
halo_loop(train=False, indices=test_indices, tot_num_ptls=test_tot_num_ptls, p_halo_ids=p_halos_id, p_snap=p_snap, p_scale_factor=p_scale_factor, p_ptl_tree=p_ptl_tree, c_snap=c_snap, c_scale_factor=c_scale_factor, c_ptl_tree=c_ptl_tree)
t2 = time.time()
print("Finished in:",t2-t1,"seconds", (t2-t1)/60, "minutes")