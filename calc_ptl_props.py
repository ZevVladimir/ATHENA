import numexpr as ne
import numpy as np
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import peaks
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from contextlib import contextmanager
import time
import h5py
import pickle
import os
import multiprocessing as mp
from itertools import repeat
import sys

from utils.data_and_loading_functions import load_or_pickle_SPARTA_data, load_or_pickle_ptl_data, save_to_hdf5, conv_halo_id_spid, get_comp_snap, create_directory, find_closest_z
from utils.calculation_functions import *
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
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
global p_red_shift
p_red_shift = config.getfloat("SEARCH","p_red_shift")
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
# num_processes = int(os.environ['SLURM_CPUS_PER_TASK'])
num_processes = mp.cpu_count()
curr_chunk_size = config.getint("SEARCH","chunk_size")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
##################################################################################################################
sys.path.insert(1, path_to_pygadgetreader)
sys.path.insert(1, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta
##################################################################################################################
@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))
    
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
        all_ptl_indices = np.empty(1)
    
    if find_mass == False and find_ptl_indices == False:
        return num_new_particles
    elif find_mass == True and find_ptl_indices == False:
        return num_new_particles, halo_mass
    elif find_mass == False and find_ptl_indices == True:
        return num_new_particles, all_ptl_indices
    else:
        return num_new_particles, halo_mass, all_ptl_indices
    
def search_halos(comp_snap, snap_dict, curr_halo_idx, curr_sparta_idx, curr_ptl_pids, curr_ptl_pos, curr_ptl_vel, 
                 halo_pos, halo_vel, halo_r200m, sparta_last_pericenter_snap=None, sparta_n_pericenter=None, sparta_tracer_ids=None,
                 sparta_n_is_lower_limit=None, dens_prf_all=None, dens_prf_1halo=None, bins=None, create_dens_prf=False):
    # Doing this this way as otherwise will have to generate super large arrays for input from multiprocessing
    snap = snap_dict["snap"]
    red_shift = snap_dict["red_shift"]
    scale_factor = snap_dict["scale_factor"]
    hubble_const = snap_dict["hubble_const"]
    box_size = snap_dict["box_size"]    
    
    halo_pos = halo_pos * 10**3 * scale_factor
    
    num_new_ptls = curr_ptl_pids.shape[0]

    curr_ptl_pids = curr_ptl_pids.astype(np.int64) # otherwise ne.evaluate doesn't work
    fnd_HIPIDs = ne.evaluate("0.5 * (curr_ptl_pids + curr_halo_idx) * (curr_ptl_pids + curr_halo_idx + 1) + curr_halo_idx")
    
    #calculate the radii of each particle based on the distance formula
    ptl_rad, coord_dist = calculate_distance(halo_pos[0], halo_pos[1], halo_pos[2], curr_ptl_pos[:,0], curr_ptl_pos[:,1], curr_ptl_pos[:,2], num_new_ptls, box_size)         
    
    if comp_snap == False:         
        compare_sparta_assn = np.zeros((sparta_tracer_ids.shape[0]))
        curr_orb_assn = np.zeros((num_new_ptls))
         # Anywhere sparta_last_pericenter is greater than the current snap then that is in the future so set to 0
        future_peri = np.where(sparta_last_pericenter_snap > snap)[0]
        adj_sparta_n_pericenter = sparta_n_pericenter
        adj_sparta_n_pericenter[future_peri] = 0
        adj_sparta_n_is_lower_limit = sparta_n_is_lower_limit
        adj_sparta_n_is_lower_limit[future_peri] = 0
        # If a particle has a pericenter of the lower limit is 1 then it is orbiting
        if (total_num_snaps - snap) <= 3:
            compare_sparta_assn[np.where((adj_sparta_n_pericenter >= 1) | (adj_sparta_n_is_lower_limit == 1))[0]] = 1
        else: 
            compare_sparta_assn[np.where(adj_sparta_n_pericenter >= 1)] = 1
        # Compare the ids between SPARTA and the found prtl ids and match the SPARTA results
        matched_ids = np.intersect1d(curr_ptl_pids, sparta_tracer_ids, return_indices = True)
        curr_orb_assn[matched_ids[1]] = compare_sparta_assn[matched_ids[2]]

    # calculate peculiar, radial, and tangential velocity
    pec_vel = calc_pec_vel(curr_ptl_vel, halo_vel)
    fnd_rad_vel, curr_v200m, phys_vel, phys_vel_comp, rhat = calc_rad_vel(pec_vel, ptl_rad, coord_dist, halo_r200m, red_shift, hubble_const)
    fnd_tang_vel_comp = calc_tang_vel(fnd_rad_vel, phys_vel_comp, rhat)
    fnd_tang_vel = np.linalg.norm(fnd_tang_vel_comp, axis = 1)
    
    scal_sqr_phys_vel = np.log(np.divide(np.square(phys_vel),np.square(curr_v200m)))
    
    scal_sqr_phys_vel[np.where(np.isinf(scal_sqr_phys_vel) == True)[0]]=0
  
    
    scaled_rad_vel = fnd_rad_vel / curr_v200m
    scaled_tang_vel = fnd_tang_vel / curr_v200m
    scaled_radii = ptl_rad / halo_r200m
    
    if comp_snap == False:
        # particles with very high radial velocities at small radii should be considered infalling
        # incorrectly classified
        curr_orb_assn[np.where((scaled_radii < 1.1) & (np.abs(scaled_rad_vel)>10))] = 0
    
    # if np.where(np.abs(scaled_rad_vel[np.where(scaled_radii < 1.1)[0]]) > 40)[0].shape[0] > 0.5 * np.where(scaled_radii < 1.1)[0].shape[0]:
    #     high_vel_idx = np.where((np.abs(scaled_rad_vel) > 50) & (scaled_radii < 1.1))[0]
    #     within_rad = np.where(scaled_radii<1.1)[0]
        
    #     plot_rhat = calc_rhat(physical_vel[:,0], physical_vel[:,1], physical_vel[:,2])

    #     print("num high vel ptls in 1.1R200m:", high_vel_idx.shape[0], "num total ptls in 1.1R200m:", np.where(scaled_radii<1.1)[0].shape)

    #     transparency = np.abs(physical_vel)/np.max(np.abs(physical_vel))
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     # ax.quiver(curr_ptl_pos[within_rad,0],curr_ptl_pos[within_rad,1],curr_ptl_pos[within_rad,2], plot_rhat[within_rad,0], plot_rhat[within_rad,1], plot_rhat[within_rad,2], alpha=transparency[within_rad], length=2, color='b')
    #     # ax.quiver(curr_ptl_pos[high_vel_idx,0],curr_ptl_pos[high_vel_idx,1],curr_ptl_pos[high_vel_idx,2], plot_rhat[high_vel_idx,0], plot_rhat[high_vel_idx,1], plot_rhat[high_vel_idx,2], alpha=transparency[high_vel_idx],length=2, color='r')
    #     ax.quiver(curr_ptl_pos[within_rad,0],curr_ptl_pos[within_rad,1],curr_ptl_pos[within_rad,2], plot_rhat[within_rad,0], plot_rhat[within_rad,1], plot_rhat[within_rad,2], length=2, color='b')
    #     ax.quiver(curr_ptl_pos[high_vel_idx,0],curr_ptl_pos[high_vel_idx,1],curr_ptl_pos[high_vel_idx,2], plot_rhat[high_vel_idx,0], plot_rhat[high_vel_idx,1], plot_rhat[high_vel_idx,2], length=2, color='r')
    #     fig.savefig("/home/zvladimi/scratch/MLOIS/Random_figures/high_vel_halo" + str(curr_halo_idx) + ".png")
    #     plt.close()

    scaled_radii_inds = scaled_radii.argsort()
    scaled_radii = scaled_radii[scaled_radii_inds]
    fnd_HIPIDs = fnd_HIPIDs[scaled_radii_inds]
    scaled_rad_vel = scaled_rad_vel[scaled_radii_inds]
    scaled_tang_vel = scaled_tang_vel[scaled_radii_inds]
    scal_sqr_phys_vel = scal_sqr_phys_vel[scaled_radii_inds]
    if comp_snap == False:
        curr_orb_assn = curr_orb_assn[scaled_radii_inds]
        
    if comp_snap == False:
        return fnd_HIPIDs, curr_orb_assn, scaled_rad_vel, scaled_tang_vel, scaled_radii, scal_sqr_phys_vel
    else:
        return fnd_HIPIDs, scaled_rad_vel, scaled_tang_vel, scaled_radii

def halo_loop(train, indices, tot_num_ptls, p_halo_ids, p_dict, p_ptls_pid, p_ptls_pos, p_ptls_vel, c_dict, c_ptls_pid, c_ptls_pos, c_ptls_vel):
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
        sparta_output = sparta.load(filename = path_to_hdf5_file, halo_ids=use_halo_ids, log_level=0)

        global c_sparta_snap
        c_sparta_snap = np.abs(sparta_output["simulation"]["snap_z"][:] - c_red_shift).argmin()
        
        new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta re-sort the indices
        use_halo_idxs = use_indices[new_idxs]

        # Search around these halos and get the number of particles and the corresponding ptl indices for them
        p_use_halos_pos = sparta_output['halos']['position'][:,p_sparta_snap] * 10**3 * p_scale_factor 
        p_use_halos_r200m = sparta_output['halos']['R200m'][:,p_sparta_snap]

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
        p_use_halo_idxs = use_halo_idxs[has_ptls]
       
        # We need to correct for having previous halos being searched so the final halo_first quantity is for all halos
        # First obtain the halo_first values for this batch and then adjust to where the hdf5 file currently is
        p_start_num_ptls = [np.sum(p_use_num_ptls[0:i+1]) for i in range(p_use_num_ptls.shape[0])]
        p_start_num_ptls = np.insert(p_start_num_ptls, 0, 0)
        p_start_num_ptls = np.delete(p_start_num_ptls, -1)
        p_start_num_ptls += hdf5_ptl_idx # scale to where we are in the hdf5 file
        
        p_tot_num_use_ptls = int(np.sum(p_use_num_ptls))

        halo_first = sparta_output['halos']['ptl_oct_first'][:]
        halo_n = sparta_output['halos']['ptl_oct_n'][:]
        
        # Use multiprocessing to search multiple halos at the same time and add information to shared arrays
        with mp.Pool(processes=num_processes) as p:
            p_all_HIPIDs, p_all_orb_assn, p_all_rad_vel, p_all_tang_vel, p_all_scal_rad, p_scal_sqr_phys_vel = zip(*p.starmap(search_halos, 
                                        zip(repeat(False), repeat(p_dict), p_use_halo_idxs, np.arange(curr_num_halos),
                                        (p_ptls_pid[p_curr_ptl_indices[i]] for i in range(curr_num_halos)), 
                                        (p_ptls_pos[p_curr_ptl_indices[j]] for j in range(curr_num_halos)),
                                        (p_ptls_vel[p_curr_ptl_indices[k]] for k in range(curr_num_halos)),
                                        (sparta_output['halos']['position'][l,p_sparta_snap,:] for l in range(curr_num_halos)),
                                        (sparta_output['halos']['velocity'][l,p_sparta_snap,:] for l in range(curr_num_halos)),
                                        (sparta_output['halos']['R200m'][l,p_sparta_snap] for l in range(curr_num_halos)),
                                        (sparta_output['tcr_ptl']['res_oct']['last_pericenter_snap'][halo_first[m]:halo_first[m]+halo_n[m]] for m in range(curr_num_halos)),
                                        (sparta_output['tcr_ptl']['res_oct']['n_pericenter'][halo_first[m]:halo_first[m]+halo_n[m]] for m in range(curr_num_halos)),
                                        (sparta_output['tcr_ptl']['res_oct']['tracer_id'][halo_first[m]:halo_first[m]+halo_n[m]] for m in range(curr_num_halos)),
                                        (sparta_output['tcr_ptl']['res_oct']['n_is_lower_limit'][halo_first[m]:halo_first[m]+halo_n[m]] for m in range(curr_num_halos)),
                                        (sparta_output['anl_prf']['M_all'][l,p_sparta_snap,:] for l in range(curr_num_halos)),
                                        (sparta_output['anl_prf']['M_1halo'][l,p_sparta_snap,:] for l in range(curr_num_halos)),
                                        # Uncomment below to create dens profiles
                                        #repeat(sparta_output["config"]['anl_prf']["r_bins_lin"]),repeat(True) 
                                        ),chunksize=curr_chunk_size))
        p.close()
        p.join()
        
        p_all_HIPIDs = np.concatenate(p_all_HIPIDs, axis = 0)
        p_all_orb_assn = np.concatenate(p_all_orb_assn, axis = 0)
        p_all_rad_vel = np.concatenate(p_all_rad_vel, axis = 0)
        p_all_tang_vel = np.concatenate(p_all_tang_vel, axis = 0)
        p_all_scal_rad = np.concatenate(p_all_scal_rad, axis = 0)
        p_scal_sqr_phys_vel = np.concatenate(p_scal_sqr_phys_vel, axis=0)
        
        p_all_HIPIDs = p_all_HIPIDs.astype(np.float64)
        p_all_orb_assn = p_all_orb_assn.astype(np.int8)
        p_all_rad_vel = p_all_rad_vel.astype(np.float32)
        p_all_tang_vel = p_all_tang_vel.astype(np.float32)
        p_all_scal_rad = p_all_scal_rad.astype(np.float32)
        p_scal_sqr_phys_vel = p_scal_sqr_phys_vel.astype(np.float32)
        
        num_bins = 30
    
        # If multiple snaps also search the comparison snaps in the same manner as with the primary snap
        if prim_only == False:
            c_use_halos_pos = sparta_output['halos']['position'][:,c_sparta_snap] * 10**3 * c_scale_factor
            c_use_halos_r200m = sparta_output['halos']['R200m'][:,c_sparta_snap] 

            with mp.Pool(processes=num_processes) as p:
                # halo position, halo r200m, if comparison snap, if train dataset, want mass?, want indices?
                c_use_num_ptls, c_curr_ptl_indices = zip(*p.starmap(initial_search, zip(c_use_halos_pos, c_use_halos_r200m, repeat(True), repeat(False), repeat(True)), chunksize=curr_chunk_size))
            p.close()
            p.join() 

            c_use_num_ptls = np.array(c_use_num_ptls)
            c_curr_ptl_indices = np.array(c_curr_ptl_indices, dtype=object)
            has_ptls = np.where(c_use_num_ptls > 0)
            c_use_num_ptls = c_use_num_ptls[has_ptls]
            c_curr_ptl_indices = c_curr_ptl_indices[has_ptls]
            c_use_halo_idxs = use_halo_idxs[has_ptls]

            c_tot_num_use_ptls = int(np.sum(c_use_num_ptls))
            
            
            with mp.Pool(processes=num_processes) as p:
                c_all_HIPIDs, c_all_rad_vel, c_all_tang_vel, c_all_scal_rad = zip(*p.starmap(search_halos, 
                                            zip(repeat(True), repeat(c_dict),c_use_halo_idxs, np.arange(curr_num_halos),
                                                (c_ptls_pid[c_curr_ptl_indices[i]] for i in range(curr_num_halos)), 
                                                (c_ptls_pos[c_curr_ptl_indices[j]] for j in range(curr_num_halos)),
                                                (c_ptls_vel[c_curr_ptl_indices[k]] for k in range(curr_num_halos)),
                                                (sparta_output['halos']['position'][l,c_sparta_snap,:] for l in range(curr_num_halos)),
                                                (sparta_output['halos']['velocity'][l,c_sparta_snap,:] for l in range(curr_num_halos)),
                                                (sparta_output['halos']['R200m'][l,c_sparta_snap] for l in range(curr_num_halos)),
                                                ),chunksize=curr_chunk_size))
            p.close()
            p.join()

            c_all_HIPIDs = np.concatenate(c_all_HIPIDs, axis = 0)
            c_all_rad_vel = np.concatenate(c_all_rad_vel, axis = 0)
            c_all_tang_vel = np.concatenate(c_all_tang_vel, axis = 0)
            c_all_scal_rad = np.concatenate(c_all_scal_rad, axis = 0)
            
            c_all_HIPIDs = c_all_HIPIDs.astype(np.float64)
            c_all_rad_vel = c_all_rad_vel.astype(np.float32)
            c_all_tang_vel = c_all_tang_vel.astype(np.float32)
            c_all_scal_rad = c_all_scal_rad.astype(np.float32)
            
            use_max_shape = (tot_num_ptls,2)                  
            save_scale_radii = np.zeros((p_tot_num_use_ptls,2), dtype = np.float32)
            save_rad_vel = np.zeros((p_tot_num_use_ptls,2), dtype = np.float32)
            save_tang_vel = np.zeros((p_tot_num_use_ptls,2), dtype = np.float32)

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
           
            
        #plot_r_rv_tv_graph(orb_inf=np.zeros(p_all_orb_assn.size), r=p_all_scal_rad, rv=p_all_rad_vel, tv=p_all_tang_vel, correct_orb_inf=p_all_scal_rad, title="test rv plot", num_bins=30, save_location="/home/zvladimi/scratch/MLOIS/Random_figures/")
        
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
                save_to_hdf5(all_particle_properties, "scal_sqr_phys_vel", dataset = p_scal_sqr_phys_vel, chunk = True, max_shape = (tot_num_ptls,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
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
                save_to_hdf5(all_particle_properties, "scal_sqr_phys_vel", dataset = p_scal_sqr_phys_vel, chunk = True, max_shape = (tot_num_ptls,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
        hdf5_ptl_idx += p_tot_num_use_ptls
        hdf5_halo_idx += p_start_num_ptls.shape[0]
        t4 = time.time()
        del sparta_output
        print("Bin", (i+1),"/",num_iter,"complete:",np.round(((t4-t3)/60),2), "min", np.round((t4-t3),2),"sec")      
                
t1 = time.time()

cosmol = cosmology.setCosmology("bolshoi") 

with timed("p_snap information load"):
    p_snap, p_red_shift = find_closest_z(p_red_shift)
    print("Snapshot number found:", p_snap, "Closest redshift found:", p_red_shift)
    sparta_output = sparta.load(filename=path_to_hdf5_file, load_halo_data=False, log_level= 0)
    all_red_shifts = sparta_output["simulation"]["snap_z"][:]
    p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
    print("corresponding SPARTA snap num:", p_sparta_snap)
    print("check sparta redshift:",all_red_shifts[p_sparta_snap])
    

    # Set constants
    p_snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

    p_scale_factor = 1/(1+p_red_shift)
    p_rho_m = cosmol.rho_m(p_red_shift)
    p_hubble_constant = cosmol.Hz(p_red_shift) * 0.001 # convert to units km/s/kpc
    sim_box_size = sparta_output["simulation"]["box_size"] #units Mpc/h comoving
    p_box_size = sim_box_size * 10**3 * p_scale_factor #convert to Kpc/h physical

    p_snap_dict = {
        "snap":p_snap,
        "red_shift":p_red_shift,
        "scale_factor": p_scale_factor,
        "hubble_const": p_hubble_constant,
        "box_size": p_box_size,
    }


# load all information needed for the primary snap
with timed("p_snap ptl load"):
    p_ptls_pid, p_ptls_vel, p_ptls_pos = load_or_pickle_ptl_data(curr_sparta_file, str(p_snap), p_snapshot_path, p_scale_factor)

with timed("p_snap SPARTA load"):
    p_halos_pos, p_halos_r200m, p_halos_id, p_halos_status, p_halos_last_snap, mass = load_or_pickle_SPARTA_data(curr_sparta_file, p_scale_factor, p_snap, p_sparta_snap)

with timed("c_snap load"):
    t_dyn = calc_t_dyn(p_halos_r200m[np.where(p_halos_r200m > 0)[0][0]], p_red_shift)
    c_snap, c_sparta_snap, c_rho_m, c_red_shift, c_scale_factor, c_hubble_constant, c_ptls_pid, c_ptls_vel, c_ptls_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap = get_comp_snap(t_dyn=t_dyn, t_dyn_step=t_dyn_step, snapshot_list=[p_snap], cosmol = cosmol, p_red_shift=p_red_shift, all_red_shifts=all_red_shifts, snap_format=snap_format)
    c_box_size = sim_box_size * 10**3 * c_scale_factor #convert to Kpc/h physical

c_snap_dict = {
    "snap":c_snap,
    "red_shift":c_red_shift,
    "scale_factor": c_scale_factor,
    "hubble_const": c_hubble_constant,
    "box_size": c_box_size
}
del sparta_output
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
    match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_sparta_snap))[0]
else:
    match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_sparta_snap) & (c_halos_status > 0) & (c_halos_last_snap >= c_sparta_snap))[0]
    
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

with timed("p_snap initial search"):
    if os.path.isfile(save_location + "train_num_ptls.pickle"):
        with open(save_location + "train_num_ptls.pickle", "rb") as pickle_file:
            train_num_ptls = pickle.load(pickle_file)
    else:
        with mp.Pool(processes=num_processes) as p:
            # halo position, halo r200m, if comparison snap, want mass?, want indices?
            train_num_ptls = p.starmap(initial_search, zip(p_halos_pos[train_indices], p_halos_r200m[train_indices], repeat(False), repeat(False), repeat(False)), chunksize=curr_chunk_size)
            with open(save_location + "train_num_ptls.pickle", "wb") as pickle_file:
                pickle.dump(train_num_ptls, pickle_file)
        p.close()
        p.join() 

with timed("c_snap initial search"):
    if os.path.isfile(save_location + "test_num_ptls.pickle"):
        with open(save_location + "test_num_ptls.pickle", "rb") as pickle_file:
            test_num_ptls = pickle.load(pickle_file)
    else:
        with mp.Pool(processes=num_processes) as p:
            # halo position, halo r200m, if comparison snap, want mass?, want indices?
            test_num_ptls = p.starmap(initial_search, zip(p_halos_pos[test_indices], p_halos_r200m[test_indices], repeat(False), repeat(False), repeat(False)), chunksize=curr_chunk_size)
            with open(save_location + "test_num_ptls.pickle", "wb") as pickle_file:
                pickle.dump(test_num_ptls, pickle_file)
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
print("Start up finished in:",np.round((t2-t1),2),"seconds", np.round(((t2-t1)/60),2), "minutes")
halo_loop(train=True, indices=train_indices, tot_num_ptls=train_tot_num_ptls, p_halo_ids=p_halos_id, p_dict=p_snap_dict, p_ptls_pid=p_ptls_pid, p_ptls_pos=p_ptls_pos, p_ptls_vel=p_ptls_vel, c_dict=c_snap_dict, c_ptls_pid=c_ptls_pid, c_ptls_pos=c_ptls_pos, c_ptls_vel=c_ptls_vel)
halo_loop(train=False, indices=test_indices, tot_num_ptls=test_tot_num_ptls, p_halo_ids=p_halos_id, p_dict=p_snap_dict, p_ptls_pid=p_ptls_pid, p_ptls_pos=p_ptls_pos, p_ptls_vel=p_ptls_vel, c_dict=c_snap_dict, c_ptls_pid=c_ptls_pid, c_ptls_pos=c_ptls_pos, c_ptls_vel=c_ptls_vel)
t3 = time.time()
print("Finished in:",np.round((t3-t1),2),"seconds", np.round(((t3-t1)/60),2), "minutes")
