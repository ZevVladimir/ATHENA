import numexpr as ne
import numpy as np
from scipy.spatial import KDTree
from colossus.cosmology import cosmology
import h5py
import os
import multiprocessing as mp
from itertools import repeat
import re
import pandas as pd
import psutil
import argparse

from src.utils.util_fxns import load_SPARTA_data, load_ptl_param, get_comp_snap_info, create_directory, load_pickle, save_pickle, load_config, get_past_z
from src.utils.calc_fxns import calc_radius, calc_pec_vel, calc_rad_vel, calc_tang_vel, calc_rho, calc_halo_mem, calc_tdyn
from src.utils.prfl_fxns import create_mass_prf, compare_prfs
from utils.util_fxns import timed, clean_dir, find_closest_z_snap, get_num_snaps, split_sparta_hdf5_name
##################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default=os.getcwd() + "/config.ini", 
    help='Path to config file (default: config.ini)'
)

args = parser.parse_args()
config_params = load_config(args.config)

curr_sparta_file = config_params["SPARTA_DATA"]["curr_sparta_file"]
known_snaps = config_params["SNAP_DATA"]["known_snaps"]
snap_path = config_params["SNAP_DATA"]["snap_path"]
SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]
pickled_path = config_params["PATHS"]["pickled_path"]
ML_dset_path = config_params["PATHS"]["ml_dset_path"]
debug_plt_path = config_params["PATHS"]["debug_plt_path"]

random_seed = config_params["MISC"]["random_seed"]
debug_mem = config_params["MISC"]["debug_mem"]
debug_indiv_dens_prf = config_params["MISC"]["debug_indiv_dens_prf"]
sim_cosmol = config_params["MISC"]["sim_cosmol"]
reset_lvl = config_params["MISC"]["reset_search"]
mp_chunk_size = config_params["MISC"]["mp_chunk_size"]
save_intermediate_data = config_params["MISC"]["save_intermediate_data"]

snap_dir_format = config_params["SNAP_DATA"]["snap_dir_format"]
snap_format = config_params["SNAP_DATA"]["snap_format"]

input_z = config_params["DSET_CREATE"]["input_z"]
all_tdyn_steps = config_params["DSET_CREATE"]["tdyn_steps"]
search_radius = config_params["DSET_CREATE"]["search_radius"]
sub_dset_mem_size = config_params["DSET_CREATE"]["sub_dset_mem_size"]

test_dset_frac = config_params["DSET_CREATE"]["test_dset_frac"]
val_dset_frac = config_params["DSET_CREATE"]["val_dset_frac"]
lin_rticks = config_params["EVAL_MODEL"]["lin_rticks"]
##################################################################################################################
create_directory(pickled_path)
create_directory(ML_dset_path)
sparta_name, sparta_search_name = split_sparta_hdf5_name(curr_sparta_file)

snap_path = snap_path + sparta_name + "/"

# Set up exact paths
sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + curr_sparta_file + ".hdf5"

num_processes = mp.cpu_count()

##################################################################################################################
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def find_start_pnt(directory):
    max_number = 0
    number_pattern = re.compile(r'(\d+)')
    
    for filename in os.listdir(directory):
        match = number_pattern.search(filename)
        if match:
            number = int(match.group(1)) + 1
            if number > max_number:
                max_number = number
                
    return max_number

def det_halo_splits(mem_arr, mem_lim):
    split_idxs = []
    split_idxs.append(0)
    curr_size = 0 
    
    for i,mem in enumerate(mem_arr):
        curr_size += mem
        # Take into 128bytes for index in pandas dataframe
        if (curr_size + 128) > mem_lim:
            split_idxs.append(i)
            curr_size = 0
    return split_idxs

def assign_halo_dset(halo_nptl):
    if not 0 <= val_dset_frac < 1 or not 0 <= test_dset_frac < 1 or (val_dset_frac + test_dset_frac) >= 1:
        raise ValueError("val_dset_frac and test_dset_frac must be in [0, 1) and sum to < 1.")

    rng = np.random.default_rng(random_seed)
    total_halos = halo_nptl.shape[0]

    # Get sorted indices by halo size
    sorted_indices = np.argsort(halo_nptl)

    # Shuffle within size groups for balanced distribution
    group_size = max(10, total_halos // 50)  # Choose a group size (tune as needed)
    grouped_indices = [
        rng.permutation(sorted_indices[i:i+group_size])
        for i in range(0, total_halos, group_size)
    ]

    shuffled_indices = np.concatenate(grouped_indices)
    
    # Assign halos to test, val, train
    n_test = int(test_dset_frac * total_halos)
    n_val = int(val_dset_frac * total_halos)

    test_idx = shuffled_indices[:n_test]
    val_idx = shuffled_indices[n_test:n_test + n_val]
    train_idx = shuffled_indices[n_test + n_val:]

    return {
        'train_idx': np.sort(train_idx),
        'val_idx': np.sort(val_idx),
        'test_idx': np.sort(test_idx)
    }
            
def init_search(use_prim_tree, halo_positions, halo_r200m, search_rad, find_mass = False, find_ptl_indices = False):
    if use_prim_tree:
        ptl_tree = p_ptl_tree
    else:
        ptl_tree = curr_ptl_tree

    if halo_r200m > 0:
        #find how many particles we are finding
        indices = ptl_tree.query_ball_point(halo_positions, r = search_rad * halo_r200m)
        indices = np.array(indices)
        # how many new particles being added and correspondingly how massive the halo is
        num_new_particles = indices.shape[0]

        if find_mass:
            halo_mass = num_new_particles * ptl_mass
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
    
def search_halos(ret_labels, snap_dict, curr_halo_idx, curr_ptl_pids, curr_ptl_pos, curr_ptl_vel, 
                 halo_pos, halo_vel, halo_r200m, sparta_last_pericenter_snap=None, sparta_n_pericenter=None, sparta_tracer_ids=None,
                 sparta_n_is_lower_limit=None, act_mass_prf_all=None, act_mass_prf_orb=None, bins=None, create_dens_prf=False):
    # Doing this this way as otherwise will have to generate super large arrays for input from multiprocessing
    snap = snap_dict["sparta_snap"]
    red_shift = snap_dict["red_shift"]
    hubble_const = snap_dict["hubble_const"]
    box_size = snap_dict["box_size"] 
    little_h = snap_dict["h"]   
    
    num_new_ptls = curr_ptl_pids.shape[0]

    curr_ptl_pids = curr_ptl_pids.astype(np.int64) # otherwise ne.evaluate doesn't work
    fnd_HIPIDs = ne.evaluate("0.5 * (curr_ptl_pids + curr_halo_idx) * (curr_ptl_pids + curr_halo_idx + 1) + curr_halo_idx")
    
    #calculate the radii of each particle based on the distance formula
    ptl_rad, coord_dist = calc_radius(halo_pos[0], halo_pos[1], halo_pos[2], curr_ptl_pos[:,0], curr_ptl_pos[:,1], curr_ptl_pos[:,2], box_size)         
    
    if ret_labels:         
        compare_sparta_assn = np.zeros((sparta_tracer_ids.shape[0]))
        curr_orb_assn = np.zeros((num_new_ptls))
         # Anywhere sparta_last_pericenter is greater than the current snap then that is in the future so set to 0
        future_peri = np.where(sparta_last_pericenter_snap > snap)[0]
        sparta_n_pericenter[future_peri] = 0
        sparta_n_is_lower_limit[future_peri] = 0
        # If a particle has a pericenter or if the lower limit is 1 then it is orbiting
        compare_sparta_assn[np.where((sparta_n_pericenter >= 1) | (sparta_n_is_lower_limit == 1))[0]] = 1
        # compare_sparta_assn[np.where(adj_sparta_n_pericenter >= 1)] = 1
        
        # Compare the ids between SPARTA and the found ptl ids and match the SPARTA results
        matched_ids = np.intersect1d(curr_ptl_pids, sparta_tracer_ids, return_indices = True)
        curr_orb_assn[matched_ids[1]] = compare_sparta_assn[matched_ids[2]]

    # calculate peculiar, radial, and tangential velocity
    pec_vel = calc_pec_vel(curr_ptl_vel, halo_vel)
    fnd_rad_vel, curr_v200m, phys_vel, phys_vel_comp, rhat = calc_rad_vel(pec_vel, ptl_rad, coord_dist, halo_r200m, red_shift, hubble_const, little_h)
    fnd_tang_vel = calc_tang_vel(fnd_rad_vel, phys_vel_comp, rhat)
    
    scaled_rad_vel = fnd_rad_vel / curr_v200m
    scaled_tang_vel = fnd_tang_vel / curr_v200m
    scaled_radii = ptl_rad / halo_r200m
    scaled_phys_vel = phys_vel / curr_v200m
    
    scaled_radii_inds = scaled_radii.argsort()
    scaled_radii = scaled_radii[scaled_radii_inds]
    fnd_HIPIDs = fnd_HIPIDs[scaled_radii_inds]
    scaled_rad_vel = scaled_rad_vel[scaled_radii_inds]
    scaled_tang_vel = scaled_tang_vel[scaled_radii_inds]
    scaled_phys_vel = scaled_phys_vel[scaled_radii_inds]
    
    if ret_labels:
        curr_orb_assn = curr_orb_assn[scaled_radii_inds]
    
    if create_dens_prf and ret_labels:      
        calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, m200m = create_mass_prf(scaled_radii, curr_orb_assn, bins, ptl_mass)

        calc_dens_prf_all = calc_rho(calc_mass_prf_all, bins[1:], halo_r200m, np.array([0]), p_rho_m)
        calc_dens_prf_orb = calc_rho(calc_mass_prf_orb, bins[1:], halo_r200m, np.array([0]), p_rho_m)
        calc_dens_prf_inf = calc_rho(calc_mass_prf_inf, bins[1:], halo_r200m, np.array([0]), p_rho_m)
        
        act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb
        
        act_dens_prf_all = calc_rho(act_mass_prf_all, bins[1:], halo_r200m, np.array([0]), p_rho_m)
        act_dens_prf_orb = calc_rho(act_mass_prf_orb, bins[1:], halo_r200m, np.array([0]), p_rho_m)
        act_dens_prf_inf = calc_rho(act_mass_prf_inf, bins[1:], halo_r200m, np.array([0]), p_rho_m,print_test=True)

        all_prfs = [calc_dens_prf_all, act_dens_prf_all]
        orb_prfs = [calc_dens_prf_orb, act_dens_prf_orb]
        inf_prfs = [calc_dens_prf_inf, act_dens_prf_inf]

        compare_prfs(all_prfs,orb_prfs,inf_prfs,bins[1:],lin_rticks,debug_plt_path,str(curr_halo_idx),prf_func=None)

    if ret_labels:
        return fnd_HIPIDs, curr_orb_assn, scaled_rad_vel, scaled_tang_vel, scaled_radii, scaled_phys_vel
    else:
        return fnd_HIPIDs, scaled_rad_vel, scaled_tang_vel, scaled_radii, scaled_phys_vel

def halo_loop(ptl_idx, curr_iter, num_iter, indices, halo_splits, snap_dict, use_prim_tree, ptls_pid, ptls_pos, ptls_vel, ret_labels, name = "Train"):
        if debug_mem == 1:
            print(f"Initial memory usage: {memory_usage() / 1024**3:.2f} GB")
        
        curr_snap = snap_dict["ptl_snap"]
        curr_sparta_snap = snap_dict["sparta_snap"]
        curr_a = snap_dict["scale_factor"]

        with timed("Split "+str(curr_iter+1)+"/"+str(num_iter)):
            # Get the indices corresponding to where we are in the number of iterations (0:num_halo_persplit) -> (num_halo_persplit:2*num_halo_persplit) etc
            if curr_iter < (num_iter - 1):
                use_indices = indices[halo_splits[curr_iter]:halo_splits[curr_iter+1]]
            else:
                use_indices = indices[halo_splits[curr_iter]:]
           
            # Load the halo information for the ids within this range
            param_paths = [["halos","R200m"],["halos","position"],["halos","ptl_oct_first"],["halos","ptl_oct_n"],["halos","velocity"],["tcr_ptl","res_oct","last_pericenter_snap"],\
                ["tcr_ptl","res_oct","n_pericenter"],["tcr_ptl","res_oct","tracer_id"],["tcr_ptl","res_oct","n_is_lower_limit"],["anl_prf","M_all"],["anl_prf","M_1halo"],["config","anl_prf","r_bins_lin"]]
            sparta_params, sparta_param_names = load_SPARTA_data(sparta_HDF5_path, param_paths, sparta_search_name ,save_data=save_intermediate_data)

            use_halos_r200m = sparta_params[sparta_param_names[0]][use_indices,curr_sparta_snap]
            use_halos_pos = sparta_params[sparta_param_names[1]][use_indices,curr_sparta_snap] * 10**3 * curr_a
            halo_first = sparta_params[sparta_param_names[2]][use_indices][:]
            halo_n = sparta_params[sparta_param_names[3]][use_indices][:]
            use_halos_v = sparta_params[sparta_param_names[4]][use_indices,curr_sparta_snap]
            last_peri_snap = sparta_params[sparta_param_names[5]][:]
            n_peri = sparta_params[sparta_param_names[6]][:]
            tracer_id = sparta_params[sparta_param_names[7]][:]
            n_lower_lim = sparta_params[sparta_param_names[8]][:]
            mass_prf_all = sparta_params[sparta_param_names[9]][use_indices,curr_sparta_snap]
            mass_prf_orb = sparta_params[sparta_param_names[10]][use_indices,curr_sparta_snap]
            bins = sparta_params[sparta_param_names[11]]
            bins = np.insert(bins, 0, 0)

            with mp.Pool(processes=num_processes) as p:
                # halo position, halo r200m, if comparison snap, want mass?, want indices?
                use_num_ptls, curr_ptl_indices = zip(*p.starmap(init_search, zip(repeat(use_prim_tree), use_halos_pos, use_halos_r200m, \
                    repeat(search_radius), repeat(False), repeat(True)), chunksize=mp_chunk_size))
            p.close()
            p.join() 
            
            # Remove halos with 0 ptls around them
            use_num_ptls = np.array(use_num_ptls)
            curr_ptl_indices = np.array(curr_ptl_indices, dtype=object)
            # has_ptls = np.where(use_num_ptls > 0)[0]
            # use_num_ptls = use_num_ptls[has_ptls]
            # curr_ptl_indices = curr_ptl_indices[has_ptls]
            # use_indices = use_indices[has_ptls]
            curr_num_halos = curr_ptl_indices.shape[0]

            # We need to correct for having previous halos being searched so the final halo_first quantity is for all halos
            # First obtain the halo_first values for this batch and then adjust to where the hdf5 file currently is
            start_num_ptls = np.cumsum(use_num_ptls)
            start_num_ptls = np.insert(start_num_ptls, 0, 0)
            start_num_ptls = np.delete(start_num_ptls, -1)
            start_num_ptls += ptl_idx # scale to where we are in the hdf5 file           
            
            # Use multiprocessing to search multiple halos at the same time and add information to shared array
            create_dens_prfs = np.zeros(curr_num_halos)
            if debug_indiv_dens_prf:
                create_dens_prfs[:1] = 1
                
            with mp.Pool(processes=num_processes) as p:
                results = tuple(zip(*p.starmap(search_halos, 
                               zip(repeat(ret_labels), repeat(snap_dict), use_indices,
                                   (ptls_pid[curr_ptl_indices[i].astype(int)] for i in range(curr_num_halos)), 
                                   (ptls_pos[curr_ptl_indices[j].astype(int)] for j in range(curr_num_halos)),
                                   (ptls_vel[curr_ptl_indices[k].astype(int)] for k in range(curr_num_halos)),
                                   (use_halos_pos[l,:] for l in range(curr_num_halos)),
                                   (use_halos_v[l,:] for l in range(curr_num_halos)),
                                   (use_halos_r200m[l] for l in range(curr_num_halos)),
                                   (last_peri_snap[halo_first[m]:halo_first[m]+halo_n[m]] for m in range(curr_num_halos)),
                                   (n_peri[halo_first[m]:halo_first[m]+halo_n[m]] for m in range(curr_num_halos)),
                                   (tracer_id[halo_first[m]:halo_first[m]+halo_n[m]] for m in range(curr_num_halos)),
                                   (n_lower_lim[halo_first[m]:halo_first[m]+halo_n[m]] for m in range(curr_num_halos)),
                                   (mass_prf_all[l,:] for l in range(curr_num_halos)),
                                   (mass_prf_orb[l,:] for l in range(curr_num_halos)),
                                   repeat(bins),
                                   (create_dens_prfs[l] for l in range(curr_num_halos)),
                                   ),
                               chunksize=mp_chunk_size)))

                if ret_labels:
                    (all_HIPIDs, all_orb_assn, all_rad_vel, all_tang_vel, all_scal_rad, all_phys_vel) = results
                else:
                    (all_HIPIDs, all_rad_vel, all_tang_vel, all_scal_rad, all_phys_vel) = results

            p.close()
            p.join()
            
            all_HIPIDs = np.concatenate(all_HIPIDs, axis = 0)
            all_rad_vel = np.concatenate(all_rad_vel, axis = 0)
            all_tang_vel = np.concatenate(all_tang_vel, axis = 0)
            all_scal_rad = np.concatenate(all_scal_rad, axis = 0)
            all_phys_vel = np.concatenate(all_phys_vel, axis=0)
            
            all_HIPIDs = all_HIPIDs.astype(np.float64)
            all_rad_vel = all_rad_vel.astype(np.float32)
            all_tang_vel = all_tang_vel.astype(np.float32)
            all_scal_rad = all_scal_rad.astype(np.float32)
            all_phys_vel = all_phys_vel.astype(np.float32)

            if ret_labels:
                all_orb_assn = np.concatenate(all_orb_assn, axis = 0)
                all_orb_assn = all_orb_assn.astype(np.int8)
                return all_HIPIDs, all_orb_assn, all_rad_vel, all_tang_vel, all_scal_rad, all_phys_vel, start_num_ptls, use_num_ptls, use_indices, use_halos_r200m
            else:
                return all_HIPIDs, all_rad_vel, all_tang_vel, all_scal_rad, all_phys_vel

with timed("Generating Datasets for " + curr_sparta_file):
    with timed("Startup"):
        if reset_lvl <= 1:
            if len(known_snaps) > 0:
                save_location = os.path.join(ML_dset_path, curr_sparta_file + "_" + "_".join(str(x) for x in known_snaps)) + "/"
                dset_params = load_pickle(os.path.join(save_location, "dset_params.pickle"))
            else:
                raise ValueError("To load the correct parameter file, provide the known snapshots. If the snapshots are unknown, set reset_lvl > 1.")


        if reset_lvl <= 1:
            sim_cosmol = dset_params["cosmology"]
        if sim_cosmol == "planck13-nbody":
            cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
        else:
            cosmol = cosmology.setCosmology(sim_cosmol) 

        all_snap_info = {}
        all_ptl_snap_list = []
        
        with timed("Primary Snapshot Information Load"):
            if reset_lvl > 1:
                prime_snap_dict = {}
                p_snap, prim_z = find_closest_z_snap(input_z,snap_path,snap_dir_format,snap_format)
                print("Snapshot number found:", p_snap, "Closest redshift found:", prim_z)
                
                with h5py.File(sparta_HDF5_path,"r") as f:
                    dic_sim = {}
                    grp_sim = f['simulation']
                    for f in grp_sim.attrs:
                        dic_sim[f] = grp_sim.attrs[f]
                    
                all_sparta_z = dic_sim['snap_z']
                p_sparta_snap = np.abs(all_sparta_z - prim_z).argmin()
                
                print("corresponding SPARTA snap num:", p_sparta_snap)
                print("check sparta redshift:",all_sparta_z[p_sparta_snap])   
                
                prime_a = 1/(1+prim_z)
                p_rho_m = cosmol.rho_m(prim_z) # units of M_sun * h^2 * kpc^-3
                p_hubble_const = cosmol.Hz(prim_z) / 1e3 # convert from km/s/Mpc to units km/s/kpc
                sim_box_size = dic_sim["box_size"] #units Mpc/h comoving
                p_box_size = sim_box_size * 10**3 * prime_a #convert to Kpc/h physical
                little_h = dic_sim["h"] # units of km/s/Mpc
                prime_snap_dict = {
                    "ptl_snap":p_snap,
                    "sparta_snap":p_sparta_snap,
                    "red_shift":prim_z,
                    "scale_factor": prime_a,
                    "hubble_const": p_hubble_const,
                    "box_size": p_box_size,
                    "h":little_h,
                    "rho_m":p_rho_m
                }
            else:
                prime_snap_dict = dset_params["all_snap_info"]["prime_snap_info"]
                p_snap = dset_params["all_snap_info"]["prime_snap_info"]["ptl_snap"]
                prim_z = dset_params["all_snap_info"]["prime_snap_info"]["red_shift"]
                p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"]
                prime_a = dset_params["all_snap_info"]["prime_snap_info"]["scale_factor"]
                p_hubble_const = dset_params["all_snap_info"]["prime_snap_info"]["hubble_const"]
                p_box_size = dset_params["all_snap_info"]["prime_snap_info"]["box_size"]
                little_h = dset_params["all_snap_info"]["prime_snap_info"]["h"]

            all_ptl_snap_list.append(p_snap)
            p_snap_path = snap_path + "snapdir_" + snap_dir_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)
            all_snap_info["prime_snap_info"] = prime_snap_dict
            
        if reset_lvl == 3:
            clean_dir(pickled_path + str(p_snap) + "_" + curr_sparta_file + "/")

        # load all information needed for the primary snap
        with timed("p_snap ptl load"):
            p_ptls_pid = load_ptl_param(curr_sparta_file, "pid", str(p_snap), p_snap_path, save_data=save_intermediate_data) 
            p_ptls_vel = load_ptl_param(curr_sparta_file, "vel", str(p_snap), p_snap_path, save_data=save_intermediate_data) # km/s
            p_ptls_pos = load_ptl_param(curr_sparta_file, "pos", str(p_snap), p_snap_path, save_data=save_intermediate_data) * 10**3 * prime_a # kpc/h
        
        with timed("p_snap SPARTA load"):
            param_paths = [["halos","position"],["halos","R200m"],["halos","id"],["halos","status"],["halos","last_snap"],["simulation","particle_mass"]]
                
            sparta_params, sparta_param_names = load_SPARTA_data(sparta_HDF5_path, param_paths, curr_sparta_file, save_data=save_intermediate_data)

            p_halos_pos = sparta_params[sparta_param_names[0]][:,p_sparta_snap,:] * 10**3 * prime_a # convert to kpc/h
            p_halos_r200m = sparta_params[sparta_param_names[1]][:,p_sparta_snap]
            p_halos_ids = sparta_params[sparta_param_names[2]][:,p_sparta_snap]
            p_halos_status = sparta_params[sparta_param_names[3]][:,p_sparta_snap]
            p_halos_last_snap = sparta_params[sparta_param_names[4]][:]
            ptl_mass = sparta_params[sparta_param_names[5]]
            
        with timed("Load Complementary Snapshots"):
            if reset_lvl > 1 or len(known_snaps) == 0:
                for tdyn_step in all_tdyn_steps:
                    tdyn = calc_tdyn(p_halos_r200m[np.where(p_halos_r200m > 0)[0][0]],prim_z,little_h)
                    curr_time = cosmol.age(prim_z)
                    past_time = curr_time - (tdyn * 1)
                    past_z = cosmol.age(past_time,inverse=True)
                    print("Calculated past redshift formula 0.5 tdyn:",past_z)
                                        
                    past_z = get_past_z(cosmol, prim_z, tdyn_step=tdyn_step)
                    print("Calculated past redshift colossus 1 tdyn:",past_z)
                    c_snap_dict = get_comp_snap_info(cosmol = cosmol, past_z=past_z, all_sparta_z=all_sparta_z,snap_dir_format=snap_dir_format,snap_format=snap_format,snap_path=snap_path)
                    
                    c_snap = c_snap_dict["ptl_snap"]
                    c_sparta_snap = c_snap_dict["sparta_snap"]
                    comp_z = c_snap_dict["red_shift"]
                    comp_a = c_snap_dict["scale_factor"]
                    c_hubble_const = c_snap_dict["hubble_const"]
                    c_rho_m = c_snap_dict["rho_m"]
                    c_box_size = sim_box_size * 10**3 * comp_a #convert to Kpc/h physical
                    
                    c_snap_dict = {
                        "ptl_snap":c_snap,
                        "sparta_snap":c_sparta_snap,
                        "red_shift":comp_z,
                        "t_dyn_step":tdyn_step,
                        "scale_factor": comp_a,
                        "hubble_const": c_hubble_const,
                        "box_size": c_box_size,
                        "h":little_h,
                        "rho_m":c_rho_m
                    }
                    all_ptl_snap_list.append(c_snap)
                    all_snap_info["comp_" + str(tdyn_step) + "_tdstp_snap_info"] = c_snap_dict
            else:
                for key, c_snap_dict in dset_params["all_snap_info"].items():
                    if key == "prime_snap_info":
                        continue
                    c_snap = c_snap_dict["ptl_snap"]
                    c_sparta_snap = c_snap_dict["sparta_snap"]
                    tdyn_step = c_snap_dict["t_dyn_step"],
                    comp_z = c_snap_dict["red_shift"]
                    comp_a = c_snap_dict["scale_factor"]
                    c_hubble_const = c_snap_dict["hubble_const"]
                    c_box_size = c_snap_dict["box_size"]
                    all_ptl_snap_list.append(c_snap)

                    all_snap_info["comp_" + str(tdyn_step[0]) + "_tdstp_snap_info"] = c_snap_dict
        if len(all_tdyn_steps) > 0:
            c_halos_status = sparta_params[sparta_param_names[3]][:,c_sparta_snap]

        all_ptl_snap_list.sort()
        all_ptl_snap_list.reverse()

        save_location = ML_dset_path + curr_sparta_file + "_" + "_".join(str(x) for x in all_ptl_snap_list) + "/"
        
        create_directory(save_location)

        if os.path.isfile(save_location + "p_ptl_tree.pickle") and reset_lvl < 2:
                p_ptl_tree = load_pickle(save_location + "p_ptl_tree.pickle")
        else:
            p_ptl_tree = KDTree(data = p_ptls_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size) # construct search trees for primary snap
            if save_intermediate_data:
                save_pickle(p_ptl_tree,save_location + "p_ptl_tree.pickle")

        if os.path.isfile(save_location + "num_ptls.pickle") and os.path.isfile(save_location + "match_halo_idxs.pickle") and reset_lvl < 2:
            num_ptls = load_pickle(save_location + "num_ptls.pickle")
            match_halo_idxs = load_pickle(save_location + "match_halo_idxs.pickle")
        else:
            # only take halos that are hosts in primary snap and exist past the p_snap 
            
            match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_sparta_snap))[0]
            
            with mp.Pool(processes=num_processes) as p:           
                # halo position, halo r200m, if comparison snap, want mass?, want indices?
                num_ptls = p.starmap(init_search, zip(repeat(True), p_halos_pos[match_halo_idxs], p_halos_r200m[match_halo_idxs], repeat(1.0), repeat(False), repeat(False)), chunksize=mp_chunk_size)
                
                # We want to remove any halos that have less than 200 particles as they are too noisy
                num_ptls = np.array(num_ptls)
                res_mask = np.where(num_ptls >= 200)[0]

                if res_mask.size > 0:
                    num_ptls = num_ptls[res_mask]
                    match_halo_idxs = match_halo_idxs[res_mask]
                
                save_pickle(num_ptls,save_location + "num_ptls.pickle")
                save_pickle(match_halo_idxs,save_location + "match_halo_idxs.pickle")
            
            if len(all_tdyn_steps) > 0:
                orig_match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_sparta_snap) & (c_halos_status > 0))[0]
                with mp.Pool(processes=num_processes) as p:           
                    # halo position, halo r200m, if comparison snap, want mass?, want indices?
                    test_num_ptls = p.starmap(init_search, zip(repeat(True), p_halos_pos[orig_match_halo_idxs], p_halos_r200m[orig_match_halo_idxs], repeat(1.0), repeat(False), repeat(False)), chunksize=mp_chunk_size)
                    
                    # We want to remove any halos that have less than 200 particles as they are too noisy
                    test_num_ptls = np.array(test_num_ptls)
                    test_res_mask = np.where(test_num_ptls >= 200)[0]

                    if test_res_mask.size > 0:
                        orig_match_halo_idxs = orig_match_halo_idxs[test_res_mask]
                
                symmetric_diff = np.setxor1d(orig_match_halo_idxs, match_halo_idxs)
                print("Num of halos that didn't exist before:", symmetric_diff.shape)
                
                p.close()
                p.join() 
            
        tot_num_ptls = np.sum(num_ptls)   
        total_num_halos = match_halo_idxs.shape[0]
        
        #TODO split the halos more intelligently so that halo sizes are evenly distributed
        #TODO create option to also create a validation set
        # split all indices into train and test groups
        split_pnt = int((1-test_dset_frac) * total_num_halos)
        train_idxs = match_halo_idxs[:split_pnt]
        test_idxs = match_halo_idxs[split_pnt:]
        train_num_ptls = num_ptls[:split_pnt]
        test_num_ptls = num_ptls[split_pnt:]

        # need to sort indices otherwise sparta.load breaks...
        train_idxs_inds = train_idxs.argsort()
        train_idxs = train_idxs[train_idxs_inds]
        train_num_ptls = train_num_ptls[train_idxs_inds]
        
        test_idxs_inds = test_idxs.argsort()
        test_idxs = test_idxs[test_idxs_inds]
        test_num_ptls = test_num_ptls[test_idxs_inds]
        
        train_halo_mem = calc_halo_mem(train_num_ptls)
        test_halo_mem = calc_halo_mem(test_num_ptls)

        train_halo_splits = det_halo_splits(train_halo_mem, sub_dset_mem_size)
        test_halo_splits = det_halo_splits(test_halo_mem, sub_dset_mem_size)
        
        print(f"Total num halos: {total_num_halos:.3e}")
        print(f"Total num ptls: {tot_num_ptls:.3e}")

        tot_num_snaps = get_num_snaps(snap_path)
        
        dset_params = {
            "sparta_file": curr_sparta_file,
            "snap_dir_format":snap_dir_format,
            "snap_format": snap_format,
            "cosmology": sim_cosmol,
            "t_dyn_steps": all_tdyn_steps,
            "search_rad": search_radius,
            "total_num_snaps": tot_num_snaps,
            "test_halos_ratio": test_dset_frac,
            "chunk_size": mp_chunk_size,
            "HDF5 Mem Size": sub_dset_mem_size,
            "all_snap_info": all_snap_info,
        }

        save_pickle(dset_params,save_location+"dset_params.pickle")
            
    with timed("Creating Datasets"):   
        create_directory(save_location + "Train/halo_info/")
        create_directory(save_location + "Test/halo_info/")
        
        for i,curr_ptl_snap in enumerate(all_ptl_snap_list):
            ptl_idx = 0
            
            create_directory(save_location + "Train/ptl_info/"+str(curr_ptl_snap)+"/")
            create_directory(save_location + "Test/ptl_info/"+str(curr_ptl_snap)+"/")
            if reset_lvl > 0: # At any level of reset we delete the calculated info for particles
                clean_dir(save_location + "Train/ptl_info/"+str(curr_ptl_snap)+"/")
                clean_dir(save_location + "Test/ptl_info/"+str(curr_ptl_snap)+"/")
                train_start_pnt=0
                test_start_pnt=0
                curr_train_halo_splits = train_halo_splits.copy()
                curr_test_halo_splits = test_halo_splits.copy()
            else: #Otherwise check to see where we were and then continue the calculations from there.
                train_start_pnt = find_start_pnt(save_location + "Train/ptl_info/"+str(curr_ptl_snap)+"/")
                test_start_pnt = find_start_pnt(save_location + "Test/ptl_info/"+str(curr_ptl_snap)+"/")
                curr_train_halo_splits = train_halo_splits[train_start_pnt:]
                curr_test_halo_splits = test_halo_splits[test_start_pnt:]

            prnt_halo_splits = curr_train_halo_splits.copy()
            prnt_halo_splits.append(len(train_idxs))        
            
            for k in range(len(prnt_halo_splits)):
                if k < len(prnt_halo_splits) - 1:
                    value_gb = ((np.sum(train_halo_mem[prnt_halo_splits[k]:prnt_halo_splits[k+1]])) + 128) * 1e-9
                    print(f"{value_gb:.2f} GB")

        
            curr_snap_path = snap_path + "/snapdir_" + snap_dir_format.format(curr_ptl_snap) + "/snapshot_" + snap_format.format(curr_ptl_snap)
            
            if i == 0:
                curr_snap_dict = dset_params["all_snap_info"]["prime_snap_info"]
                curr_ptls_pid = p_ptls_pid
                curr_ptls_vel = p_ptls_vel
                curr_ptls_pos = p_ptls_pos
                curr_ptl_tree = p_ptl_tree
            else:
                curr_snap_dict = dset_params["all_snap_info"]["comp_" + str(all_tdyn_steps[i-1]) + "_tdstp_snap_info"]
            curr_sparta_snap = curr_snap_dict["sparta_snap"]
            curr_z = curr_snap_dict["red_shift"]
            curr_a = curr_snap_dict["scale_factor"]
            curr_hubble_const = curr_snap_dict["hubble_const"]
            curr_box_size = curr_snap_dict["box_size"]
        
            if reset_lvl == 3:
                clean_dir(pickled_path + str(curr_ptl_snap) + "_" + curr_sparta_file + "/")       
                
            if i > 0:
                # load particle data for the comparison snap
                with timed("c_snap ptl load"):
                    curr_ptls_pid = load_ptl_param(curr_sparta_file, "pid", str(curr_ptl_snap), curr_snap_path, save_data=save_intermediate_data) 
                    curr_ptls_vel = load_ptl_param(curr_sparta_file, "vel", str(curr_ptl_snap), curr_snap_path, save_data=save_intermediate_data) # km/s
                    curr_ptls_pos = load_ptl_param(curr_sparta_file, "pos", str(curr_ptl_snap), curr_snap_path, save_data=save_intermediate_data) * 10**3 * curr_a # kpc/h
            
                if os.path.isfile(save_location + str(curr_ptl_snap) + "_ptl_tree.pickle") and reset_lvl < 2:
                    curr_ptl_tree = load_pickle(save_location + str(curr_ptl_snap) + "_ptl_tree.pickle")
                else:
                    curr_ptl_tree = KDTree(data = curr_ptls_pos, leafsize = 3, balanced_tree = False, boxsize = curr_box_size)
                    if save_intermediate_data:
                        save_pickle(curr_ptl_tree, save_location +  str(curr_ptl_snap) + "_ptl_tree.pickle")        
    
            train_num_iter = len(curr_train_halo_splits)
            train_prnt_halo_splits = curr_train_halo_splits.copy()
            train_prnt_halo_splits.append(train_idxs.size)
            print("Train Splits")
            print("Num halos in each split:", ", ".join(map(str, np.diff(train_prnt_halo_splits))) + ".", train_num_iter, "splits")

            for j in range(train_num_iter):
                if i == 0:
                    ret_labels = True
                    train_p_HIPIDs, p_orb_assn, p_rad_vel, p_tang_vel, p_scale_rad, p_phys_vel, p_start_num_ptls, p_use_num_ptls, use_indices, use_halos_r200m = halo_loop(ptl_idx=ptl_idx, curr_iter=j, num_iter=train_num_iter, \
                        indices=train_idxs, halo_splits=curr_train_halo_splits, snap_dict=curr_snap_dict, use_prim_tree=True, ptls_pid=curr_ptls_pid, \
                        ptls_pos=curr_ptls_pos, ptls_vel=curr_ptls_vel, ret_labels=ret_labels)
                    
                    p_train_nptls = train_p_HIPIDs.shape[0]
                    
                    halo_df = pd.DataFrame({
                        "Halo_first":p_start_num_ptls,
                        "Halo_n":p_use_num_ptls,
                        "Halo_indices":use_indices,
                        "Halo_R200m":use_halos_r200m,
                    })
                    halo_df.to_hdf(save_location + "Train/halo_info/halo_" + str(j+train_start_pnt) + ".h5", key='data', mode='w',format='table')  

                    ptl_df = pd.DataFrame({
                        "HIPIDS":train_p_HIPIDs,
                        "Orbit_infall":p_orb_assn,
                        "p_Scaled_radii":p_scale_rad,
                        "p_Radial_vel":p_rad_vel,
                        "p_Tangential_vel":p_tang_vel,
                        "p_phys_vel":p_phys_vel
                    })
                else:                   
                    ret_labels = False
                    c_HIPIDs, c_rad_vel, c_tang_vel, c_scale_rad, c_phys_vel = halo_loop(ptl_idx=ptl_idx, curr_iter=j, num_iter=train_num_iter, indices=train_idxs, \
                        halo_splits=curr_train_halo_splits, snap_dict=curr_snap_dict, use_prim_tree=False, ptls_pid=curr_ptls_pid, ptls_pos=curr_ptls_pos, ptls_vel=curr_ptls_vel, ret_labels=ret_labels)
                    
                    # We load the comparison HIPIDs to deal with cases where the present snap data got completed before the past snap data
                    curr_train_p_HIPIDs = pd.read_hdf(save_location + "Train/ptl_info/" + str(p_snap) + "/ptl_" + str(j) + ".h5", key='data', columns=['HIPIDS'])
                    match_hipid_idx = np.intersect1d(curr_train_p_HIPIDs, c_HIPIDs, return_indices=True)

                    p_train_nptls = curr_train_p_HIPIDs.shape[0]

                    save_scale_rad = np.zeros((p_train_nptls,), dtype = np.float32)
                    save_rad_vel = np.zeros((p_train_nptls,), dtype = np.float32)
                    save_tang_vel = np.zeros((p_train_nptls,), dtype = np.float32)
                    save_phys_vel = np.zeros((p_train_nptls,), dtype = np.float32)
                    save_scale_rad[match_hipid_idx[1]] = c_scale_rad[match_hipid_idx[2]]
                    save_rad_vel[match_hipid_idx[1]] = c_rad_vel[match_hipid_idx[2]]
                    save_tang_vel[match_hipid_idx[1]] = c_tang_vel[match_hipid_idx[2]]
                    save_phys_vel[match_hipid_idx[1]] = c_phys_vel[match_hipid_idx[2]]
                    
                    save_scale_rad[save_scale_rad == 0] = np.nan
                    save_rad_vel[save_rad_vel == 0] = np.nan
                    save_tang_vel[save_tang_vel == 0] = np.nan
                    save_phys_vel[save_phys_vel == 0] = np.nan

                    ptl_df = pd.DataFrame({
                        str(all_tdyn_steps[i-1]) + "_Scaled_radii":save_scale_rad,
                        str(all_tdyn_steps[i-1]) + "_Radial_vel":save_rad_vel,
                        str(all_tdyn_steps[i-1]) + "_Tangential_vel":save_tang_vel,
                        str(all_tdyn_steps[i-1]) + "_phys_vel":save_phys_vel
                    })
                ptl_df.to_hdf(save_location + "Train/ptl_info/" + str(curr_ptl_snap) + "/ptl_" + str(j+train_start_pnt) + ".h5", key='data', mode='w',format='table')  
                ptl_idx += p_train_nptls
                if debug_mem == 1:
                    print(f"Final memory usage: {memory_usage() / 1024**3:.2f} GB")
                
            test_num_iter = len(curr_test_halo_splits)
            test_prnt_halo_splits = curr_test_halo_splits.copy()
            test_prnt_halo_splits.append(test_idxs.size)
            print("Test Splits")
            print("Num halos in each split:", ", ".join(map(str, np.diff(test_prnt_halo_splits))) + ".", test_num_iter, "splits")
            ptl_idx = 0
            
            for j in range(test_num_iter):
                if i == 0:
                    ret_labels = True
                    test_p_HIPIDs, p_orb_assn, p_rad_vel, p_tang_vel, p_scale_rad, p_phys_vel, p_start_num_ptls, p_use_num_ptls, use_indices, use_halos_r200m = halo_loop(ptl_idx=ptl_idx, curr_iter=j, num_iter=test_num_iter, \
                        indices=test_idxs, halo_splits=curr_test_halo_splits, snap_dict=curr_snap_dict, use_prim_tree=True, ptls_pid=curr_ptls_pid, \
                        ptls_pos=curr_ptls_pos, ptls_vel=curr_ptls_vel, ret_labels=ret_labels)
                    
                    p_test_nptls = test_p_HIPIDs.shape[0]
                    
                    halo_df = pd.DataFrame({
                        "Halo_first":p_start_num_ptls,
                        "Halo_n":p_use_num_ptls,
                        "Halo_indices":use_indices,
                        "Halo_R200m":use_halos_r200m,
                    })
                    halo_df.to_hdf(save_location + "Test/halo_info/halo_" + str(j+test_start_pnt) + ".h5", key='data', mode='w',format='table')  

                    ptl_df = pd.DataFrame({
                        "HIPIDS":test_p_HIPIDs,
                        "Orbit_infall":p_orb_assn,
                        "p_Scaled_radii":p_scale_rad,
                        "p_Radial_vel":p_rad_vel,
                        "p_Tangential_vel":p_tang_vel,
                        "p_phys_vel":p_phys_vel
                    })
                else:
                    # Access the correct number of particles for this batch 
                    ret_labels = False
                    c_HIPIDs, c_rad_vel, c_tang_vel, c_scale_rad, c_phys_vel = halo_loop(ptl_idx=ptl_idx, curr_iter=j, num_iter=test_num_iter, indices=test_idxs, \
                        halo_splits=curr_test_halo_splits, snap_dict=curr_snap_dict, use_prim_tree=False, ptls_pid=curr_ptls_pid, ptls_pos=curr_ptls_pos, ptls_vel=curr_ptls_vel, ret_labels=ret_labels,name="Test")

                    curr_test_p_HIPIDs = pd.read_hdf(save_location + "Test/ptl_info/" + str(p_snap) + "/ptl_" + str(j) + ".h5", key='data', columns=['HIPIDS'])
                    match_hipid_idx = np.intersect1d(curr_test_p_HIPIDs, c_HIPIDs, return_indices=True)
                    
                    p_test_nptls = curr_test_p_HIPIDs.shape[0]
                    
                    save_scale_rad = np.zeros((p_test_nptls,), dtype = np.float32)
                    save_rad_vel = np.zeros((p_test_nptls,), dtype = np.float32)
                    save_tang_vel = np.zeros((p_test_nptls,), dtype = np.float32)
                    save_phys_vel = np.zeros((p_test_nptls,), dtype = np.float32)
                    save_scale_rad[match_hipid_idx[1]] = c_scale_rad[match_hipid_idx[2]]
                    save_rad_vel[match_hipid_idx[1]] = c_rad_vel[match_hipid_idx[2]]
                    save_tang_vel[match_hipid_idx[1]] = c_tang_vel[match_hipid_idx[2]]
                    save_phys_vel[match_hipid_idx[1]] = c_phys_vel[match_hipid_idx[2]]
                    
                    save_scale_rad[save_scale_rad == 0] = np.nan
                    save_rad_vel[save_rad_vel == 0] = np.nan
                    save_tang_vel[save_tang_vel == 0] = np.nan
                    save_phys_vel[save_phys_vel == 0] = np.nan
                    
                    ptl_df = pd.DataFrame({
                        str(all_tdyn_steps[i-1]) + "_Scaled_radii":save_scale_rad,
                        str(all_tdyn_steps[i-1]) + "_Radial_vel":save_rad_vel,
                        str(all_tdyn_steps[i-1]) + "_Tangential_vel":save_tang_vel,
                        str(all_tdyn_steps[i-1]) + "_phys_vel":save_phys_vel
                    })
                
                ptl_df.to_hdf(save_location + "Test/ptl_info/" + str(curr_ptl_snap) + "/ptl_" + str(j+test_start_pnt) + ".h5", key='data', mode='w',format='table')  
                ptl_idx += p_test_nptls
                if debug_mem == 1:
                    print(f"Final memory usage: {memory_usage() / 1024**3:.2f} GB")

                
