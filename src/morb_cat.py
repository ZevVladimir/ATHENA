import numpy as np
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from contextlib import contextmanager
import time
import h5py
import pickle
import os
import multiprocessing as mp
from itertools import repeat
import configparser
from utils.data_and_loading_functions import load_SPARTA_data, load_ptl_param, save_to_hdf5, conv_halo_id_spid, save_pickle, create_directory, find_closest_z
from utils.calculation_functions import *
##################################################################################################################
# LOAD CONFIG PARAMETERS
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
rand_seed = config.getint("MISC","random_seed")
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
path_to_hdf5_file = path_to_SPARTA_data + curr_sparta_file + ".hdf5"
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
create_directory(path_to_calc_info)
snap_format = config["MISC"]["snap_format"]
p_red_shift = config.getfloat("SEARCH","p_red_shift")
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
num_processes = mp.cpu_count()
curr_chunk_size = config.getint("SEARCH","chunk_size")
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
##################################################################################################################
import sys
sys.path.insert(0, path_to_sparta)
from sparta_tools import sparta
##################################################################################################################
@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))
    
def initial_search(halo_positions, halo_r200m, find_mass = False, find_ptl_indices = False):
    global search_rad
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
    
def search_halos(snap_dict, curr_halo_idx, curr_sparta_idx, curr_ptl_pids, curr_ptl_pos, curr_ptl_vel, 
                 halo_pos, halo_vel, halo_r200m, sparta_last_pericenter_snap=None, sparta_n_pericenter=None, sparta_tracer_ids=None,
                 sparta_n_is_lower_limit=None, find_subhalos=False, dens_prf_all=None, dens_prf_1halo=None, bins=None, create_dens_prf=False):
    # Doing this this way as otherwise will have to generate super large arrays for input from multiprocessing
    snap = snap_dict["snap"]
    scale_factor = snap_dict["scale_factor"]  
    red_shift = snap_dict["red_shift"]
    
    halo_pos = halo_pos * 10**3 * scale_factor
    
    num_new_ptls = curr_ptl_pids.shape[0]

    curr_ptl_pids = curr_ptl_pids.astype(np.int64) # otherwise ne.evaluate doesn't work
    
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
    
    m_orb = np.where(curr_orb_assn == 1)[0].shape[0] * mass  
    
    halo_m200m = mass_so.R_to_M(halo_r200m, red_shift, "200c")       
    halo_id = p_halos_id[curr_halo_idx]
    
    curr_orb_pid = curr_ptl_pids[np.where(curr_orb_assn == 1)[0]]

    if find_subhalos:
        subhalo_ids = p_halos_id[np.where(p_parent_id == halo_id)[0]]

        return halo_id, halo_pos, halo_vel, m_orb, halo_m200m, halo_r200m, curr_orb_pid, subhalo_ids
    return halo_id, halo_pos, halo_vel, m_orb, halo_m200m, halo_r200m, curr_orb_pid


def halo_loop(indices, p_halo_ids, p_dict, p_ptls_pid, p_ptls_pos, p_ptls_vel, find_subhalos=True):
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
        sparta_output = sparta.load(filename=path_to_hdf5_file, halo_ids=use_halo_ids, log_level=0)

        new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta re-sort the indices
        use_halo_idxs = use_indices[new_idxs]

        # Search around these halos and get the number of particles and the corresponding ptl indices for them
        p_use_halos_pos = sparta_output['halos']['position'][:,p_sparta_snap] * 10**3 * p_scale_factor 
        p_use_halos_r200m = sparta_output['halos']['R200m'][:,p_sparta_snap]

        with mp.Pool(processes=num_processes) as p:
            # halo position, halo r200m, if comparison snap, want mass?, want indices?
            p_use_num_ptls, p_curr_ptl_indices = zip(*p.starmap(initial_search, zip(p_use_halos_pos, p_use_halos_r200m, repeat(False), repeat(True)), chunksize=curr_chunk_size))
        p.close()
        p.join() 

        # Remove halos with 0 ptls around them
        p_use_num_ptls = np.array(p_use_num_ptls)
        p_curr_ptl_indices = np.array(p_curr_ptl_indices, dtype=object)
        has_ptls = np.where(p_use_num_ptls > 0)
        p_use_num_ptls = p_use_num_ptls[has_ptls]
        p_curr_ptl_indices = p_curr_ptl_indices[has_ptls]
        p_use_halo_idxs = use_halo_idxs[has_ptls]
       
        p_start_num_ptls = [np.sum(p_use_num_ptls[0:i+1]) for i in range(p_use_num_ptls.shape[0])]
        p_start_num_ptls = np.insert(p_start_num_ptls, 0, 0)
        p_start_num_ptls = np.delete(p_start_num_ptls, -1)
        p_start_num_ptls += hdf5_ptl_idx # scale to where we are in the hdf5 file
        
        p_tot_num_use_ptls = int(np.sum(p_use_num_ptls))

        halo_first = sparta_output['halos']['ptl_oct_first'][:]
        halo_n = sparta_output['halos']['ptl_oct_n'][:]
        
        # Use multiprocessing to search multiple halos at the same time and add information to shared arrays
        with mp.Pool(processes=num_processes) as p:
            if find_subhalos:
                all_halo_id, all_halo_pos, all_halo_vel, all_m_orb, all_halo_m200m, all_halo_r200m, all_orb_pid, all_subhalo_id = zip(*p.starmap(search_halos, 
                                            zip(repeat(p_dict), p_use_halo_idxs, np.arange(curr_num_halos),
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
                                            repeat(find_subhalos),
                                            # Uncomment below to create dens profiles
                                            #(sparta_output['anl_prf']['M_all'][l,p_sparta_snap,:] for l in range(curr_num_halos)),
                                            #(sparta_output['anl_prf']['M_1halo'][l,p_sparta_snap,:] for l in range(curr_num_halos)),
                                            #repeat(sparta_output["config"]['anl_prf']["r_bins_lin"]),repeat(True) 
                                            ),chunksize=curr_chunk_size))
            else:
                all_halo_id, all_halo_pos, all_halo_vel, all_m_orb, all_halo_m200m, all_halo_r200m, all_orb_pid = zip(*p.starmap(search_halos, 
                                            zip(repeat(p_dict), p_use_halo_idxs, np.arange(curr_num_halos),
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
                                            repeat(find_subhalos),
                                            # Uncomment below to create dens profiles
                                            #(sparta_output['anl_prf']['M_all'][l,p_sparta_snap,:] for l in range(curr_num_halos)),
                                            #(sparta_output['anl_prf']['M_1halo'][l,p_sparta_snap,:] for l in range(curr_num_halos)),
                                            #repeat(sparta_output["config"]['anl_prf']["r_bins_lin"]),repeat(True) 
                                            ),chunksize=curr_chunk_size))
        p.close()
        p.join()
        
        all_halo_id = np.array(all_halo_id)
        all_m_orb = np.array(all_m_orb)
        all_halo_m200m = np.array(all_halo_m200m)
        all_halo_r200m = np.array(all_halo_r200m)

        all_halo_pos = np.row_stack(all_halo_pos)
        all_halo_vel = np.row_stack(all_halo_vel)

        all_halo_id = all_halo_id.astype(np.int32)
        all_halo_pos = all_halo_pos.astype(np.float32)
        all_halo_vel = all_halo_vel.astype(np.float32)
        all_m_orb = all_m_orb.astype(np.float32)
        all_halo_m200m = all_halo_m200m.astype(np.float32)
        all_halo_r200m = all_halo_r200m.astype(np.float32)
        
        #TODO change this to use pandas dataframes
        if find_subhalos:
            if os.path.isfile(save_location + "catologue_" + curr_sparta_file + ".hdf5") and i == 0:
                os.remove(save_location + "catologue_" + curr_sparta_file + ".hdf5")
            with h5py.File(save_location + "member_catologue_" + curr_sparta_file + ".hdf5", 'a') as file:
                if "Halos" not in file:
                    file.create_group("Halos")
                for j,halo_id in enumerate(all_halo_id):
                    if str(halo_id) not in file["Halos"]:
                        file["Halos"].create_group(str(halo_id))
                
                    file["Halos"][str(halo_id)]['particle_ids'] = all_orb_pid[j]
                    file["Halos"][str(halo_id)]['sub_halo_ids'] = all_subhalo_id[j]
            with h5py.File((save_location + "catologue_" + curr_sparta_file + ".hdf5"), 'a') as file:
                if "Halos" not in file:
                    file.create_group("Halos")
                save_location + "catologue_" + curr_sparta_file + ".hdf5"
                save_to_hdf5(file["Halos"], "Halo_ID", dataset = all_halo_id, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_halo_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Halos"], "Halo_pos", dataset = all_halo_pos, chunk = True, max_shape = ((total_num_halos,3)), curr_idx = hdf5_halo_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Halos"], "Halo_vel", dataset = all_halo_vel, chunk = True, max_shape = ((total_num_halos,3)), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Halos"], "M_orb", dataset = all_m_orb, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Halos"], "M200m", dataset = all_halo_m200m, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Halos"], "R200m", dataset = all_halo_r200m, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)

        else:
            with h5py.File(save_location + "member_catologue_" + curr_sparta_file + ".hdf5", 'a') as file:
                if "Sub_Halos" not in file:
                    file.create_group("Sub_Halos")
                for j,halo_id in enumerate(all_halo_id):
                    if str(halo_id) not in file["Sub_Halos"]:
                        file["Sub_Halos"].create_group(str(halo_id))
                
                    file["Sub_Halos"][str(halo_id)]['particle_ids'] = all_orb_pid[j]
            with h5py.File((save_location + "catologue_" + curr_sparta_file + ".hdf5"), 'a') as file:
                if "Sub_Halos" not in file:
                    file.create_group("Sub_Halos")
                save_location + "catologue_" + curr_sparta_file + ".hdf5"
                save_to_hdf5(file["Sub_Halos"], "Halo_ID", dataset = all_halo_id, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_halo_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Sub_Halos"], "Halo_pos", dataset = all_halo_pos, chunk = True, max_shape = ((total_num_halos,3)), curr_idx = hdf5_halo_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Sub_Halos"], "Halo_vel", dataset = all_halo_vel, chunk = True, max_shape = ((total_num_halos,3)), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Sub_Halos"], "M_orb", dataset = all_m_orb, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Sub_Halos"], "M200m", dataset = all_halo_m200m, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)
                save_to_hdf5(file["Sub_Halos"], "R200m", dataset = all_halo_r200m, chunk = True, max_shape = (total_num_halos,), curr_idx = hdf5_ptl_idx, max_num_keys = num_save_ptl_params)

        
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
    print("corresponding SPARTA snap num:", p_sparta_snap,"sparta redshift:",all_red_shifts[p_sparta_snap])
    
    # Set constants
    p_snap_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

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
    p_ptls_pid = load_ptl_param(curr_sparta_file, "pid", str(p_snap), p_snap_path) * 10**3 * p_scale_factor # kpc/h
    p_ptls_vel = load_ptl_param(curr_sparta_file, "vel", str(p_snap), p_snap_path) # km/s
    p_ptls_pos = load_ptl_param(curr_sparta_file, "pos", str(p_snap), p_snap_path)

with timed("p_snap SPARTA load"):
    p_halos_pos, p_halos_r200m, p_halos_id, p_halos_status, p_halos_last_snap, p_parent_id, mass = load_SPARTA_data(curr_sparta_file, p_scale_factor, p_snap, p_sparta_snap)

save_location =  path_to_calc_info + curr_sparta_file + "_" + str(p_snap) + "_" + str(search_rad) + "r200msearch/"

if os.path.exists(save_location) != True:
    os.makedirs(save_location)

if os.path.isfile(save_location + "p_ptl_tree.pickle"):
        with open(save_location + "p_ptl_tree.pickle", "rb") as pickle_file:
            p_ptl_tree = pickle.load(pickle_file)
else:
    p_ptl_tree = cKDTree(data = p_ptls_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size) # construct search trees for primary snap
    save_pickle(p_ptl_tree,save_location + "p_ptl_tree.pickle")

# only take halos that are hosts in primary snap and exist past the p_snap and exist in some form at the comparison snap
match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_sparta_snap))[0]
  
total_num_halos = match_halo_idxs.shape[0]

num_halo_per_split = int(np.ceil(per_n_halo_per_split * total_num_halos))

with timed("p_snap initial search"):
    if os.path.isfile(save_location + "num_ptls.pickle"):
        with open(save_location + "num_ptls.pickle", "rb") as pickle_file:
            num_ptls = pickle.load(pickle_file)
    else:
        with mp.Pool(processes=num_processes) as p:
            # halo position, halo r200m, if comparison snap, want mass?, want indices?
            num_ptls = p.starmap(initial_search, zip(p_halos_pos[match_halo_idxs], p_halos_r200m[match_halo_idxs], repeat(False), repeat(False)), chunksize=curr_chunk_size)
            save_pickle(num_ptls,save_location + "num_ptls.pickle")

        p.close()
        p.join() 

tot_num_ptls = np.sum(num_ptls)

print("Total num halos:", total_num_halos, "Total num ptls:", tot_num_ptls)

if os.path.isfile(save_location + "member_catologue_" + curr_sparta_file + ".hdf5"):
    os.remove(save_location + "member_catologue_" + curr_sparta_file + ".hdf5")

with timed("Host Halos"):
    halo_loop(indices=match_halo_idxs, p_halo_ids=p_halos_id, p_dict=p_snap_dict, p_ptls_pid=p_ptls_pid, p_ptls_pos=p_ptls_pos, p_ptls_vel=p_ptls_vel)

# Now do the same but for subhalos
match_halo_idxs = np.where((p_halos_status == 20) & (p_halos_last_snap >= p_sparta_snap))[0]
total_num_halos = match_halo_idxs.shape[0]

num_halo_per_split = int(np.ceil(per_n_halo_per_split * total_num_halos))

print("Total num subhalos:", total_num_halos)

with timed("Sub halos"):
    halo_loop(indices=match_halo_idxs, p_halo_ids=p_halos_id, p_dict=p_snap_dict, p_ptls_pid=p_ptls_pid, p_ptls_pos=p_ptls_pos, p_ptls_vel=p_ptls_vel,find_subhalos=False)
t2 = time.time()
print("Finished in:",np.round((t2-t1),2),"seconds", np.round(((t2-t1)/60),2), "minutes")
