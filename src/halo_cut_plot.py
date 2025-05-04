import os
import json
import random
import pickle
import re
from scipy.spatial import cKDTree
import numpy as np
from sparta_tools import sparta

from utils.ML_support import get_combined_name,reform_dataset_dfs,split_sparta_hdf5_name,get_model_name
from src.utils.vis_fxns import plot_halo_slice
from utils.data_and_loading_functions import create_directory,load_SPARTA_data,timed,load_ptl_param,load_config
from utils.calculation_functions import calc_radius, nptl_inc_placement_r200m

dset_params = load_config(os.getcwd() + "/config.ini")

snap_path = dset_params["SNAP_DATA"]["snap_path"]
SPARTA_output_path = dset_params["SPARTA_DATA"]["sparta_output_path"]
ML_dset_path = dset_params["PATHS"]["ml_dset_path"]
path_to_models = dset_params["PATHS"]["path_to_models"]

curr_sparta_file = dset_params["SPARTA_DATA"]["curr_sparta_file"]
snap_dir_format = dset_params["SNAP_DATA"]["snap_dir_format"]
snap_format = dset_params["SNAP_DATA"]["snap_format"]
sim_cosmol = dset_params["MISC"]["sim_cosmol"]

search_radius = dset_params["DSET_CREATE"]["search_radius"]
test_sims = dset_params["EVAL_MODEL"]["test_sims"]
model_sims = dset_params["TRAIN_MODEL"]["model_sims"]
model_type = dset_params["TRAIN_MODEL"]["model_type"]

sim_name, search_name = split_sparta_hdf5_name(curr_sparta_file)

snap_path = snap_path + sim_name + "/"

if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
else:
    sim_pat = r"cbol_l(\d+)_n(\d+)"
match = re.search(sim_pat, curr_sparta_file)
if match:
    sparta_name = match.group(0)
else:
    sparta_name = curr_sparta_file
    
SPARTA_hdf5_path = SPARTA_output_path + sparta_name + "/" + curr_sparta_file + ".hdf5"

sim = test_sims[0][0]

comb_model_sims = get_combined_name(model_sims) 
        
model_name = get_model_name(model_type, model_sims, hpo_done=dset_params["OPTIMIZE"]["hpo"], opt_param_dict=dset_params["OPTIMIZE"])    
model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"
gen_plot_save_loc = model_fldr_loc + "plots/"

dset_name = "Test"
test_comb_name = get_combined_name(test_sims[0]) 

plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/halo_slices/"
create_directory(plot_loc)

halo_ddf = reform_dataset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/")
all_idxs = halo_ddf["Halo_indices"].values


        
sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
# find the snapshots for this simulation
snap_pat = r"(\d+)to(\d+)"
match = re.search(snap_pat, sim)
if match:
    curr_snap_list = [match.group(1), match.group(2)]   
    p_snap = int(curr_snap_list[0])

with open(ML_dset_path + sim + "/dset_params.pickle", "rb") as file:
    dset_params = pickle.load(file)
    
    curr_z = dset_params["p_snap_info"]["red_shift"][()]
    curr_snap_dir_format = dset_params["snap_dir_format"]
    curr_snap_format = dset_params["snap_format"]
    p_scale_factor = dset_params["p_snap_info"]["scale_factor"][()]
    p_sparta_snap = dset_params["p_snap_info"]["sparta_snap"]
    p_box_size = dset_params["p_snap_info"]["box_size"]
    
curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"

param_paths = [["halos","position"],["halos","R200m"],["halos","id"],["halos","status"],["halos","last_snap"],["simulation","particle_mass"]]
sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, curr_sparta_file, p_snap)

halos_pos = sparta_params[sparta_param_names[0]][:,p_sparta_snap,:] * 10**3 * p_scale_factor # convert to kpc/h
halos_r200m = sparta_params[sparta_param_names[1]][:,p_sparta_snap]
halos_ids = sparta_params[sparta_param_names[2]][:,p_sparta_snap]
halos_status = sparta_params[sparta_param_names[3]][:,p_sparta_snap]
halos_last_snap = sparta_params[sparta_param_names[4]][:]
ptl_mass = sparta_params[sparta_param_names[5]]

p_snap_path = snap_path + "snapdir_" + snap_dir_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

ptls_pid = load_ptl_param(curr_sparta_file, "pid", str(p_snap), p_snap_path) 
ptls_vel = load_ptl_param(curr_sparta_file, "vel", str(p_snap), p_snap_path) # km/s
ptls_pos = load_ptl_param(curr_sparta_file, "pos", str(p_snap), p_snap_path) * 10**3 * p_scale_factor # kpc/h

if os.path.isfile(ML_dset_path + sim + "/p_ptl_tree.pickle"):
    with open(ML_dset_path + sim + "/p_ptl_tree.pickle", "rb") as pickle_file:
        tree = pickle.load(pickle_file)
else:
    tree = cKDTree(data = ptls_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size)

random.seed(365)
used_numbers = set()
while len(used_numbers) < 25:
    with timed("Halo Slice Plot"):
        num = random.randint(0, all_idxs.shape[0])
        if num not in used_numbers:
            use_idx = all_idxs[num]

        use_halo_pos = halos_pos[use_idx]
        use_halo_r200m = halos_r200m[use_idx]
        use_halo_id = halos_ids[use_idx]

        ptl_indices = tree.query_ball_point(use_halo_pos, r = search_radius * 1.5 * use_halo_r200m)
        ptl_indices = np.array(ptl_indices)

        curr_ptl_pos = ptls_pos[ptl_indices]
        curr_ptl_pids = ptls_pid[ptl_indices]

        num_new_ptls = curr_ptl_pos.shape[0]
        
        if num_new_ptls > 1000:
            used_numbers.add(num)
        else:
            continue

        sparta_output = sparta.load(filename = SPARTA_hdf5_path, halo_ids=use_halo_id, log_level=0)

        sparta_last_pericenter_snap = sparta_output['tcr_ptl']['res_oct']['last_pericenter_snap']
        sparta_n_pericenter = sparta_output['tcr_ptl']['res_oct']['n_pericenter']
        sparta_tracer_ids = sparta_output['tcr_ptl']['res_oct']['tracer_id']
        sparta_n_is_lower_limit = sparta_output['tcr_ptl']['res_oct']['n_is_lower_limit']

        compare_sparta_assn = np.zeros((sparta_tracer_ids.shape[0]))
        curr_orb_assn = np.zeros((num_new_ptls))
            # Anywhere sparta_last_pericenter is greater than the current snap then that is in the future so set to 0
        future_peri = np.where(sparta_last_pericenter_snap > p_snap)[0]
        adj_sparta_n_pericenter = sparta_n_pericenter
        adj_sparta_n_pericenter[future_peri] = 0
        adj_sparta_n_is_lower_limit = sparta_n_is_lower_limit
        adj_sparta_n_is_lower_limit[future_peri] = 0
        # If a particle has a pericenter or if the lower limit is 1 then it is orbiting

        compare_sparta_assn[np.where((adj_sparta_n_pericenter >= 1) | (adj_sparta_n_is_lower_limit == 1))[0]] = 1
        # compare_sparta_assn[np.where(adj_sparta_n_pericenter >= 1)] = 1

        # Compare the ids between SPARTA and the found prtl ids and match the SPARTA results
        matched_ids = np.intersect1d(curr_ptl_pids, sparta_tracer_ids, return_indices = True)
        curr_orb_assn[matched_ids[1]] = compare_sparta_assn[matched_ids[2]]
        
        r, = calc_radius(use_halo_pos[0],use_halo_pos[1],use_halo_pos[2],curr_ptl_pos[:,0],curr_ptl_pos[:,1],curr_ptl_pos[:,2],p_box_size)
        r_scale = r/use_halo_r200m

        print("\nHalo:",str(num))
        nptl_inc_placement_r200m(r_scale, 1.0, curr_orb_assn)

        plot_halo_slice(curr_ptl_pos,curr_orb_assn,use_halo_pos,use_halo_r200m,plot_loc,search_rad=4,title=str(num)+"_")