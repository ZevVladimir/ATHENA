from utils.ML_support import *
from utils.update_vis_fxns import *
from utils.data_and_loading_functions import *
import os
import json
import random
import pickle
import re
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
sim_cosmol = config["MISC"]["sim_cosmol"]
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
else:
    sim_pat = r"cbol_l(\d+)_n(\d+)"
match = re.search(sim_pat, curr_sparta_file)
if match:
    sparta_name = match.group(0)
path_to_hdf5_file = path_to_SPARTA_data + sparta_name + "/" + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
path_to_xgboost = config["PATHS"]["path_to_xgboost"]
test_sims = json.loads(config.get("XGBOOST","test_sims"))
model_sims = json.loads(config.get("XGBOOST","model_sims"))
model_type = config["XGBOOST"]["model_type"]
nu_splits = config["XGBOOST"]["nu_splits"]
nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)
sim = test_sims[0][0]

model_comb_name = get_combined_name(model_sims) 

model_dir = model_type + "_" + model_comb_name + "nu" + nu_string 

model_save_loc = path_to_xgboost + model_comb_name + "/" + model_dir + "/"
dset_name = "Test"
test_comb_name = get_combined_name(test_sims[0]) 

plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/halo_slices/"
create_directory(plot_loc)

halo_ddf = reform_df(path_to_calc_info + sim + "/" + "Test" + "/halo_info/")
all_idxs = halo_ddf["Halo_indices"].values

with open(path_to_calc_info + sim + "/p_ptl_tree.pickle", "rb") as pickle_file:
    tree = pickle.load(pickle_file)
        
sparta_name, sparta_search_name = split_calc_name(sim)
# find the snapshots for this simulation
snap_pat = r"(\d+)to(\d+)"
match = re.search(snap_pat, sim)
if match:
    curr_snap_list = [match.group(1), match.group(2)]   
    p_snap = int(curr_snap_list[0])

with open(path_to_calc_info + sim + "/config.pickle", "rb") as file:
    config_dict = pickle.load(file)
    
    curr_z = config_dict["p_snap_info"]["red_shift"][()]
    curr_snap_dir_format = config_dict["snap_dir_format"]
    curr_snap_format = config_dict["snap_format"]
    new_p_snap, curr_z = find_closest_z(curr_z,path_to_snaps + sparta_name + "/",curr_snap_dir_format,curr_snap_format)
    p_scale_factor = 1/(1+curr_z)
    
with h5py.File(path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5","r") as f:
    dic_sim = {}
    grp_sim = f['simulation']

    for attr in grp_sim.attrs:
        dic_sim[attr] = grp_sim.attrs[attr]

all_red_shifts = dic_sim['snap_z']
p_sparta_snap = np.abs(all_red_shifts - curr_z).argmin()

halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, parent_id, ptl_mass = load_or_pickle_SPARTA_data(sparta_search_name, p_scale_factor, p_snap, p_sparta_snap)

snap_loc = path_to_snaps + sparta_name + "/"
p_snapshot_path = snap_loc + "snapdir_" + snap_dir_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)
ptls_pid, ptls_vel, ptls_pos = load_or_pickle_ptl_data(curr_sparta_file, str(p_snap), p_snapshot_path, p_scale_factor)

random.seed(365)
used_numbers = set()
while len(used_numbers) < 10:
    with timed("Halo Slice Plot"):
        num = random.randint(0, all_idxs.shape[0])
        if num not in used_numbers:
            used_numbers.add(num)
            use_idx = all_idxs[num]

        use_halo_pos = halos_pos[use_idx]
        use_halo_r200m = halos_r200m[use_idx]
        use_halo_id = halos_id[use_idx]

        ptl_indices = tree.query_ball_point(use_halo_pos, r = search_rad * 1.5 * use_halo_r200m)
        ptl_indices = np.array(ptl_indices)

        curr_ptl_pos = ptls_pos[ptl_indices]
        curr_ptl_pids = ptls_pid[ptl_indices]

        num_new_ptls = curr_ptl_pos.shape[0]

        sparta_output = sparta.load(filename = path_to_hdf5_file, halo_ids=use_halo_id, log_level=0)

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

        plot_halo_slice(curr_ptl_pos,curr_orb_assn,use_halo_pos,use_halo_r200m,plot_loc,search_rad=4,title=str(num)+"_")