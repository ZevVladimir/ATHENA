import numpy as np
from scipy.optimize import minimize
import os
import pandas as pd
import argparse
import multiprocessing as mp

from src.utils.ML_fxns import get_combined_name, get_feature_labels, extract_snaps
from src.utils.util_fxns import create_directory, load_pickle, load_config, save_pickle, load_pickle, timed, parse_ranges, split_sparta_hdf5_name, load_SPARTA_data, load_all_sim_cosmols, load_all_tdyn_steps
from src.utils.ke_cut_fxns import load_ke_data, opt_ke_predictor
from src.utils.vis_fxns import plt_SPARTA_KE_dist
from src.utils.prfl_fxns import paper_dens_prf_plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default=os.getcwd() + "/config.ini", 
    help='Path to config file (default: config.ini)'
)

args = parser.parse_args()
config_params = load_config(args.config)

ML_dset_path = config_params["PATHS"]["ml_dset_path"]
path_to_models = config_params["PATHS"]["path_to_models"]
snap_path = config_params["SNAP_DATA"]["snap_path"]
SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]

debug_mem = config_params["MISC"]["debug_mem"]
reset_opt_ke_calib = config_params["MISC"]["reset_opt_ke_calib"]
save_intermediate_data = config_params["MISC"]["save_intermediate_data"]

plt_nu_splits = parse_ranges(config_params["EVAL_MODEL"]["plt_nu_splits"])
plt_macc_splits = parse_ranges(config_params["EVAL_MODEL"]["plt_macc_splits"])
linthrsh = config_params["EVAL_MODEL"]["linthrsh"]
lin_nbin = config_params["EVAL_MODEL"]["lin_nbin"]
log_nbin = config_params["EVAL_MODEL"]["log_nbin"]
lin_rvticks = config_params["EVAL_MODEL"]["lin_rvticks"]
log_rvticks = config_params["EVAL_MODEL"]["log_rvticks"]
lin_tvticks = config_params["EVAL_MODEL"]["lin_tvticks"]
log_tvticks = config_params["EVAL_MODEL"]["log_tvticks"]
lin_rticks = config_params["EVAL_MODEL"]["lin_rticks"]
log_rticks = config_params["EVAL_MODEL"]["log_rticks"]
dens_prf_nu_split = config_params["EVAL_MODEL"]["dens_prf_nu_split"]
dens_prf_macc_split = config_params["EVAL_MODEL"]["dens_prf_macc_split"]

features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]

ke_test_dsets = config_params["KE_CUT"]["ke_test_dsets"]
fast_ke_calib_sims = config_params["KE_CUT"]["fast_ke_calib_sims"]
opt_ke_calib_sims = config_params["KE_CUT"]["opt_ke_calib_sims"]
perc = config_params["KE_CUT"]["perc"]
width = config_params["KE_CUT"]["width"]
grad_lims = config_params["KE_CUT"]["grad_lims"]
r_cut_calib = config_params["KE_CUT"]["r_cut_calib"]
r_cut_pred = config_params["KE_CUT"]["r_cut_pred"]
ke_test_sims = config_params["KE_CUT"]["ke_test_sims"]

num_processes = mp.cpu_count()
    
def overlap_loss(params, lnv2_bin, sparta_labels_bin):
    decision_boundary = params[0]
    line_classif = (lnv2_bin <= decision_boundary).astype(int)  

    # Count misclassified particles
    misclass_orb = np.sum((line_classif == 0) & (sparta_labels_bin == 1))
    misclass_inf = np.sum((line_classif == 1) & (sparta_labels_bin == 0))

    # Loss is the absolute difference between the two misclassification counts
    return abs(misclass_orb - misclass_inf)

def optimize_bin(i, lnv2_bin, sparta_labels_bin, def_b):
    ptls_in_bin = len(lnv2_bin)
    if ptls_in_bin == 0:
        return def_b

    initial_guess = [np.max(lnv2_bin)]
    result_max = minimize(overlap_loss, initial_guess, args=(lnv2_bin, sparta_labels_bin), method="Nelder-Mead")

    initial_guess = [np.mean(lnv2_bin)]
    result_mean = minimize(overlap_loss, initial_guess, args=(lnv2_bin, sparta_labels_bin), method="Nelder-Mead")

    if result_mean.fun < result_max.fun:
        result = result_mean
    else:
        result = result_max

    calc_b = result.x[0]
    if np.isnan(calc_b):
        print(f"Warning: NaN value encountered in bin {i}. Using default value {def_b}.")
        calc_b = def_b
    return calc_b

def opt_func(bins, r, lnv2, sparta_labels, def_b):    
    # Assign bin indices based on radius
    bin_indices = np.digitize(r.values, bins) - 1

    n_bins = bins.shape[0] - 1
    tasks = []
    for i in range(n_bins):
        mask = bin_indices == i
        lnv2_bin = lnv2.values[mask]
        sparta_labels_bin = sparta_labels.values[mask]
        tasks.append((i, lnv2_bin, sparta_labels_bin, def_b))

    with mp.Pool(processes=num_processes) as p:
        results = p.starmap(optimize_bin, tasks)

    return {"b": results}
    
if __name__ == "__main__":
    model_type = "kinetic_energy_cut"
    comb_fast_model_sims = get_combined_name(fast_ke_calib_sims) 
    comb_opt_model_sims = get_combined_name(opt_ke_calib_sims)   
     
    fast_model_fldr_loc = path_to_models + comb_fast_model_sims + "/" + model_type + "/"
    opt_model_fldr_loc = path_to_models + comb_opt_model_sims + "/" + model_type + "/"  
    create_directory(opt_model_fldr_loc)
    
    split_scale_dict = {
            "linthrsh":linthrsh, 
            "lin_nbin":lin_nbin,
            "log_nbin":log_nbin,
            "lin_rvticks":lin_rvticks,
            "log_rvticks":log_rvticks,
            "lin_tvticks":lin_tvticks,
            "log_tvticks":log_tvticks,
            "lin_rticks":lin_rticks,
            "log_rticks":log_rticks,
    }
    
    param_path = fast_model_fldr_loc + "ke_fastparams_dict.pickle"
    if os.path.exists(param_path):
        ke_param_dict = load_pickle(param_path)
    else:
        raise FileNotFoundError(
            f"Fast KE cut parameter file not found at {param_path}. Please run the fast_ke_cut.py to generate it."
        )
    with timed("SPARTA load"):
        sparta_name, sparta_search_name = split_sparta_hdf5_name(opt_ke_calib_sims[0])
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5" 
        param_paths = [["config","anl_prf","r_bins_lin"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, save_data=save_intermediate_data)
        bins = sparta_params[sparta_param_names[0]]
        bins = np.insert(bins, 0, 0)

    snap_list = extract_snaps(opt_ke_calib_sims[0])
    
    if not os.path.isfile(opt_model_fldr_loc + "ke_optparams_dict.pickle") or reset_opt_ke_calib == 1:
        with timed("Optimizing phase-space cut"):
            with timed("Loading Fitting Data"):
                all_sim_cosmol_list = load_all_sim_cosmols(opt_ke_calib_sims)
                all_tdyn_steps_list = load_all_tdyn_steps(opt_ke_calib_sims)
                
                r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ke_data(curr_sims=opt_ke_calib_sims,sim_cosmol_list=all_sim_cosmol_list,snap_list=snap_list,dset_name="Full")
                
                # We use the full dataset since for our custom fitting it does not only specific halos (?)
                r_fit = my_data["p_Scaled_radii"]
                vr_fit = my_data["p_Radial_vel"]
                vphys_fit = my_data["p_phys_vel"]
                sparta_labels_fit = my_data["Orbit_infall"]
                lnv2_fit = np.log(vphys_fit**2)

                mask_vr_neg = (vr_fit < 0)
                mask_vr_pos = (vr_fit > 0)                
            
            vr_pos = opt_func(bins, r_fit[mask_vr_pos], lnv2_fit[mask_vr_pos], sparta_labels_fit[mask_vr_pos], ke_param_dict["b_pos"])

            vr_neg = opt_func(bins, r_fit[mask_vr_neg], lnv2_fit[mask_vr_neg], sparta_labels_fit[mask_vr_neg], ke_param_dict["b_neg"])
        
            opt_param_dict = {
                "orb_vr_pos": vr_pos,
                "orb_vr_neg": vr_neg,
                "inf_vr_neg": vr_neg,
                "inf_vr_pos": vr_pos,
            }
        
            save_pickle(opt_param_dict,opt_model_fldr_loc+"ke_optparams_dict.pickle")
    else:
        print("Loading parameters from saved file")
        opt_param_dict = load_pickle(opt_model_fldr_loc + "ke_optparams_dict.pickle")

    for curr_test_sims in ke_test_sims:
        for dset_name in ke_test_dsets:
            all_sim_cosmol_list = load_all_sim_cosmols(curr_test_sims)
            all_tdyn_steps_list = load_all_tdyn_steps(curr_test_sims)
            
            feature_columns = get_feature_labels(features,all_tdyn_steps_list[0])
            
            test_comb_name = get_combined_name(curr_test_sims) 
        
            plot_loc = opt_model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
            create_directory(plot_loc)           
            
            # if the testing simulations are the same as the model simulations we don't need to reload the data
            if (not os.path.isfile(opt_model_fldr_loc + "ke_optparams_dict.pickle") or reset_opt_ke_calib == 1) and sorted(curr_test_sims) == sorted(opt_ke_calib_sims) and sorted(ke_test_dsets) == "Full":
                print("Using fitting simulations for testing")
                r_test = r_fit
                vr_test = vr_fit
                vphys_test = vphys_fit
                sparta_labels_test = sparta_labels_fit
                lnv2_test = lnv2_fit
                     
            with timed("Loading Testing Data"):
                r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ke_data(curr_sims=curr_test_sims,sim_cosmol_list=all_sim_cosmol_list,snap_list=snap_list,dset_name=dset_name)
                r_test = my_data["p_Scaled_radii"]
                vr_test = my_data["p_Radial_vel"]
                vphys_test = my_data["p_phys_vel"]
                sparta_labels_test = my_data["Orbit_infall"]
                lnv2_test = np.log(vphys_test**2)      
            
            mask_orb = sparta_labels_test == 1
            mask_inf = sparta_labels_test == 0
            
            mask_vr_neg = vr_test < 0
            mask_vr_pos = vr_test > 0
                
            fltr_combs = {
                "orb_vr_neg": mask_orb & mask_vr_neg,
                "orb_vr_pos": mask_orb & mask_vr_pos,
                "inf_vr_neg": mask_inf & mask_vr_neg,
                "inf_vr_pos": mask_inf & mask_vr_pos,
            } 

            plt_SPARTA_KE_dist(ke_param_dict, fltr_combs, bins, r_test, lnv2_test, r_cut = r_cut_calib, plot_loc = plot_loc, title = "bin_fit_", plot_lin_too=True, cust_line_dict = opt_param_dict)

        #######################################################################################################################################    
            preds_opt_ke = opt_ke_predictor(opt_param_dict, bins, r_test, vr_test, lnv2_test, r_cut_pred)
            
            X = my_data[feature_columns]
            y = my_data[target_column]

            paper_dens_prf_plt(X, y, pd.DataFrame(preds_opt_ke), halo_df, curr_test_sims, all_sim_cosmol_list, split_scale_dict, plot_loc, snap_path, split_by_nu=dens_prf_nu_split, split_by_macc=dens_prf_macc_split, prf_name_0="Optimized KE cut")

            