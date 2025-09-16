import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost as xgb
import argparse

from src.utils.ML_fxns import get_combined_name, make_preds, get_model_name, extract_snaps, get_feature_labels
from src.utils.util_fxns import create_directory, load_pickle, load_config, load_pickle, timed, load_SPARTA_data, load_ML_dsets, load_all_sim_cosmols, load_all_tdyn_steps
from src.utils.ke_cut_fxns import fast_ke_predictor, opt_ke_predictor
from src.utils.util_fxns import parse_ranges, split_sparta_hdf5_name

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
SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]
debug_plt_path = config_params["PATHS"]["debug_plt_path"]

save_intermediate_data = config_params["MISC"]["save_intermediate_data"]

use_gpu = config_params["ENVIRONMENT"]["use_gpu"]

model_sims = config_params["TRAIN_MODEL"]["model_sims"]
features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]
model_type = config_params["TRAIN_MODEL"]["model_type"]

eval_test_dsets = config_params["EVAL_MODEL"]["eval_test_dsets"]
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

fast_ke_calib_sims = config_params["KE_CUT"]["fast_ke_calib_sims"]
opt_ke_calib_sims = config_params["KE_CUT"]["opt_ke_calib_sims"]
r_cut_calib = config_params["KE_CUT"]["r_cut_calib"]
ke_test_sims = config_params["KE_CUT"]["ke_test_sims"]
    
    
if __name__ == "__main__":    
    comb_model_sims = get_combined_name(model_sims) 
        
    model_name = get_model_name(model_type, model_sims)    
    model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"
    model_save_loc = model_fldr_loc + model_name + ".json"
    
    try:
        bst = xgb.Booster()
        bst.load_model(model_save_loc)
        if use_gpu:
            bst.set_param({"device": "cuda:0"})
        print("Loaded Model Trained on:",model_sims)
    except:
        print("Couldn't load Booster Located at: " + model_save_loc)

    model_type = "kinetic_energy_cut"
    
    comb_fast_model_sims = get_combined_name(fast_ke_calib_sims) 
    comb_opt_model_sims = get_combined_name(opt_ke_calib_sims)   
      
    fast_model_fldr_loc = path_to_models + comb_fast_model_sims + "/" + model_type + "/"
    opt_model_fldr_loc = path_to_models + comb_opt_model_sims + "/" + model_type + "/"    
    
    snap_list = extract_snaps(fast_ke_calib_sims[0])
    
    if os.path.isfile(opt_model_fldr_loc + "ke_optparams_dict.pickle"):
        print("Loading parameters from saved file")
        opt_param_dict = load_pickle(opt_model_fldr_loc + "ke_optparams_dict.pickle")
    else:
        raise FileNotFoundError(f"Expected to find optimized parameters at {os.path.join(opt_model_fldr_loc, 'ke_optparams_dict.pickle')}")
    
    for i,curr_test_sims in enumerate(ke_test_sims):
        test_comb_name = get_combined_name(curr_test_sims) 
        dset_name = eval_test_dsets[0]
        plot_loc = opt_model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
        create_directory(plot_loc)
        
        all_sim_cosmol_list = load_all_sim_cosmols(curr_test_sims)
        all_tdyn_steps_list = load_all_tdyn_steps(curr_test_sims)
        
        with timed("Fraction of Orb by Rad Plot Creation"):
            test_comb_name = get_combined_name(curr_test_sims) 
            
            data,scale_pos_weight,halo_df = load_ML_dsets(curr_test_sims,dset_name,all_sim_cosmol_list)
            
            columns_to_keep = [col for col in data.columns if col != target_column[0]]
            X_df = data[columns_to_keep]
            y_df = data[target_column]
            
            feature_columns = get_feature_labels(features,all_tdyn_steps_list[0])
            preds_ML = make_preds(bst, data[feature_columns])
            
            r_test = X_df["p_Scaled_radii"]
            vr_test = X_df["p_Radial_vel"]
            vphys = X_df["p_phys_vel"]
            lnv2_test = np.log(vphys**2)
            X_df["p_lnv2"] = lnv2_test
            
            mask_vr_neg = (vr_test < 0)
            mask_vr_pos = ~mask_vr_neg

            ke_fastparam_dict = load_pickle(fast_model_fldr_loc + "ke_fastparams_dict.pickle")        
            fast_mask_orb, preds_fast_ke = fast_ke_predictor(ke_fastparam_dict,r_test.values,vr_test.values,lnv2_test.values,r_cut_calib)
            
            sparta_name, sparta_search_name = split_sparta_hdf5_name(curr_test_sims[0])
            curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5" 
            
            param_paths = [["config","anl_prf","r_bins_lin"]]
            sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, save_data=save_intermediate_data)
            bins = sparta_params[sparta_param_names[0]]
            bins = np.insert(bins, 0, 0)

            preds_opt_ke = opt_ke_predictor(opt_param_dict, bins, r_test.values, vr_test.values, lnv2_test.values, r_cut_calib)

            y_df = y_df.values

            param_path = fast_model_fldr_loc + "ke_fastparams_dict.pickle"
            ke_param_dict = load_pickle(param_path)      
            
            bin_indices = np.digitize(X_df["p_Scaled_radii"], bins) - 1  # subtract 1 to make bins zero-indexed

            # Step 2: Initialize counters
            num_bins = len(bins) - 1
            sparta_orbiting_counts = np.zeros(num_bins)
            sparta_infalling_counts = np.zeros(num_bins)
            ml_orbiting_counts = np.zeros(num_bins)
            ml_infalling_counts = np.zeros(num_bins)
            fast_ke_orbiting_counts = np.zeros(num_bins)
            fast_ke_infalling_counts = np.zeros(num_bins)
            opt_ke_orbiting_counts = np.zeros(num_bins)
            opt_ke_infalling_counts = np.zeros(num_bins)
        
            for i in range(num_bins):
                in_bin = bin_indices == i
                sparta_orbiting_counts[i] = np.sum(y_df[in_bin] == 1)
                sparta_infalling_counts[i] = np.sum(y_df[in_bin] == 0)
                ml_orbiting_counts[i] = np.sum(preds_ML[in_bin] == 1)
                ml_infalling_counts[i] = np.sum(preds_ML[in_bin] == 0)
                fast_ke_orbiting_counts[i] = np.sum(preds_fast_ke[in_bin] == 1)
                fast_ke_infalling_counts[i] = np.sum(preds_fast_ke[in_bin] == 0)
                opt_ke_orbiting_counts[i] = np.sum(preds_opt_ke[in_bin] == 1)
                opt_ke_infalling_counts[i] = np.sum(preds_opt_ke[in_bin] == 0)

            with np.errstate(divide='ignore', invalid='ignore'):
                sparta_ratio = np.where(sparta_infalling_counts > 0, sparta_orbiting_counts / (sparta_infalling_counts + sparta_orbiting_counts), np.nan)
                ml_ratio = np.where(ml_infalling_counts > 0, ml_orbiting_counts / (ml_infalling_counts + ml_orbiting_counts), np.nan)
                fast_ke_ratio = np.where(fast_ke_infalling_counts > 0, fast_ke_orbiting_counts / (fast_ke_infalling_counts + fast_ke_orbiting_counts), np.nan)
                opt_ke_ratio = np.where(opt_ke_infalling_counts > 0, opt_ke_orbiting_counts / (opt_ke_infalling_counts + opt_ke_orbiting_counts), np.nan)

            fig, ax = plt.subplots(1, figsize=(10,5))
            ax.plot(bins[1:], sparta_ratio, label='SPARTA Classification')
            ax.plot(bins[1:], fast_ke_ratio, label='Fast KE Cut Classification')
            ax.plot(bins[1:], opt_ke_ratio, label='Optimized KE Cut Classification')
            ax.plot(bins[1:], ml_ratio, label='ML Classification')     
        
            legend_fntsize = 14
            axis_fntsize = 18
            tick_fntsize = 14
            ax.legend(fontsize=legend_fntsize, loc="upper right")
            ax.set_xlabel(r"$r/R_{\rm 200m}$",fontsize=axis_fntsize)
            ax.set_ylabel(r"$N_{\rm orb}/N_{\rm tot}$",fontsize=axis_fntsize)
            ax.set_xlim(0,3)
            ax.set_ylim(0,1)
            ax.tick_params(axis='both', labelsize=tick_fntsize, length=6,width=2, direction="in")
            
            fig.savefig(debug_plt_path + test_comb_name + "_forb_by_rad.pdf",dpi=400)
            