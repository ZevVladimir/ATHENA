import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex":True, "font.family": "serif", "figure.dpi": 150})
import os
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import dask.dataframe as dd
from dask import delayed
import pandas as pd

from utils.ML_support import setup_client, get_combined_name, parse_ranges, load_sparta_mass_prf, load_data, extract_snaps, get_feature_labels, set_cosmology, filter_ddf, filter_df, split_sparta_hdf5_name
from utils.data_and_loading_functions import create_directory, load_pickle, load_config, load_pickle, timed, load_SPARTA_data
from src.utils.ke_cut_support import fast_ke_predictor, opt_ke_predictor
from src.utils.vis_fxns import plt_SPARTA_KE_dist

config_params = load_config(os.getcwd() + "/config.ini")

ML_dset_path = config_params["PATHS"]["ml_dset_path"]
path_to_models = config_params["PATHS"]["path_to_models"]
SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]
debug_plt_path = config_params["PATHS"]["debug_plt_path"]

pickle_data = config_params["MISC"]["pickle_data"]

features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]

eval_datasets = config_params["EVAL_MODEL"]["eval_datasets"]
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
    client = setup_client()

    model_type = "kinetic_energy_cut"
    
    comb_fast_model_sims = get_combined_name(fast_ke_calib_sims) 
    comb_opt_model_sims = get_combined_name(opt_ke_calib_sims)   
      
    fast_model_fldr_loc = path_to_models + comb_fast_model_sims + "/" + model_type + "/"
    opt_model_fldr_loc = path_to_models + comb_opt_model_sims + "/" + model_type + "/" 
    #TODO make this a loop through test sims
    curr_test_sims = ke_test_sims[0]
    test_comb_name = get_combined_name(curr_test_sims) 
    dset_name = eval_datasets[0]
    plot_loc = opt_model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
    create_directory(plot_loc)
    
    dset_params = load_pickle(ML_dset_path + curr_test_sims[0] + "/dset_params.pickle")
    sim_cosmol = dset_params["cosmology"]
    all_tdyn_steps = dset_params["t_dyn_steps"]
    
    feature_columns = get_feature_labels(features,all_tdyn_steps)
    snap_list = extract_snaps(curr_test_sims[0])
    cosmol = set_cosmology(sim_cosmol)
    
    if os.path.isfile(opt_model_fldr_loc + "ke_optparams_dict.pickle"):
        print("Loading parameters from saved file")
        opt_param_dict = load_pickle(opt_model_fldr_loc + "ke_optparams_dict.pickle")
    else:
        raise FileNotFoundError(f"Expected to find optimized parameters at {os.path.join(opt_model_fldr_loc, 'ke_optparams_dict.pickle')}")
    with timed("Histogram Creation"):
        # Get the redshifts for each simulation's primary snapshot

        test_comb_name = get_combined_name(curr_test_sims) 
        
        data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,sim_cosmol,snap_list[0],limit_files=False)
        
        columns_to_keep = [col for col in data.columns if col != target_column[0]]
        X_df = data[columns_to_keep]
        y_df = data[target_column]
            
        X_df = X_df.to_backend('pandas')
        y_df = y_df.to_backend('pandas')

        # new_columns = ["Current $r/R_{\mathrm{200m}}$","Current $v_{\mathrm{r}}/V_{\mathrm{200m}}$","Current $v_{\mathrm{t}}/V_{\mathrm{200m}}$","Past $r/R_{\mathrm{200m}}$","Past $v_{\mathrm{r}}/V_{\mathrm{200m}}$","Past $v_{\mathrm{t}}/V_{\mathrm{200m}}$"]
        # col2num = {col: i for i, col in enumerate(new_columns)}
        # order = list(map(col2num.get, new_columns))
        
        r_test = X_df["p_Scaled_radii"]
        vr_test = X_df["p_Radial_vel"]
        vphys = X_df["p_phys_vel"]
        lnv2_test = np.log(vphys**2)
        X_df["p_lnv2"] = lnv2_test
        
        mask_vr_neg = (vr_test < 0)
        mask_vr_pos = ~mask_vr_neg

        ke_fastparam_dict = load_pickle(fast_model_fldr_loc + "ke_fastparams_dict.pickle")        
        fast_mask_orb, preds_fast_ke = fast_ke_predictor(ke_fastparam_dict,r_test.compute().to_numpy(),vr_test.compute().to_numpy(),lnv2_test.compute().to_numpy(),r_cut_calib)
        
        sparta_name, sparta_search_name = split_sparta_hdf5_name(curr_test_sims[0])
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5" 
        
        param_paths = [["config","anl_prf","r_bins_lin"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, pickle_data=pickle_data)
        bins = sparta_params[sparta_param_names[0]]
        
        
        preds_fit_ke = opt_ke_predictor(opt_param_dict, bins, r_test.compute().to_numpy(), vr_test.compute().to_numpy(), lnv2_test.compute().to_numpy(), r_cut_calib)
        
        y_df = y_df.values.compute().squeeze()
        X_df = X_df.compute()
        
        param_path = fast_model_fldr_loc + "ke_fastparams_dict.pickle"
        ke_param_dict = load_pickle(param_path)      
        
        bin_indices = np.digitize(X_df["p_Scaled_radii"], bins) - 1  # subtract 1 to make bins zero-indexed

        # Step 2: Initialize counters
        num_bins = len(bins) - 1
        ml_orbiting_counts = np.zeros(num_bins)
        ml_infalling_counts = np.zeros(num_bins)
        ke_orbiting_counts = np.zeros(num_bins)
        ke_infalling_counts = np.zeros(num_bins)
    
        for i in range(num_bins):
            in_bin = bin_indices == i
            ml_orbiting_counts[i] = np.sum(y_df[in_bin] == 1)
            ml_infalling_counts[i] = np.sum(y_df[in_bin] == 0)
            ke_orbiting_counts[i] = np.sum(preds_fast_ke[in_bin] == 1)
            ke_infalling_counts[i] = np.sum(preds_fast_ke[in_bin] == 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ml_ratio = np.where(ml_infalling_counts > 0, ml_orbiting_counts / (ml_infalling_counts + ml_orbiting_counts), np.nan)
            ke_ratio = np.where(ke_infalling_counts > 0, ke_orbiting_counts / (ke_infalling_counts + ke_orbiting_counts), np.nan)
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        bin_widths = bins[1:] - bins[:-1]
        bar_widths = bin_widths * 0.4
        offset = bar_widths / 2

        fig, ax = plt.subplots(1, figsize=(10,5))
        ax.bar(bin_centers - offset, ml_ratio, width=bar_widths, label='ML Classification', alpha=0.7)
        ax.bar(bin_centers + offset, ke_ratio, width=bar_widths, label='Fast KE Cut Classification', alpha=0.7)     
        ax.set_xscale("log")  
        ax.legend()
        ax.set_xlabel(r"Radius $r/R_{\rm 200m}$")
        ax.set_ylabel(r"$N_{\rm orb}/N_{\rm tot}$")
        
        fig.savefig(debug_plt_path + test_comb_name + "_orb_rat_by_rad.png")
        