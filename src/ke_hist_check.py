import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex":True, "font.family": "serif", "figure.dpi": 300})
import os

from src.utils.ML_fxns import setup_client, get_combined_name, parse_ranges, extract_snaps, get_feature_labels, filter_df, split_sparta_hdf5_name
from src.utils.save_load_fxns import create_directory, load_pickle, load_config, load_pickle, timed, load_SPARTA_data, load_ML_dsets
from src.utils.ke_cut_fxns import fast_ke_predictor, opt_ke_predictor
from src.utils.misc_fxns import set_cosmology

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
        
        data,scale_pos_weight = load_ML_dsets(client,curr_test_sims,dset_name,sim_cosmol,snap_list[0],limit_files=False)
        
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
        bins = np.insert(bins, 0, 0)
        
        preds_fit_ke = opt_ke_predictor(opt_param_dict, bins, r_test.compute().to_numpy(), vr_test.compute().to_numpy(), lnv2_test.compute().to_numpy(), r_cut_calib)
        
        orb_agree_dict = {
            'X_filter': {
                "p_Scaled_radii": [('>',1.7),('<',1.9)],
                "p_lnv2": [('>',-4),('<',-3)]
            },
            'label_filter': {
                "sparta":1,
                "pred":1
            }
        }
        
        inf_agree_dict = {
            'X_filter': {
                "p_Scaled_radii": [('>',1.7),('<',1.9)],
                "p_lnv2": [('>',-4),('<',-3)]
            },
            'label_filter': {
                "sparta":0,
                "pred":0
            }
        }
        
        orb_disagree_dict = {
            'X_filter': {
                "p_Scaled_radii": [('>',1.7),('<',1.9)],
                "p_lnv2": [('>',-4),('<',-3)]
            },
            'label_filter': {
                "sparta":1,
                "pred":0
            }
        }
        
        inf_disagree_dict = {
            'X_filter': {
                "p_Scaled_radii": [('>',1.7),('<',1.9)],
                "p_lnv2": [('>',-4),('<',-3)]
            },
            'label_filter': {
                "sparta":0,
                "pred":1
            }
        }
        
        y_df = y_df.values.compute().squeeze()
        X_df = X_df.compute()
        orb_agree, orb_agree_fltr = filter_df(X_df,y_df,preds_fast_ke,orb_agree_dict)
        orb_disagree, orb_disagree_fltr = filter_df(X_df, y_df, preds_fast_ke, orb_disagree_dict)
        inf_agree, inf_agree_fltr = filter_df(X_df, y_df, preds_fast_ke, inf_agree_dict)
        inf_disagree, inf_disgree_fltr = filter_df(X_df, y_df, preds_fast_ke, inf_disagree_dict) 
        
        param_path = fast_model_fldr_loc + "ke_fastparams_dict.pickle"
        ke_param_dict = load_pickle(param_path)      
        
        
        nbins = 25
        
        fig, ax = plt.subplots(2,4, figsize=(24,12))
        
        ax[0,0].hist(orb_agree["p_Scaled_radii"],nbins)
        ax[0,0].set_xlabel(r"Present Radii $r/R_{\mathrm{200m}}$")
        ax[0,0].set_ylabel("Number of Counts")
        ax[0,0].set_title("Orbiting Particles Same Classification")
        
        ax[0,1].hist(inf_agree["p_Scaled_radii"],nbins)
        ax[0,1].set_xlabel(r"Present Radii $r/R_{\mathrm{200m}}$")
        ax[0,1].set_ylabel("Number of Counts")
        ax[0,1].set_title("Infalling Particles Same Classification")
        
        ax[0,2].hist(orb_disagree["p_Scaled_radii"],nbins)
        ax[0,2].set_xlabel(r"Present Radii $r/R_{\mathrm{200m}}$")
        ax[0,2].set_ylabel("Number of Counts")
        ax[0,2].set_title("SPARTA: Orbiting Fast: Infalling")
        
        ax[0,3].hist(inf_disagree["p_Scaled_radii"],nbins)
        ax[0,3].set_xlabel(r"Present Radii $r/R_{\mathrm{200m}}$")
        ax[0,3].set_ylabel("Number of Counts")
        ax[0,3].set_title("SPARTA: Infalling Fast: Orbiting")
        
        ax[1,0].hist(orb_agree["1_Scaled_radii"],nbins)
        ax[1,0].set_xlabel(r"Past Radii $r/R_{\mathrm{200m}}$")
        ax[1,0].set_ylabel("Number of Counts")
        ax[1,0].set_title("Orbiting Particles Same Classification")
        
        ax[1,1].hist(inf_agree["1_Scaled_radii"],nbins)
        ax[1,1].set_xlabel(r"Past Radii $r/R_{\mathrm{200m}}$")
        ax[1,1].set_ylabel("Number of Counts")
        ax[1,1].set_title("Infalling Particles Same Classification")
        
        ax[1,2].hist(orb_disagree["1_Scaled_radii"],nbins)
        ax[1,2].set_xlabel(r"Past Radii $r/R_{\mathrm{200m}}$")
        ax[1,2].set_ylabel("Number of Counts")
        ax[1,2].set_title("SPARTA: Orbiting Fast: Infalling")
        
        ax[1,3].hist(inf_disagree["1_Scaled_radii"],nbins)
        ax[1,3].set_xlabel(r"Past Radii $r/R_{\mathrm{200m}}$")
        ax[1,3].set_ylabel("Number of Counts")
        ax[1,3].set_title("SPARTA: Infalling Fast: Orbiting")
        
        fig.savefig(debug_plt_path + test_comb_name + "_radius_hists.png")
        