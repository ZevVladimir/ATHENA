import xgboost as xgb
import pickle
import os
import pandas as pd
import argparse

from src.utils.ML_fxns import setup_client, get_combined_name, get_model_name, extract_snaps, make_preds, get_feature_labels
from src.utils.util_fxns import create_directory, timed, load_pickle, save_pickle, load_config, load_ML_dsets,reform_dset_dfs, load_all_sim_cosmols, load_all_tdyn_steps
from src.utils.prfl_fxns import paper_dens_prf_plt
from src.utils.vis_fxns import gen_ptl_dist_plt, gen_missclass_dist_plt
##################################################################################################################
# LOAD CONFIG PARAMETERS
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

use_gpu = config_params["DASK_CLIENT"]["use_gpu"]

model_sims = config_params["TRAIN_MODEL"]["model_sims"]
model_type = config_params["TRAIN_MODEL"]["model_type"]
features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]

test_sims = config_params["EVAL_MODEL"]["test_sims"]
eval_datasets = config_params["EVAL_MODEL"]["eval_datasets"]
dens_prf_plt = config_params["EVAL_MODEL"]["dens_prf_plt"]
misclass_plt = config_params["EVAL_MODEL"]["misclass_plt"]
full_dist_plt = config_params["EVAL_MODEL"]["fulldist_plt"]
dens_prf_nu_split = config_params["EVAL_MODEL"]["dens_prf_nu_split"]
dens_prf_macc_split = config_params["EVAL_MODEL"]["dens_prf_macc_split"]

linthrsh = config_params["EVAL_MODEL"]["linthrsh"]
lin_nbin = config_params["EVAL_MODEL"]["lin_nbin"]
log_nbin = config_params["EVAL_MODEL"]["log_nbin"]
lin_rvticks = config_params["EVAL_MODEL"]["lin_rvticks"]
log_rvticks = config_params["EVAL_MODEL"]["log_rvticks"]
lin_tvticks = config_params["EVAL_MODEL"]["lin_tvticks"]
log_tvticks = config_params["EVAL_MODEL"]["log_tvticks"]
lin_rticks = config_params["EVAL_MODEL"]["lin_rticks"]
log_rticks = config_params["EVAL_MODEL"]["log_rticks"]

###############################################################################################################

if __name__ == "__main__":    
    client = setup_client()
    dset_params = load_pickle(ML_dset_path + model_sims[0] + "/dset_params.pickle")

    comb_model_sims = get_combined_name(model_sims) 
        
    model_name = get_model_name(model_type, model_sims)    
    model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"
    model_save_loc = model_fldr_loc + model_name + ".json"
    gen_plot_save_loc = model_fldr_loc + "plots/"

    # Try loading the model if it can't be thats an error!
    try:
        bst = xgb.Booster()
        bst.load_model(model_save_loc)
        if use_gpu:
            bst.set_param({"device": "cuda:0"})
        print("Loaded Model Trained on:",model_sims)
    except:
        print("Couldn't load Booster Located at: " + model_save_loc)
    
    # Try loading the model info it it can't that's an error!
    try:
        with open(model_fldr_loc + "model_info.pickle", "rb") as pickle_file:
            model_info = pickle.load(pickle_file)
    except FileNotFoundError:
        print("Model info could not be loaded please ensure the path is correct or rerun train_xgboost.py")
    
    # Loop through each set of test sims in the user inputted list
    for curr_test_sims in test_sims:
        test_comb_name = get_combined_name(curr_test_sims) 
        
        all_sim_cosmol_list = load_all_sim_cosmols(curr_test_sims)
        all_tdyn_steps_list = load_all_tdyn_steps(curr_test_sims)

        feature_columns = get_feature_labels(features,all_tdyn_steps_list[0])
        
        # Loop through and/or for Train/Test/All datasets and evaluate the model
        for dset_name in eval_datasets:
            with timed("Model Evaluation on " + dset_name + "_" + test_comb_name): 
                plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
                create_directory(plot_loc)

                # Load the halo information
                halo_files = []
                halo_dfs = []
                if dset_name == "Full":    
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_dset_dfs(ML_dset_path + sim + "/" + "Train" + "/halo_info/"))
                        halo_dfs.append(reform_dset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/"))
                else:
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_dset_dfs(ML_dset_path + sim + "/" + dset_name + "/halo_info/"))

                halo_df = pd.concat(halo_dfs)
                
                # Load the particle information
                all_snaps = extract_snaps(model_sims[0])
                data,scale_pos_weight = load_ML_dsets(client,curr_test_sims,dset_name,all_sim_cosmol_list,all_snaps[0])

                X = data[feature_columns]
                y = data[target_column]
                
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
                
                r_ticks = split_scale_dict["lin_rticks"] + split_scale_dict["log_rticks"]
        
                rv_ticks = split_scale_dict["lin_rvticks"] + split_scale_dict["log_rvticks"]
                rv_ticks = rv_ticks + [-x for x in rv_ticks if x != 0]
                rv_ticks.sort()

                tv_ticks = split_scale_dict["lin_tvticks"] + split_scale_dict["log_tvticks"]  
                
                with timed(f"Predictions for {y.size.compute():.3e} particles"):
                    preds = make_preds(client, bst, X)
                if dens_prf_plt:
                    paper_dens_prf_plt(X, y, preds, halo_df, curr_test_sims, all_sim_cosmol_list, split_scale_dict, plot_loc, split_by_nu=dens_prf_nu_split, split_by_macc=dens_prf_macc_split)
                if full_dist_plt or misclass_plt:
                    p_act_labels=y.compute().values.flatten()
                    p_r=X["p_Scaled_radii"].values.compute()
                    p_rv=X["p_Radial_vel"].values.compute()
                    p_tv=X["p_Tangential_vel"].values.compute()
                    c_r=X[str(all_tdyn_steps_list[0][0]) + "_Scaled_radii"].values.compute()
                    c_rv=X[str(all_tdyn_steps_list[0][0]) +"_Radial_vel"].values.compute()
                    plt_data_dict = {
                        "p_r":p_r,
                        "p_rv":p_rv,
                        "p_tv":p_tv,
                        "c_r":c_r,
                        "c_rv":c_rv,
                    }
                if full_dist_plt:
                    ptl_dist_plot_list = [
                        {"x": "p_r", "y": "p_rv", "split_x": False, "split_y": True, "x_label":"$r/R_{200m}$", "y_label":"$v_r/v_{200m}$","title":"Current Snapshot", "x_ticks":r_ticks, "y_ticks":rv_ticks, "hide_ytick_labels":False},
                        {"x": "p_r", "y": "p_tv", "split_x": False, "split_y": True, "x_label":"$r/R_{200m}$", "y_label":"$v_t/v_{200m}$","title":"", "x_ticks":r_ticks, "y_ticks":tv_ticks, "hide_ytick_labels":False},
                        {"x": "p_rv", "y": "p_tv", "split_x": True, "split_y": True, "x_label":"$v_r/v_{200m}$", "y_label":"","title":"", "x_ticks":rv_ticks, "y_ticks":tv_ticks, "hide_ytick_labels":True},
                        {"x": "c_r", "y": "c_rv", "split_x": False, "split_y": True, "x_label":"$r/R_{200m}$", "y_label":"$v_r/v_{200m}$","title":"Past Snapshot", "x_ticks":r_ticks, "y_ticks":rv_ticks, "hide_ytick_labels":False},
                    ]
                    gen_ptl_dist_plt(p_act_labels,split_scale_dict,plot_loc,plt_data_dict,ptl_dist_plot_list)
                if misclass_plt:
                    misclass_plot_list = [
                        {"x": "p_r", "y": "p_rv", "split_x": False, "split_y": True, "x_label":"$r/R_{200m}$", "y_label":"$v_r/v_{200m}$","title":"Current Snapshot", "x_ticks":r_ticks, "y_ticks":rv_ticks, "hide_ytick_labels":False},
                        {"x": "p_r", "y": "p_tv", "split_x": False, "split_y": True, "x_label":"$r/R_{200m}$", "y_label":"$v_t/v_{200m}$","title":"", "x_ticks":r_ticks, "y_ticks":tv_ticks, "hide_ytick_labels":False},
                        {"x": "p_rv", "y": "p_tv", "split_x": True, "split_y": True, "x_label":"$v_r/v_{200m}$", "y_label":"","title":"", "x_ticks":rv_ticks, "y_ticks":tv_ticks, "hide_ytick_labels":True},
                        {"x": "c_r", "y": "c_rv", "split_x": False, "split_y": True, "x_label":"$r/R_{200m}$", "y_label":"$v_r/v_{200m}$","title":"Past Snapshot", "x_ticks":r_ticks, "y_ticks":rv_ticks, "hide_ytick_labels":False},
                    ]
                    gen_missclass_dist_plt(p_act_labels, preds, split_scale_dict, plot_loc, model_info, dset_name, plt_data_dict, misclass_plot_list)
                del data 
                del X
                del y
        
        save_pickle(model_info,model_fldr_loc + "model_info.pickle")

    client.close()
