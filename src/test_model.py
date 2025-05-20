import xgboost as xgb
import pickle
import os
import numpy as np
import pandas as pd

from utils.ML_support import setup_client, get_combined_name, reform_dataset_dfs, load_data, eval_model, get_model_name, make_preds, get_feature_labels
from utils.data_and_loading_functions import create_directory, timed, load_pickle, save_pickle, load_config
##################################################################################################################
# LOAD CONFIG PARAMETERS
config_params = load_config(os.getcwd() + "/config.ini")

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
fulldist_plt = config_params["EVAL_MODEL"]["fulldist_plt"]
io_frac_plt = config_params["EVAL_MODEL"]["io_frac_plt"]
dens_prf_nu_split = config_params["EVAL_MODEL"]["dens_prf_nu_split"]
dens_prf_macc_split = config_params["EVAL_MODEL"]["dens_prf_macc_split"]

###############################################################################################################

if __name__ == "__main__":    
    client = setup_client()
    
    feature_columns = get_feature_labels(model_sims[0],features)
    
    comb_model_sims = get_combined_name(model_sims) 
        
    model_name = get_model_name(model_type, model_sims, hpo_done=config_params["OPTIMIZE"]["hpo"], opt_param_dict=config_params["OPTIMIZE"])    
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
            
        #TODO check that the right sims and datasets are chosen
        print("Testing on:", curr_test_sims)
        # Loop through and/or for Train/Test/All datasets and evaluate the model
        for dset_name in eval_datasets:
            with timed("Model Evaluation on " + dset_name + " dataset"): 
                #TODO Manage to deal with different cosmologies for each sim  
                dset_params = load_pickle(ML_dset_path + curr_test_sims[0] + "/dset_params.pickle")
                sim_cosmol = dset_params["cosmology"]

                plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
                create_directory(plot_loc)

                # Load the halo information
                halo_files = []
                halo_dfs = []
                if dset_name == "Full":    
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + "Train" + "/halo_info/"))
                        halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/"))
                else:
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + dset_name + "/halo_info/"))

                halo_df = pd.concat(halo_dfs)
                
                # Load the particle information
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,sim_cosmol,limit_files=False)

                X = data[feature_columns]
                y = data[target_column]
                
                with timed(f"Predictions for {y.size.compute():.3e} particles"):
                    preds = make_preds(client, bst, X)
    
                eval_model(model_info, preds, use_sims=curr_test_sims, dst_type=dset_name, X=X, y=y, halo_ddf=halo_df, sim_cosmol=sim_cosmol, plot_save_loc=plot_loc,dens_prf=dens_prf_plt,missclass=misclass_plt,\
                    full_dist=fulldist_plt,io_frac=io_frac_plt,split_nu=dens_prf_nu_split,split_macc=dens_prf_macc_split)
                del data 
                del X
                del y
        
        save_pickle(model_info,model_fldr_loc + "model_info.pickle")

    client.close()
