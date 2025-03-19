from dask.distributed import Client
import xgboost as xgb
import pickle
import os
import numpy as np
import json
import multiprocessing as mp
import pandas as pd

from utils.ML_support import setup_client, get_combined_name, reform_dataset_dfs, load_data, eval_model
from utils.data_and_loading_functions import create_directory, timed, save_pickle
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")

on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")

ML_dset_path = config["PATHS"]["ML_dset_path"]
path_to_models = config["PATHS"]["path_to_models"]

model_sims = json.loads(config.get("TRAIN_MODEL","model_sims"))
dask_task_cpus = config.getint("XGBOOST","dask_task_cpus")
model_type = config["TRAIN_MODEL"]["model_type"]
feature_columns = json.loads(config.get("TRAIN_MODEL","feature_columns"))
target_column = json.loads(config.get("TRAIN_MODEL","target_columns"))

test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))

reduce_rad = config.getfloat("XGBOOST","reduce_rad")
reduce_perc = config.getfloat("XGBOOST", "reduce_perc")

weight_rad = config.getfloat("XGBOOST","weight_rad")
min_weight = config.getfloat("XGBOOST","min_weight")
opt_wghts = config.getboolean("XGBOOST","opt_wghts")
opt_scale_rad = config.getboolean("XGBOOST","opt_scale_rad")

dens_prf_plt = config.getboolean("XGBOOST","dens_prf_plt")
misclass_plt = config.getboolean("XGBOOST","misclass_plt")
fulldist_plt = config.getboolean("XGBOOST","fulldist_plt")
io_frac_plt = config.getboolean("XGBOOST","io_frac_plt")
dens_prf_nu_split = config.getboolean("XGBOOST","dens_prf_nu_split")

###############################################################################################################

if __name__ == "__main__":    
    client = setup_client()
    
    # Adjust name based off what things are being done to the model. This keeps each model unique
    model_comb_name = get_combined_name(model_sims) 
    scale_rad=False
    use_weights=False
    if reduce_rad > 0 and reduce_perc > 0:
        scale_rad = True
    if weight_rad > 0 and min_weight > 0:
        use_weights=True    
    
    model_dir = model_type
    
    if scale_rad:
        model_dir += "scl_rad" + str(reduce_rad) + "_" + str(reduce_perc)
    if use_weights:
        model_dir += "wght" + str(weight_rad) + "_" + str(min_weight)
    
    model_save_loc = path_to_models + model_comb_name + "/" + model_dir + "/"

    # Try loading the model if it can't be thats an error!
    try:
        bst = xgb.Booster()
        bst.load_model(model_save_loc + model_dir + ".json")
        bst.set_param({"device": "cuda:0"})
        print("Loaded Model Trained on:",model_sims)
    except:
        print("Couldn't load Booster Located at: " + model_save_loc + model_dir + ".json")
    
    # Try loading the model info it it can't that's an error!
    try:
        with open(model_save_loc + "model_info.pickle", "rb") as pickle_file:
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
                plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
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
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)

                X = data[feature_columns]
                y = data[target_column]

                eval_model(model_info, client, bst, use_sims=curr_test_sims, dst_type=dset_name, X=X, y=y, halo_ddf=halo_df, plot_save_loc=plot_loc,dens_prf=dens_prf_plt,missclass=misclass_plt,\
                    full_dist=fulldist_plt,io_frac=io_frac_plt,split_nu=dens_prf_nu_split)
                del data 
                del X
                del y
        
        save_pickle(model_info,model_save_loc + "model_info.pickle")

    client.close()
