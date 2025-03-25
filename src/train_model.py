import dask.dataframe as dd

import xgboost as xgb
from xgboost import dask as dxgb
    
import json
import pickle
import os
import numpy as np
import pandas as pd

from utils.data_and_loading_functions import create_directory, timed, save_pickle
from utils.ML_support import setup_client, get_combined_name, load_data, reform_dataset_dfs, optimize_weights, optimize_scale_rad, weight_by_rad, scale_by_rad, eval_model

##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")

ML_dset_path = config["PATHS"]["ML_dset_path"]
path_to_models = config["PATHS"]["path_to_models"]

test_dset_frac = config.getfloat("DSET_CREATE","test_dset_frac")

feature_columns = json.loads(config.get("TRAIN_MODEL","feature_columns"))
target_column = json.loads(config.get("TRAIN_MODEL","target_column"))
model_sims = json.loads(config.get("TRAIN_MODEL","model_sims"))
model_type = config["TRAIN_MODEL"]["model_type"]

retrain = config.getint("MISC","retrain_model")

dens_prf_plt = config.getboolean("EVAL_MODEL","dens_prf_plt")
misclass_plt = config.getboolean("EVAL_MODEL","misclass_plt")
fulldist_plt = config.getboolean("EVAL_MODEL","fulldist_plt")
dens_prf_nu_split = config.getboolean("EVAL_MODEL","dens_prf_nu_split")

reduce_rad = config.getfloat("OPTIMIZE","reduce_rad")
reduce_perc = config.getfloat("OPTIMIZE", "reduce_perc")
weight_rad = config.getfloat("OPTIMIZE","weight_rad")
min_weight = config.getfloat("OPTIMIZE","min_weight")
opt_wghts = config.getboolean("OPTIMIZE","opt_wghts")
opt_scale_rad = config.getboolean("OPTIMIZE","opt_scale_rad")

if __name__ == "__main__":        
    client = setup_client()
    # Adjust name based off what things are being done to the model. This keeps each model unique
    scale_rad=False
    use_weights=False
    
    if reduce_rad > 0 and reduce_perc > 0 and not opt_scale_rad:
        scale_rad = True
    if weight_rad > 0 and min_weight > 0 and not opt_wghts:
        use_weights=True
        
    combined_name = get_combined_name(model_sims) 
        
    model_dir = model_type

    if scale_rad:
        model_dir += "scl_rad" + str(reduce_rad) + "_" + str(reduce_perc)
    if use_weights:
        model_dir += "wght" + str(weight_rad) + "_" + str(min_weight)

    model_save_loc = path_to_models + combined_name + "/" + model_dir + "/"
    gen_plot_save_loc = model_save_loc + "plots/"
     
    # See if there is already a parameter file for the model
    if os.path.isfile(model_save_loc + "model_info.pickle") and retrain < 2:
        with open(model_save_loc + "model_info.pickle", "rb") as pickle_file:
            model_info = pickle.load(pickle_file)
    else:
        train_sims = ""
        for i,sim in enumerate(model_sims):
            if i != len(model_sims) - 1:
                train_sims += sim + ", "
            else:
                train_sims += sim
        model_info = {
            'Misc Info':{
                'Model trained on': train_sims,
                }}
    
    # Load previously trained booster or start the training process
    if os.path.isfile(model_save_loc + model_dir + ".json") and retrain < 1:
        bst = xgb.Booster()
        bst.load_model(model_save_loc + model_dir + ".json")
        params = model_info.get('Training Info',{}).get('Training Params')
        print("Loaded Booster")
    else:
        print("Making Datasets") 
            
        scale_rad_info = {
            "Used scale_rad": scale_rad,
            "Reducing radius start": reduce_rad,
            "Reducing radius percent": reduce_perc,
        }
            
        weight_rad_info = {
            "Used weighting by radius": use_weights,
            "Optimized weights": opt_wghts,
            "Start for weighting": weight_rad,
            "Minimum weight assign": min_weight
        }
            
        # Do desired adjustments to data if selected, otherwise just load the data normally
        if scale_rad and use_weights:
            train_data,scale_pos_weight,train_weights,bin_edges = load_data(client,model_sims,"Train",scale_rad=scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        elif scale_rad and not use_weights:
            train_data,scale_pos_weight,bin_edges = load_data(client,model_sims,"Train",scale_rad=scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        elif not scale_rad and use_weights:
            train_data,scale_pos_weight,train_weights = load_data(client,model_sims,"Train",scale_rad=scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        else:
            train_data,scale_pos_weight = load_data(client,model_sims,"Train",scale_rad=scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)

        print("Calculated Scale Position Weight:",scale_pos_weight)
        
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]

        # Perform optimization on either option for adjustment of the data if desired
        if opt_wghts or opt_scale_rad:
            params = {
                    "verbosity": 0,
                    "tree_method": "hist",
                    "scale_pos_weight":scale_pos_weight,
                    "max_depth":4,
                    "device": "cuda",
                    "subsample": 0.5,
                    'objective': 'binary:logistic',
                    }

            halo_files = []

            halo_dfs = []
            for sim in model_sims:
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/Train/halo_info/"))

            halo_df = pd.concat(halo_dfs)

        # Optimize the weighting of particles
        if opt_wghts:  
            use_weights = True          
            opt_weight_rad, opt_min_weight, opt_weight_exp = optimize_weights(client,params,train_data,halo_df,model_sims,feature_columns,target_column)
            
            train_weights = weight_by_rad(X_train["p_Scaled_radii"].values.compute(),y_train.compute().values.flatten(), opt_weight_rad, opt_min_weight)
            dask_weights = []
            scatter_weight = client.scatter(train_weights)
            dask_weight = dd.from_delayed(scatter_weight) 
            dask_weights.append(dask_weight)
                
            train_weights = dd.concat(dask_weights)
            
            train_weights = train_weights.repartition(npartitions=X_train.npartitions)
            
            model_dir += "wght" + str(np.round(opt_weight_rad,3)) + "_" + str(np.round(opt_min_weight,3)) + "_" + str(np.round(opt_weight_exp,3))
            weight_rad_info = {
                "Used weighting by radius": use_weights,
                "Optimized weights": opt_wghts,
                "Start for weighting": opt_weight_rad,
                "Minimum weight assign": opt_min_weight,
                "Weight exponent": opt_weight_exp
            }
        
        # Optimize the scaling of the number of particles at each radius 
        if opt_scale_rad:
            scale_rad = True
            num_bins=100
            bin_edges = np.logspace(np.log10(0.001),np.log10(10),num_bins)
            opt_reduce_rad, opt_reduce_perc = optimize_scale_rad(client,params,train_data,halo_df,model_sims,feature_columns,target_column)
            scld_data = scale_by_rad(train_data.compute(),bin_edges,opt_reduce_rad,opt_reduce_perc)
            
            scatter_train = client.scatter(scld_data)
            scld_train_data = dd.from_delayed(scatter_train)
            
            X_train = scld_train_data[feature_columns]
            y_train = scld_train_data[target_column]
            
            # model_name += "scl_rad" + str(np.round(opt_reduce_rad,3)) + "_" + str(np.round(opt_reduce_perc,3))
            model_dir += "scl_rad" + str(np.round(opt_reduce_rad,3)) + "_" + str(np.round(opt_reduce_perc,3))
            scale_rad_info = {
            "Used scale_rad": scale_rad,
            "Optimized Scale by Radius": opt_scale_rad,
            "Reducing radius start": opt_reduce_rad,
            "Reducing radius percent": opt_reduce_perc,
            }
        
        model_info["Training Data Adjustments"] = {
            "Scaling by Radii": scale_rad_info,
            "Weighting by Radii": weight_rad_info,
        }
        
        create_directory(model_save_loc)
        create_directory(gen_plot_save_loc)
        
        # Construct the DaskDMatrix used for training (if using different weights input those as well)
        if use_weights:
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train, weight=train_weights)
        else:
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
        
        if 'Training Info' in model_info: 
            params = model_info.get('Training Info',{}).get('Training Params')
        else:
            params = {
                "verbosity": 1,
                "tree_method": "hist",
                "scale_pos_weight":scale_pos_weight,
                "max_depth":4,
                "device": "cuda",
                "subsample": 0.5,
                'objective': 'binary:logistic',
                }
            model_info['Training Info']={
                'Fraction of Training Data Used': test_dset_frac,
                'Training Params': params}
        
        # Train and save the model
        with timed("Trained Model"):
            print("Starting train using params:", params)
            output = dxgb.train(
                client,
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train")],
                early_stopping_rounds=10,      
                )
            bst = output["booster"]
            history = output["history"]
            bst.save_model(model_save_loc + model_dir + ".json")
            save_pickle(model_info,model_save_loc + "model_info.pickle")
        
            del dtrain
    
    # Evaluate the model on the corresponding testing dataset to what the model was trained on
    with timed("Model Evaluation"):
        plot_loc = model_save_loc + "Test_" + combined_name + "/plots/"
        create_directory(plot_loc)
        
        halo_files = []
        halo_dfs = []
        
        for sim in model_sims:
            halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/Test/halo_info/"))

        halo_df = pd.concat(halo_dfs)
        
        test_data,test_scale_pos_weight = load_data(client,model_sims,"Test",limit_files=False)
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        eval_model(model_info, client, bst, use_sims=model_sims, dst_type="Test", X=X_test, y=y_test, halo_ddf=halo_df, plot_save_loc=plot_loc, dens_prf=dens_prf_plt,missclass=misclass_plt,full_dist=fulldist_plt,split_nu=dens_prf_nu_split)
   
    # Save the updated model and model info file
    bst.save_model(model_save_loc + model_dir + ".json")
    save_pickle(model_info,model_save_loc + "model_info.pickle")
    
    client.close()
