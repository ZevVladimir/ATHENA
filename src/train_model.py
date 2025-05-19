import dask.dataframe as dd

import xgboost as xgb
from xgboost import dask as dxgb
    
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd

from utils.data_and_loading_functions import create_directory, timed, save_pickle, load_config
from utils.ML_support import setup_client, get_combined_name, load_data, reform_dataset_dfs, optimize_weights, optimize_scale_rad, weight_by_rad, scale_by_rad, get_model_name, get_feature_labels

##################################################################################################################
# LOAD CONFIG PARAMETERS
config_params = load_config(os.getcwd() + "/config.ini")

ML_dset_path = config_params["PATHS"]["ml_dset_path"]
path_to_models = config_params["PATHS"]["path_to_models"]

test_dset_frac = config_params["DSET_CREATE"]["test_dset_frac"]

features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]
model_sims = config_params["TRAIN_MODEL"]["model_sims"]
model_type = config_params["TRAIN_MODEL"]["model_type"]
tree_err = config_params["TRAIN_MODEL"]["tree_err"]

retrain = config_params["MISC"]["retrain_model"]

dens_prf_plt = config_params["EVAL_MODEL"]["dens_prf_plt"]
misclass_plt = config_params["EVAL_MODEL"]["misclass_plt"]
fulldist_plt = config_params["EVAL_MODEL"]["fulldist_plt"]
dens_prf_nu_split = config_params["EVAL_MODEL"]["dens_prf_nu_split"]

def evaluate_accuracy_and_speed(model, dtest, max_trees):
    accuracies = []
    times = []
    
    for num_trees in range(1, max_trees + 1):
        # Start the timer
        start_time = time.time()
        
        # Make predictions using the first 'num_trees' trees
        preds = model.predict(dtest, iteration_range=(0,num_trees))
        
        # End the timer and calculate time taken
        elapsed_time = time.time() - start_time
        
        # Convert probabilities to binary class predictions
        preds = np.round(preds)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, preds)
        
        # Append results
        accuracies.append(accuracy)
        times.append(elapsed_time)
    
    return accuracies, times

if __name__ == "__main__":        
    client = setup_client()
    
    feature_columns = get_feature_labels(model_sims[0],features)
    
    comb_model_sims = get_combined_name(model_sims) 
        
    model_name = get_model_name(model_type, model_sims, hpo_done=config_params["OPTIMIZE"]["hpo"], opt_param_dict=config_params["OPTIMIZE"])    
    model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"
    model_save_loc = model_fldr_loc + model_name + ".json"
    gen_plot_save_loc = model_fldr_loc + "plots/"
     
    # See if there is already a parameter file for the model
    if os.path.isfile(model_fldr_loc + "model_info.pickle") and retrain < 2:
        with open(model_fldr_loc + "model_info.pickle", "rb") as pickle_file:
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
    if os.path.isfile(model_save_loc) and retrain < 1:
        bst = xgb.Booster()
        bst.load_model(model_save_loc)
        params = model_info.get('Training Info',{}).get('Training Params')
        print("Loaded Booster")
    else:
        print("Making Datasets") 
        
        use_scale_rad = config_params["OPTIMIZE"]["opt_scale_rad"]
        reduce_rad = config_params["OPTIMIZE"]["reduce_rad"]
        reduce_perc = config_params["OPTIMIZE"]["reduce_perc"]
        
        use_weights = config_params["OPTIMIZE"]["opt_weights"]
        weight_rad = config_params["OPTIMIZE"]["weight_rad"]
        min_weight = config_params["OPTIMIZE"]["min_weight"]
        weight_exp = config_params["OPTIMIZE"]["weight_exp"]
            
        scale_rad_info = {
            "Used scale_rad": use_scale_rad,
            "Reducing radius start": reduce_rad,
            "Reducing radius percent": reduce_perc,
        }
            
        weight_rad_info = {
            "Used weighting by radius": use_weights,
            "Start for weighting": weight_rad,
            "Minimum weight assign": min_weight,
            "Exponent for weighting function": weight_exp
        }
            
        # Do desired adjustments to data if selected, otherwise just load the data normally
        if use_scale_rad and use_weights:
            train_data,scale_pos_weight,train_weights,bin_edges = load_data(client,model_sims,"Train",scale_rad=use_scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        elif use_scale_rad and not use_weights:
            train_data,scale_pos_weight,bin_edges = load_data(client,model_sims,"Train",scale_rad=use_scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        elif not use_scale_rad and use_weights:
            train_data,scale_pos_weight,train_weights = load_data(client,model_sims,"Train",scale_rad=use_scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        else:
            train_data,scale_pos_weight = load_data(client,model_sims,"Train",scale_rad=use_scale_rad,use_weights=use_weights,filter_nu=False,limit_files=True)
        test_data,scale_pos_weight = load_data(client,model_sims,"Test",scale_rad=False,use_weights=False,filter_nu=False,limit_files=False)
            
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        print("Calculated Scale Position Weight:",scale_pos_weight)
        
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]

        # Perform optimization on either option for adjustment of the data if desired
        if use_weights or use_scale_rad:
            params = {
                    "verbosity": 0,
                    "tree_method": "hist",
                    "scale_pos_weight":scale_pos_weight,
                    "max_depth":4,
                    "device": "cuda",
                    "subsample": 0.5,
                    'objective': 'binary:logistic',
                    'eval_metric': 'error',
                    }

            halo_files = []

            halo_dfs = []
            for sim in model_sims:
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/Train/halo_info/"))

            halo_df = pd.concat(halo_dfs)

        # Optimize the weighting of particles
        if use_weights:          
            opt_weight_rad, opt_min_weight, opt_weight_exp = optimize_weights(client,params,train_data,halo_df,model_sims,feature_columns,target_column)
            
            train_weights = weight_by_rad(X_train["p_Scaled_radii"].values.compute(),y_train.compute().values.flatten(), opt_weight_rad, opt_min_weight)
            dask_weights = []
            scatter_weight = client.scatter(train_weights)
            dask_weight = dd.from_delayed(scatter_weight) 
            dask_weights.append(dask_weight)
                
            train_weights = dd.concat(dask_weights)
            
            train_weights = train_weights.repartition(npartitions=X_train.npartitions)
            
            weight_rad_info = {
                "Used weighting by radius": use_weights,
                "Start for weighting": opt_weight_rad,
                "Minimum weight assign": opt_min_weight,
                "Weight exponent": opt_weight_exp
            }
        
        # Optimize the scaling of the number of particles at each radius 
        if use_scale_rad:
            num_bins=100
            bin_edges = np.logspace(np.log10(0.001),np.log10(10),num_bins)
            opt_reduce_rad, opt_reduce_perc = optimize_scale_rad(client,params,train_data,halo_df,model_sims,feature_columns,target_column)
            scld_data = scale_by_rad(train_data.compute(),bin_edges,opt_reduce_rad,opt_reduce_perc)
            
            scatter_train = client.scatter(scld_data)
            scld_train_data = dd.from_delayed(scatter_train)
            
            X_train = scld_train_data[feature_columns]
            y_train = scld_train_data[target_column]
            
            scale_rad_info = {
            "Used scale_rad": use_scale_rad,
            "Reducing radius start": opt_reduce_rad,
            "Reducing radius percent": opt_reduce_perc,
            }
            
            model_name = get_model_name(model_type, model_sims, hpo_done=config_params["OPTIMIZE"]["hpo"], opt_param_dict=config_params["OPTIMIZE"])    
            model_fldr_loc = path_to_models + comb_model_sims + "/" + model_name + "/"
            model_save_loc = model_fldr_loc + model_name + ".json"
            gen_plot_save_loc = model_fldr_loc + "plots/"
        
        model_info["Training Data Adjustments"] = {
            "Scaling by Radii": scale_rad_info,
            "Weighting by Radii": weight_rad_info,
        }
        
        create_directory(model_fldr_loc)
        create_directory(gen_plot_save_loc)
        
        # Construct the DaskDMatrix used for training (if using different weights input those as well)
        if use_weights:
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train, weight=train_weights)
        else:
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
        dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)
        
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
                'eval_metric': 'error',
                }
            model_info['Training Info']={
                'Fraction of Training Data Used': test_dset_frac,
                'Training Params': params}

        # Train and save the model
        #TODO make params selectable params?
        num_trees = 100
        with timed("Trained Model"):
            print("Starting train using params:", params)
            output = dxgb.train(
                client,
                params,
                dtrain,
                num_boost_round=num_trees,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=10, 
                )
            bst = output["booster"]
            history = output["history"]
            bst.save_model(model_save_loc)
            save_pickle(model_info,model_fldr_loc + "model_info.pickle")
            
        if tree_err:
            with timed("Evaluate Tree Error"):
                #TODO make this optional and document evaluation of model
                #TODO see if it is possible to save the history
                # Get accuracy at each boosting round
                # Extract training and validation error (1 - accuracy) at each round
                train_errors = history['train']['error']
                test_errors = history['test']['error']

                # Calculate accuracies
                train_accuracies = [1 - error for error in train_errors]
                test_accuracies = [1 - error for error in test_errors]

                # Plot or analyze accuracy vs. number of trees (boosting rounds)
                plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
                plt.plot(range(len(test_accuracies)), test_accuracies, label='Test Accuracy')
                plt.xlabel('Number of Trees')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig(gen_plot_save_loc + "train_test_acc.png")

                # You can also inspect the final accuracy after training
                final_train_accuracy = train_accuracies[-1]
                final_test_accuracy = test_accuracies[-1]
                print(f"Final train accuracy: {final_train_accuracy}")
                print(f"Final test accuracy: {final_test_accuracy}")
                
                
                X_test_local = X_test.compute()
                y_test_local = y_test.compute()
                dtest = xgb.DMatrix(X_test_local, label=y_test_local)

                accuracies, times = evaluate_accuracy_and_speed(bst, dtest, max_trees=num_trees)

                # Plot accuracy vs number of trees
                plt.figure(figsize=(12, 6))

                # Subplot for accuracy
                plt.subplot(1, 2, 1)
                plt.plot(range(1, 101), accuracies, label='Test Accuracy', color='blue')
                plt.xlabel('Number of Trees')
                plt.ylabel('Accuracy')
                plt.legend()

                # Subplot for time
                plt.subplot(1, 2, 2)
                plt.plot(range(1, 101), times, label='Prediction Time', color='red')
                plt.xlabel('Number of Trees')
                plt.ylabel('Time (seconds)')
                plt.legend()

                plt.tight_layout()
                plt.savefig(gen_plot_save_loc + "tree_time_acc.png")

    
    client.close()
