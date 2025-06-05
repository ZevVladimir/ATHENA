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
import argparse

from utils.data_and_loading_functions import create_directory, timed, load_pickle, save_pickle, load_config
from utils.ML_support import setup_client, get_combined_name, load_data, extract_snaps, get_model_name, get_feature_labels

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

test_dset_frac = config_params["DSET_CREATE"]["test_dset_frac"]

features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]
model_sims = config_params["TRAIN_MODEL"]["model_sims"]
model_type = config_params["TRAIN_MODEL"]["model_type"]
tree_err = config_params["TRAIN_MODEL"]["tree_err"]
file_lim = config_params["TRAIN_MODEL"]["file_lim"]

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
    
    dset_params = load_pickle(ML_dset_path + model_sims[0] + "/dset_params.pickle")
    sim_cosmol = dset_params["cosmology"]
    all_tdyn_steps = dset_params["t_dyn_steps"]
    feature_columns = get_feature_labels(features,all_tdyn_steps)
    all_snaps = extract_snaps(model_sims[0])
    
    comb_model_sims = get_combined_name(model_sims) 
        
    model_name = get_model_name(model_type, model_sims)    
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
        with timed("Loading Datasets"):
            # Do desired adjustments to data if selected, otherwise just load the data normally
            if file_lim > 0:
                lim_train_files = True
            else:
                lim_train_files = False
            train_data,scale_pos_weight = load_data(client,model_sims,"Train",sim_cosmol,prime_snap=all_snaps[0],filter_nu=False,limit_files=lim_train_files)
            test_data,scale_pos_weight = load_data(client,model_sims,"Test",sim_cosmol,prime_snap=all_snaps[0],filter_nu=False,limit_files=False)
                
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            print("Calculated Scale Position Weight:",scale_pos_weight)
            
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            
            create_directory(model_fldr_loc)
            create_directory(gen_plot_save_loc)
            
            # Construct the DaskDMatrix used for training
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
