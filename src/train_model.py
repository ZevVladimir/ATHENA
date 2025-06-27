import xgboost as xgb
from xgboost import dask as dxgb
import pickle
import os
import argparse

from src.utils.util_fxns import create_directory, timed, load_pickle, save_pickle, load_config, load_ML_dsets, load_all_sim_cosmols, load_all_tdyn_steps
from src.utils.ML_fxns import setup_client, get_combined_name, extract_snaps, get_model_name, get_feature_labels

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

eval_train_dsets = config_params["TRAIN_MODEL"]["eval_train_dsets"]
features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]
model_sims = config_params["TRAIN_MODEL"]["model_sims"]
model_type = config_params["TRAIN_MODEL"]["model_type"]
file_lim = config_params["TRAIN_MODEL"]["file_lim"]
ntrees = config_params["TRAIN_MODEL"]["ntrees"]
n_early_stopping_rounds = config_params["TRAIN_MODEL"]["n_early_stopping_rounds"]

retrain = config_params["MISC"]["retrain_model"]

if __name__ == "__main__":        
    client = setup_client()
    
    all_sim_cosmol_list = load_all_sim_cosmols(model_sims)
    all_tdyn_steps_list = load_all_tdyn_steps(model_sims)

    feature_columns = get_feature_labels(features,all_tdyn_steps_list[0])
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
            train_data,train_scale_pos_weight = load_ML_dsets(client,model_sims,"Train",all_sim_cosmol_list,prime_snap=all_snaps[0],file_lim=file_lim,filter_nu=False)
                
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]

            create_directory(model_fldr_loc)
            create_directory(gen_plot_save_loc)
            
            # Construct the DaskDMatrix used for evaluatin the model
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
            
            eval_list = []
            for dset in eval_train_dsets:
                curr_data,curr_scale_pos_weight = load_ML_dsets(client,model_sims,dset,all_sim_cosmol_list,prime_snap=all_snaps[0],file_lim=file_lim,filter_nu=False)
                
                curr_X = curr_data[feature_columns]
                curr_y = curr_data[target_column]
                
                # Construct the DaskDMatrix used for evaluatin the model
                eval_list.append((xgb.dask.DaskDMatrix(client, curr_X, curr_y),dset))

        if 'Training Info' in model_info: 
            params = model_info.get('Training Info',{}).get('Training Params')
        else:
            params = {
                "verbosity": 1,
                "tree_method": "hist",
                "scale_pos_weight":train_scale_pos_weight,
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
                num_boost_round=ntrees,
                evals=eval_list,
                early_stopping_rounds=n_early_stopping_rounds, 
                )
            bst = output["booster"]
            history = output["history"]
            bst.save_model(model_save_loc)
            save_pickle(model_info,model_fldr_loc + "model_info.pickle")
    
    client.close()
