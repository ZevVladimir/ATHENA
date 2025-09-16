import xgboost as xgb
import argparse
import numpy as np
import os
import pickle
import re
import pandas as pd

from .util_fxns import load_config
from .util_fxns import split_sparta_hdf5_name, timed, parse_ranges

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
rand_seed = config_params["MISC"]["random_seed"]
curr_sparta_file = config_params["SPARTA_DATA"]["curr_sparta_file"]
debug_indiv_dens_prf = config_params["MISC"]["debug_indiv_dens_prf"]

snap_path = config_params["SNAP_DATA"]["snap_path"]

SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]
pickled_path = config_params["PATHS"]["pickled_path"]
ML_dset_path = config_params["PATHS"]["ml_dset_path"]
debug_plt_path = config_params["PATHS"]["debug_plt_path"]

on_zaratan = config_params["ENVIRONMENT"]["on_zaratan"]
use_gpu = config_params["ENVIRONMENT"]["use_gpu"]

file_lim = config_params["TRAIN_MODEL"]["file_lim"]

plt_nu_splits = config_params["EVAL_MODEL"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)

plt_macc_splits = config_params["EVAL_MODEL"]["plt_macc_splits"]
plt_macc_splits = parse_ranges(plt_macc_splits)
min_halo_nu_bin = config_params["EVAL_MODEL"]["min_halo_nu_bin"]

sim_name, search_name = split_sparta_hdf5_name(curr_sparta_file)
snap_path = snap_path + sim_name + "/"

###############################################################################################################

def make_preds(bst, X, threshold = 0.5):
    preds = bst.predict(xgb.DMatrix(X))
    preds = (preds >= threshold).astype(np.int8)
    return preds

def extract_snaps(sim_name):
    match = re.search(r'((?:_\d+)+)$', sim_name)
    if not match:
        return []
    number_strs = match.group(1).split('_')[1:]  # First split is an empty string
    snap_list = [int(num) for num in number_strs]
    snap_list.sort()
    snap_list.reverse()
    return snap_list

def get_feature_labels(features, tdyn_steps):
    all_features = []
    for feature in features:
        all_features.append("p_" + feature)
    for t_dyn_step in tdyn_steps:
        for feature in features:
            all_features.append(str(t_dyn_step) + "_" + feature)
    
    return all_features

# This function prints out all the model information such as the training simulations, training parameters, and results
# The results are split by simulation that the model was tested on and reports the misclassification rate on each population
def print_model_prop(model_dict, indent=''):
    # If using from command line and passing path to the pickled dictionary instead of dict load the dict from the file path
    if isinstance(model_dict, str):
        with open(model_dict, "rb") as file:
            model_dict = pickle.load(file)
        
    for key, value in model_dict.items():
        # use recursion for dictionaries within the dictionary
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            print_model_prop(value, indent + '  ')
        # if the value is instead a list join them all together with commas
        elif isinstance(value, list):
            print(f"{indent}{key}: {', '.join(map(str, value))}")
        else:
            print(f"{indent}{key}: {value}")

def get_model_name(model_type, trained_on_sims):
    model_name = f"{model_type}_{get_combined_name(trained_on_sims)}"        
    return model_name

def get_combined_name(model_sims):
    combined_name = ""
    for i,sim in enumerate(model_sims):
        split_string = sim.split('_')
        snap_list = extract_snaps(sim)
        
        r_patt = r'(\d+-\d+|\d+)r'
        r_match = re.search(r_patt,split_string[3])

        
        v_patt = r'(\d+-\d+|\d+)v'
        v_match = re.search(v_patt, split_string[4])


        cond_string = split_string[0] + split_string[1] + split_string[2] + "s" 
        cond_string = cond_string + "_".join(str(x) for x in snap_list)
        # can add these for more information per name
        #+ "r" + r_match.group(1) + "v" + v_match.group(1) + "s" + split_string[5]
        
        combined_name += cond_string
    
    return combined_name
    
def filter_ddf(X, y = None, preds = None, fltr_dic = None, col_names = None, max_size=500):
    with timed("Filter DF"):
        full_filter = None
        if fltr_dic is not None:
            if "X_filter" in fltr_dic:
                for feature, conditions in fltr_dic["X_filter"].items():
                    if not isinstance(conditions, list):
                        conditions = [conditions]
                    for operator, value in conditions:
                        if operator == '>':
                            condition = X[feature] > value               
                        elif operator == '<':
                            condition = X[feature] < value    
                        elif operator == '>=':
                            condition = X[feature] >= value
                        elif operator == '<=':
                            condition = X[feature] <= value
                        elif operator == '==':
                            if value == "nan":
                                condition = X[feature].isna()
                            else:
                                condition = X[feature] == value
                        elif operator == '!=':
                            condition = X[feature] != value
                            
                        full_filter = condition if full_filter is None else full_filter & condition
                
            if "label_filter" in fltr_dic:
                for feature, value in fltr_dic["label_filter"].items():
                    if feature == "sparta":
                        if isinstance(y, (dd.DataFrame, pd.DataFrame)):
                            y = y["Orbit_infall"]      
                        condition = y == value
                    elif feature == "pred":
                        if isinstance(preds, (dd.DataFrame, pd.DataFrame)):
                            preds = preds["preds"]
                        condition = preds == value
                        if isinstance(condition, dd.DataFrame):
                            condition = condition["preds"]
                            condition = condition.reset_index(drop=True)
                            
                    full_filter = condition if full_filter is None else full_filter & condition

            X = X[full_filter]

        nrows = X.shape[0].compute()
            
        if nrows > max_size and max_size > 0:
            sample = max_size / nrows
        else:
            sample = 1.0
            
        if sample > 0 and sample < 1:
            X = X.sample(frac=sample,random_state=rand_seed)
        
        if col_names != None:
            X.columns = col_names
            
        if full_filter is not None:
            full_filter = full_filter.compute()
        # Return the filtered array and the indices of the original array that remain
        return X.compute(),full_filter
  
def filter_df(X, y=None, preds=None, fltr_dic=None, col_names=None, max_size=500, rand_seed=42):
    full_filter = pd.Series(True, index=X.index)

    if fltr_dic is not None:
        if "X_filter" in fltr_dic:
            for feature, conditions in fltr_dic["X_filter"].items():
                if not isinstance(conditions, list):
                    conditions = [conditions]
                for operator, value in conditions:
                    if operator == '>':
                        condition = X[feature] > value
                    elif operator == '<':
                        condition = X[feature] < value
                    elif operator == '>=':
                        condition = X[feature] >= value
                    elif operator == '<=':
                        condition = X[feature] <= value
                    elif operator == '==':
                        if value == "nan":
                            condition = X[feature].isna()
                        else:
                            condition = X[feature] == value
                    elif operator == '!=':
                        condition = X[feature] != value
                    else:
                        raise ValueError(f"Unknown operator: {operator}")
                    full_filter &= condition

        if "label_filter" in fltr_dic:
            for feature, value in fltr_dic["label_filter"].items():
                if feature == "sparta":
                    if isinstance(y, pd.DataFrame):
                        y = y["Orbit_infall"]
                    condition = y == value
                elif feature == "pred":
                    if isinstance(preds, pd.DataFrame):
                        preds = preds["preds"]
                    condition = preds == value
                else:
                    raise ValueError(f"Unknown label_filter feature: {feature}")
                full_filter &= condition

        X = X[full_filter]

    # Sample if needed
    nrows = len(X)
    if nrows > max_size and max_size > 0:
        sample_frac = max_size / nrows
        X = X.sample(frac=sample_frac, random_state=rand_seed)

    if col_names is not None:
        X.columns = col_names

    return X, full_filter  
    
# Can set max_size to 0 to include all the particles
def shap_with_filter(explainer, X, y, preds, fltr_dic = None, col_names = None, max_size=500):
    X_fltr,fltr = filter_df(X, y, preds, fltr_dic = fltr_dic, col_names = col_names, max_size=max_size)
    return explainer(X_fltr), explainer.shap_values(X_fltr), X_fltr
    
    