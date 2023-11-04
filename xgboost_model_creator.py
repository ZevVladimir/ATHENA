
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
import time 
import pickle
import re
import os
from imblearn import under_sampling, over_sampling
from data_and_loading_functions import build_ml_dataset, check_pickle_exist_gadget, choose_halo_split, create_directory
from visualization_functions import *
from typing import List, Callable
from sklearn.preprocessing import LabelEncoder

import configparser
config = configparser.ConfigParser()
config.read("config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]

class Iterator(xgboost.DataIter):
  def __init__(self, svm_file_paths: List[str], inc_labels = True):
    self._file_paths = svm_file_paths
    self._it = 0
    self.inc_labels = inc_labels
    # XGBoost will generate some cache files under current directory with the prefix
    # "cache"
    super().__init__(cache_prefix=os.path.join(".", "cache"))

  def next(self, input_data: Callable):
    """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
    called by XGBoost during the construction of ``DMatrix``

    """
    if self._it == len(self._file_paths):
      # return 0 to let XGBoost know this is the end of iteration
      return 0

    # input_data is a function passed in by XGBoost who has the exact same signature of
    # ``DMatrix``
    with open(self._file_paths[self._it], 'rb') as pickle_file:
        curr_dataset = pickle.load(pickle_file)
    if self.inc_labels:
        input_data(data=curr_dataset[:,2:], label=curr_dataset[:,1])
    else:
        input_data(data=curr_dataset[:,2:])
    self._it += 1
    # Return 1 to let XGBoost know we haven't seen all the files yet.
    return 1

  def reset(self):
    """Reset the iterator to its beginning"""
    self._it = 0

class model_creator:
    def __init__(self, save_location, model_name, radii_splits, radii_loc, rad_vel_loc, tang_vel_loc, keys, paths_to_split_data = None, paths_to_split_val_data = None, dataset_split = False):
        self.name = model_name
        self.save_location = save_location
        self.radii_splits = radii_splits
        self.radii_loc = radii_loc
        self.rad_vel_loc = rad_vel_loc
        self.tang_vel_loc = tang_vel_loc
        self.keys = keys
        self.sub_model_loc = []
        self.model_location = self.save_location + "sub_models/"
        create_directory(self.model_location)
            
        if paths_to_split_val_data != None:
            train_it = Iterator(paths_to_split_data)
            val_it = Iterator(paths_to_split_val_data)
            self.train_dataset = xgboost.DMatrix(train_it)
            self.val_dataset = xgboost.DMatrix(val_it)

    def train_val_split(self):
        # Split the dataset inputted into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(self.dataset[:,2:], self.dataset[:,1], test_size=0.20, random_state=0)
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        
        # mix the data to eliminate any connections 
        self.X_train, self.y_train = self.mix_data(X_train, y_train) 
        self.X_val, self.y_val = self.mix_data(X_val, y_val)
        
    def mix_data(self, x, y):
        # generate random indices and then mix the training and validation sets
        train_rand_idxs = np.random.permutation(x.shape[0])
        x = x[train_rand_idxs]
        y = y[train_rand_idxs]

        return x, y

    def over_sample():
        #TODO implement
        return
    
    def under_sample():
        #TODO implement
        return
    
    def normalize(values):
        for col in range(values.shape[1]):
            values[:,col] = (values[:,col] - values[:,col].min())/(values[:,col].max() - values[:,col].min())
        return values
    
    def standardize(values):
        for col in range(values.shape[1]):
            values[:,col] = (values[:,col] - values[:,col].mean())/values[:,col].std()
        return values
    
    def split_by_dist(self, low_cutoff, high_cutoff, x_dataset, y_dataset):
        # Get the values of X and y within the specified range
        X_within = x_dataset[np.where((x_dataset[:,self.radii_loc] > low_cutoff) & (x_dataset[:,self.radii_loc] < high_cutoff))[0]]
        y_within = y_dataset[np.where((x_dataset[:,self.radii_loc] > low_cutoff) & (x_dataset[:,self.radii_loc] < high_cutoff))[0]]

        return X_within, y_within
    
    def load_ensemble(self):
        for filename in os.listdir(self.save_location + "sub_models/"):      
            self.sub_model_loc.append(self.save_location + "sub_models/" + filename)
    
    def load_model(self, location):
        with open(location, "rb") as pickle_file:
            return pickle.load(pickle_file)
                
    def train_model(self):
        # Create a sub model for each radii split determined
        if len(self.radii_splits) > 1:
            for i in range(len(self.radii_splits) + 1):
                if i == 0:
                    low_cutoff = 0
                else:
                    low_cutoff = self.radii_splits[i - 1]
                if i == len(self.radii_splits):
                    high_cutoff = np.max(self.X_train[:,self.radii_loc])
                else:
                    high_cutoff = self.radii_splits[i]
                
                X, y = self.split_by_dist(low_cutoff, high_cutoff, self.X_train, self.y_train)
                
                curr_model_location = self.model_location + "range_" + str(low_cutoff) + "_" + str(np.round(high_cutoff,2)) + "_" + curr_sparta_file + ".pickle"

                if os.path.exists(curr_model_location) == False:
                    model = None
                    t3 = time.time()
                    
                    # Train and fit each model with gpu
                    model = XGBClassifier(tree_method='gpu_hist', eta = 0.01, n_estimators = 100)
                    
                    le = LabelEncoder()
                    y = y.astype(np.int16)
                    y = le.fit_transform(y)

                    model = model.fit(X, y)

                    t4 = time.time()
                    print("Fitted model", t4 - t3, "seconds")

                    model.save_model(self.model_location + "model.json")
                
                self.sub_model_loc.append(curr_model_location)
        else:
            curr_model_location = self.model_location + "range_all_" + curr_sparta_file + ".json"

            if os.path.exists(curr_model_location) == False:

                model = None
                t3 = time.time()
                
                # Train and fit each model with gpu
                model = XGBClassifier(tree_method='gpu_hist', eta = 0.01, n_estimators = 100)
                
                # le = LabelEncoder()
                # y = y.astype(np.int16)
                # y = le.fit_transform(y)

                model = model.fit(self.train_dataset)

                t4 = time.time()
                print("Fitted model", t4 - t3, "seconds")

                model.save_model(curr_model_location)
                #pickle.dump(model, open(curr_model_location, "wb"), pickle.HIGHEST_PROTOCOL)
            
            self.sub_model_loc.append(curr_model_location)

    def ensemble_predict(self, dataset = None):
        # If there is no dataset submitted then just use the validation sets otherwise predict on the inputted one
        try:
            use_dataset = dataset
        except:
            print("Using Validation Set")
            use_dataset = self.val_dataset
    
        # for each submodel get the predictions
        all_predicts = np.zeros((use_dataset.shape[0],(len(self.sub_model_loc)*2)))
        for i, loc in enumerate(self.sub_model_loc):
            model = xgboost.Booster()
            model.load_model(self.sub_model_loc)
            all_predicts[:,2*i:(2*i+2)] = model.predict_proba(use_dataset)

        # Then determine which class each particle belongs to based of each model's prediction
        all_predicts = self.det_class_argmax(all_predicts)
        le = LabelEncoder()
        use_labels = use_labels.astype(np.int16)
        use_labels = le.fit_transform(use_labels)
        print(classification_report(use_labels, all_predicts))

        return all_predicts
    
    def predict_by_model(self, dataset):
        try:
            use_dataset = dataset[:,2:]
            use_labels = dataset[:,1]
        except:
            print("Using Validation Set")
            use_dataset = self.X_val
            use_labels = self.y_val
            
        all_preds = np.ones(use_dataset.shape[0]) * -1
        
        start = 0
        for i in range(len(self.radii_splits) + 1):
            if i == 0:
                low_cutoff = 0
            else:
                low_cutoff = self.radii_splits[i - 1]
            if i == len(self.radii_splits):
                high_cutoff = np.max(self.X_train[:,self.radii_loc])
            else:
                high_cutoff = self.radii_splits[i]
            
            X, y = self.split_by_dist(low_cutoff, high_cutoff, use_dataset, use_labels)
            
            curr_model_location = self.model_location + "range_" + str(low_cutoff) + "_" + str(np.round(high_cutoff,2)) + "_" + self.curr_sparta_file + ".pickle"
            with open(curr_model_location, "rb") as pickle_file:
                curr_model = pickle.load(pickle_file)
            predicts = curr_model.predict_proba(X)
            
            all_preds[start:start+X.shape[0]] = self.det_class_by_model(predicts)
            start = start + X.shape[0]
        
        self.predicts = all_preds
        le = LabelEncoder()
        use_labels = use_labels.astype(np.int16)
        use_labels = le.fit_transform(use_labels)
        print(classification_report(use_labels, self.predicts))
    
    def det_class_argmax(self, predicts):
        # Determine which class a particle is based off which model gave the highest probability
        #TODO add option to use an average of predictions instead
        pred_loc = np.argmax(predicts, axis = 1)

        final_predicts = np.zeros(predicts.shape[0])
        final_predicts[np.where((pred_loc % 2) != 0)] = 1

        return final_predicts
        
    def det_class_by_model(self, predicts):
        pred = np.argmax(predicts, axis = 1)
        return pred
        

    def graph(self, corr_matrix = False, feat_imp = False):
        # implement functionality to create graphs of data if wanted
        if corr_matrix:
            #TODO fix so dataframe is either not used or created
            graph_correlation_matrix(self.dataset_df, self.save_location, title = "model_num_params_" + str(len(self.keys) - 2),show = False, save = True)
        if feat_imp:
            for i, loc in enumerate(self.sub_model_loc):
                model = self.load_model(loc)
                if i == 0:
                    low_cutoff = 0
                else:
                    low_cutoff = self.radii_splits[i - 1]
                if i == len(self.radii_splits):
                    high_cutoff = np.max(self.X_train[:,self.radii_loc])
                else:
                    high_cutoff = self.radii_splits[i]
                if len(self.sub_model_loc) > 1:
                    graph_feature_importance(np.array(self.keys[2:]), model.feature_importances_, "radii: " + str(low_cutoff) + "_" + str(np.round(high_cutoff,2)), False, True, self.save_location)
                else:
                    graph_feature_importance(np.array(self.keys[2:]), model.feature_importances_, "all_radii", False, True, self.save_location)
        
    def get_predicts(self):
        return self.predicts
    
    def get_sub_models(self):
        return self.sub_models