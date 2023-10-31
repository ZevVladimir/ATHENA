
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

class Iterator(xgboost.DataIter):
  def __init__(self, svm_file_paths: List[str]):
    self._file_paths = svm_file_paths
    self._it = 0
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
    X, y = load_svmlight_file(self._file_paths[self._it])
    input_data(data=X, label=y)
    self._it += 1
    # Return 1 to let XGBoost know we haven't seen all the files yet.
    return 1

  def reset(self):
    """Reset the iterator to its beginning"""
    self._it = 0

it = Iterator(["file_0.svm", "file_1.svm", "file_2.svm"])
Xy = xgboost.DMatrix(it)

# The ``approx`` also work, but with low performance. GPU implementation is different from CPU.
# as noted in following sections.
booster = xgboost.train({"tree_method": "hist"}, Xy)

class model_creator:
    def __init__(self, dataset, keys, snapshot_list, num_params_per_snap, save_location, scaled_radii_loc, rad_vel_loc, tang_vel_loc, radii_splits, curr_sparta_file):
        self.dataset = dataset
        self.keys = keys
        self.snapshot_list = snapshot_list
        self.num_params = num_params_per_snap
        self.save_location = save_location
        self.scaled_radii_loc = scaled_radii_loc
        self.rad_vel_loc = rad_vel_loc
        self.tang_vel_loc = tang_vel_loc
        self.radii_splits = radii_splits
        self.curr_sparta_file = curr_sparta_file
        self.dataset_df = pd.DataFrame(dataset[:,2:], columns = keys[2:])
        self.sub_models = []
        self.train_val_split()

    def train_val_split(self):
        # Split the dataset inputted into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(self.dataset[:,2:], self.dataset[:,1], test_size=0.20, random_state=0)
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.mix_data() # mix the data to eliminate any connections 
    
    def mix_data(self):
        # generate random indices and then mix the training and validation sets
        train_rand_idxs = np.random.permutation(self.X_train.shape[0])
        val_rand_idxs = np.random.permutation(self.X_val.shape[0])
        self.X_train = self.X_train[train_rand_idxs]
        self.y_train = self.y_train[train_rand_idxs]
        self.X_val = self.X_val[val_rand_idxs]
        self.y_val = self.y_val[val_rand_idxs]

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
        X_within = x_dataset[np.where((x_dataset[:,self.scaled_radii_loc] > low_cutoff) & (x_dataset[:,self.scaled_radii_loc] < high_cutoff))[0]]
        y_within = y_dataset[np.where((x_dataset[:,self.scaled_radii_loc] > low_cutoff) & (x_dataset[:,self.scaled_radii_loc] < high_cutoff))[0]]

        return X_within, y_within
    
    def load_models(self):
        for filename in os.listdir(self.save_location + "sub_models/"):
            if int(re.search(r'\d+', filename).group()) == self.num_params:
                with open(self.save_location + "/sub_models/" + filename, "rb") as pickle_file:
                    model = pickle.load(pickle_file)
                    self.sub_models.append(model)
                
    def train_model(self):
        model_location = self.save_location + "/sub_models/"
        create_directory(model_location)

        sub_models = []

        # Create a sub model for each radii split determined
        for i in range(len(self.radii_splits) + 1):
            if i == 0:
                low_cutoff = 0
            else:
                low_cutoff = self.radii_splits[i - 1]
            if i == len(self.radii_splits):
                high_cutoff = np.max(self.X_train[:,self.scaled_radii_loc])
            else:
                high_cutoff = self.radii_splits[i]
            
            X, y = self.split_by_dist(low_cutoff, high_cutoff, self.X_train, self.y_train)
        
            curr_model_location = model_location + str(self.num_params) + "_params_" + "_range_" + str(low_cutoff) + "_" + str(np.round(high_cutoff,2)) + "_" + self.curr_sparta_file + ".pickle"

            if os.path.exists(curr_model_location):
                with open(curr_model_location, "rb") as pickle_file:
                    model = pickle.load(pickle_file)
            else:
                model = None

                t3 = time.time()
                
                # Train and fit each model with gpu
                model = XGBClassifier(tree_method='gpu_hist', eta = 0.01, n_estimators = 100)
                model = model.fit(X, y)

                t4 = time.time()
                print("Fitted model", t4 - t3, "seconds")

                pickle.dump(model, open(curr_model_location, "wb"), pickle.HIGHEST_PROTOCOL)
            
            sub_models.append(model)
        
        self.sub_models = sub_models

    def predict_all_models(self, dataset):
        #TODO set dataset=None
        # If there is no dataset submitted then just use the validation sets otherwise predict on the inputted one
        if np.all(dataset):
            use_dataset = self.X_val
            use_labels = self.y_val
        else:
            use_dataset = dataset[:,2:]
            use_labels = dataset[:,1]

        # for each submodel get the predictions
        all_predicts = np.zeros((use_dataset.shape[0],(len(self.sub_models)*2)))
        for i,model in enumerate(self.sub_models):
            all_predicts[:,2*i:(2*i+2)] = model.predict_proba(use_dataset)

        # Then determine which class each particle belongs to based of each model's prediction

        self.det_class_argmax(all_predicts)
        print(classification_report(use_labels, self.predicts))
    
    def predict_by_model(self, dataset):
        if np.all(dataset):
            use_dataset = self.X_val
            use_labels = self.y_val
        else:
            use_dataset = dataset[:,2:]
            use_labels = dataset[:,1]
        all_preds = np.ones(use_dataset.shape[0]) * -1
        
        start = 0
        for i in range(len(self.radii_splits) + 1):
            if i == 0:
                low_cutoff = 0
            else:
                low_cutoff = self.radii_splits[i - 1]
            if i == len(self.radii_splits):
                high_cutoff = np.max(self.X_train[:,self.scaled_radii_loc])
            else:
                high_cutoff = self.radii_splits[i]
            
            X, y = self.split_by_dist(low_cutoff, high_cutoff, use_dataset, use_labels)
            
            model_location = self.save_location + "/sub_models/"
            curr_model_location = model_location + str(self.num_params) + "_params_" + "_range_" + str(low_cutoff) + "_" + str(np.round(high_cutoff,2)) + "_" + self.curr_sparta_file + ".pickle"
            with open(curr_model_location, "rb") as pickle_file:
                curr_model = pickle.load(pickle_file)
            predicts = curr_model.predict_proba(X)
            
            all_preds[start:start+X.shape[0]] = self.det_class_by_model(predicts)
            start = start + X.shape[0]
        
        self.predicts = all_preds
        print(classification_report(use_labels, self.predicts))
    
    def det_class_argmax(self, predicts):
        # Determine which class a particle is based off which model gave the highest probability
        #TODO add option to use an average of predictions instead
        pred_loc = np.argmax(predicts, axis = 1)

        final_predicts = np.zeros(predicts.shape[0])
        final_predicts[np.where((pred_loc % 2) != 0)] = 1

        self.predicts = final_predicts
        
    def det_class_by_model(self, predicts):
        pred = np.argmax(predicts, axis = 1)
        print(pred)
        return pred
        
        

    def graph(self, corr_matrix = False, feat_imp = False):
        # implement functionality to create graphs of data if wanted
        if corr_matrix:
            graph_correlation_matrix(self.dataset_df, self.save_location, title = "model_num_params_" + str(len(self.keys) - 2),show = False, save = True)
        if feat_imp:
            for i, model in enumerate(self.sub_models):
                if i == 0:
                    low_cutoff = 0
                else:
                    low_cutoff = self.radii_splits[i - 1]
                if i == len(self.radii_splits):
                    high_cutoff = np.max(self.X_train[:,self.scaled_radii_loc])
                else:
                    high_cutoff = self.radii_splits[i]
                graph_feature_importance(np.array(self.keys[2:]), model.feature_importances_, "num_params_" + str(len(self.keys) - 2) +  "_radii: " + str(low_cutoff) + "_" + str(np.round(high_cutoff,2)), False, True, self.save_location)
        
    def get_predicts(self):
        return self.predicts
    
    def get_sub_models(self):
        return self.sub_models