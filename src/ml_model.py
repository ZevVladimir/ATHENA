import xgboost as xgb
from xgboost import dask as dxgb
import numpy as np
import os
import matplotlib.pyplot as plt

from src.utils.util_fxns import timed

class MLModel:
    def __init__(self, model_name="", model_params=None, ntrees=100, early_stopping_rounds=10, verbosity = 1, model_dir=""):
        self.model_params = model_params or {
                "verbosity": 1,
                "tree_method": "hist",
                "max_depth":4,
                "subsample": 0.5,
                'objective': 'binary:logistic',
            }
        self.model = None
        self.trained = False
        self.ntrees = ntrees
        self.verbosity = verbosity
        self.model_name = model_name
        self.early_stopping_rounds = early_stopping_rounds
        self.model_dir = model_dir
        self.use_dask = False
        
    def set_scale_pos_weight(self,scale_pos_weight):
        self.model_params["scale_pos_weight"] = scale_pos_weight

    def set_ntrees(self, ntrees):
        self.ntrees = ntrees
    
    def comb_model_sim_str(self,model_sims):
        train_sims = ""
        for i,sim in enumerate(model_sims):
            if i != len(model_sims) - 1:
                train_sims += sim + ", "
            else:
                train_sims += sim
        return train_sims
        
    def train(self, dtrain, model_sims, evals=None, save_model=True):
        self.sims_trained_on = self.comb_model_sim_str(model_sims)
    
        with timed("Training model:"):
            if evals == None:
                self.model = xgb.train(
                    self.model_params,
                    dtrain,
                    num_boost_round=self.ntrees,
                    verbose_eval=self.verbosity,
                )
            else:
                evals_result = {}
                self.model = xgb.train(
                    self.model_params,
                    dtrain,
                    num_boost_round=self.ntrees,
                    evals=evals,
                    early_stopping_rounds= self.early_stopping_rounds,
                    evals_result=evals_result,
                    verbose_eval=self.verbosity,
                )

                self.history = evals_result
            self.trained = True
            
            if save_model:
                self.save_model(self.model_dir)
        
    def dask_train(self, client, dtrain, model_sims, evals=None):
        self.sims_trained_on = self.comb_model_sim_str(model_sims)
        self.use_dask = True
        with timed("Training model:"):
            if evals == None:
                output = dxgb.train(
                    client,
                    self.model_params,
                    dtrain,
                    num_boost_round=self.ntrees,
                    verbose_eval=self.verbosity,
                )
            else:
                output = dxgb.train(
                    client,
                    self.model_params,
                    dtrain,
                    num_boost_round=self.ntrees,
                    evals=evals,
                    early_stopping_rounds= self.early_stopping_rounds,
                    verbose_eval=self.verbosity,
                )

            self.model = output["booster"]
            self.history = output["history"]
            self.trained = True

    def predict(self, X, class_thrshld = 0.5, client=None):
        if not self.trained:
            raise RuntimeError("Model is not trained yet.")
        
        if self.use_dask and client is not None:
            try:
                preds = dxgb.predict(client,self.model,X)
                preds = preds.map_partitions(lambda df: (df >= class_thrshld).astype(int))
                return preds
            except AttributeError:
                preds = dxgb.inplace_predict(client, self.model, X).compute()
                preds = (preds >= class_thrshld).astype(np.int8)
    
            return preds
        elif self.use_dask is False:
            try:
                preds = self.model.inplace_predict(X)
                preds = (preds >= class_thrshld).astype(np.int8)
            except AttributeError:
                # fallback to DMatrix + predict if inplace_predict not available
                dmatrix = xgb.DMatrix(X)
                preds = self.model.predict(dmatrix)
            return preds
        
        else:
            raise TypeError(f"Unrecognized model type: {type(self.model)}")

    def full_save(self):
        self.save_model(self.model_dir)
        self.save_config(self.model_dir)

    def save_model(self, filepath):
        print(self.model_name)
        self.model.save_model(os.path.join(filepath,self.model_name + self.sims_trained_on + ".json"))
        
    def save_config(self, filepath):
        self.model.save_config(os.path.join(filepath,"config.json"))        
        
    def load_model(self, filepath):
        if os.path.isfile(filepath):
            self.model = xgb.Booster()
            self.model.load_model(filepath)
            self.trained = True
        else:
            raise FileNotFoundError(f"No model found at {filepath}")
        
        print(f"Model loaded from {filepath}")

    def plot_tree(bst,tree_num,save_loc):
        fig, ax = plt.subplots(figsize=(400, 10))
        xgb.plot_tree(bst, num_trees=tree_num, ax=ax,rankdir='LR')
        fig.savefig(save_loc + "/tree_plot.pdf")