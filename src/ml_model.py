import xgboost as xgb
from xgboost import dask as dxgb
import numpy as np
import os

class MLModel:
    def __init__(self, model_name="", model_params=None, ntrees=100, early_stopping_rounds=10):
        self.model_params = model_params or {
                "verbosity": 1,
                "tree_method": "hist",
                "max_depth":4,
                "subsample": 0.5,
                'objective': 'binary:logistic',
                'eval_metric': 'error',
            }
        self.model = None
        self.trained = False
        self.ntrees = ntrees
        self.model_name = model_name
        self.early_stopping_rounds = early_stopping_rounds
        
    def set_scale_pos_weight(self,scale_pos_weight):
        self.model_params["scale_pos_weight"] = scale_pos_weight

    def set_ntrees(self, ntrees):
        self.ntrees = ntrees
        
    def train(self, X_train, y_train, adtnl_evals=None):
        evals = [(X_train,y_train)]
        if adtnl_evals is not None:
            evals.extend(adtnl_evals)
            
        self.model = xgb.XGBClassifier(**self.model_params)

        evals_result = {}
        self.model.fit(
            X_train,
            y_train,
            eval_set=evals,
            early_stopping_rounds= self.early_stopping_rounds,
            eval_metric=self.model_params.get("eval_metric", "logloss"),
            verbose=False,
            callbacks=[xgb.callback.EvaluationMonitor()],
            evals_result=evals_result
        )

        self.history = evals_result
        self.trained = True
        
    def dask_train(self, client, dtrain, adtnl_evals):
        evals = [(dtrain, 'train')]
        if adtnl_evals is not None:
            evals.extend(adtnl_evals)
        output = dxgb.train(
            client,
            self.model_params,
            dtrain,
            num_boost_round=self.ntrees,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds, 
        )
        self.model = output["booster"]
        self.history = output["history"]
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Model is not trained yet.")
        
        if isinstance(self.model, xgb.XGBClassifier):
            # sklearn API: direct predict
            return self.model.predict(X)
        
        elif isinstance(self.model, xgb.Booster):
            # Booster object (from dask_train)
            # Use inplace_predict for efficiency if possible
            try:
                preds = self.model.inplace_predict(X)
            except AttributeError:
                # fallback to DMatrix + predict if inplace_predict not available
                dmatrix = xgb.DMatrix(X)
                preds = self.model.predict(dmatrix)
            return preds
        
        else:
            raise TypeError(f"Unrecognized model type: {type(self.model)}")

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)

    def get_feature_importance(self, importance_type="gain"):
        if not self.trained:
            raise RuntimeError("Model must be trained to retrieve feature importances.")
        return self.model.get_booster().get_score(importance_type=importance_type)

    def save_model(self, filepath):
        self.model.save_model(filepath + self.model_name + ".json")        
        
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
        fig.savefig(save_loc + "/tree_plot.png")