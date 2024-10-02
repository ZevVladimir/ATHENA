import os
from utils.ML_support import *
import json
import configparser
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
import shap

config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
sim_cosmol = config["MISC"]["sim_cosmol"]

if __name__ == '__main__':
    with timed("Setup"):
        client = get_CUDA_cluster()

        if sim_cosmol == "planck13-nbody":
            cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
        else:
            cosmol = cosmology.setCosmology(sim_cosmol) 

        feature_columns = ["p_Scaled_radii","p_Radial_vel","p_Tangential_vel","c_Scaled_radii","c_Radial_vel","c_Tangential_vel"]
        target_column = ["Orbit_infall"]

        model_comb_name = get_combined_name(model_sims)
        model_dir = model_type + "_" + model_comb_name + "nu" + nu_string 

        model_name =  model_dir + model_comb_name
                
        model_save_loc = path_to_xgboost + model_comb_name + "/" + model_dir + "/"
        gen_plot_save_loc = model_save_loc + "plots/"

        try:
            bst = xgb.Booster()
            bst.load_model(model_save_loc + model_name + ".json")
            print("Loaded Model Trained on:",model_sims)
        except:
            print("Couldn't load Booster Located at: " + model_save_loc + model_name + ".json")

        for curr_test_sims in test_sims:
            for dset_name in eval_datasets:
                
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)
                X_df = data[feature_columns]
                y_df = data[target_column]

        print(X_df.shape[0].compute(),X_df.shape[1])
        X = X_df.sample(frac = 0.001, random_state = 42)
        X = X.compute()
        print(X.shape)

        bst.set_param({"device": "cuda:0"})
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X)

    with timed("Summary Bar Chart"):
        fig = shap.summary_plot(shap_values, X, plot_type="bar",show=False)
        plt.savefig(gen_plot_save_loc + "shap_summ_bar.png")

    with timed("SHAP Summary Chart"):
        fig = shap.summary_plot(shap_values, X,show=False)
        plt.savefig(gen_plot_save_loc + "shap_summ.png")

    with timed("SHAP PR vs PRV"):
        fig = shap.dependence_plot("p_Scaled_radii",shap_values,X,interaction_index="p_Radial_vel",show=False)
        plt.savefig(gen_plot_save_loc + "pr_prv.png")
