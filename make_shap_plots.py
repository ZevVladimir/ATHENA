import os
from utils.ML_support import *
import json
import configparser
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
import shap
import matplotlib.cm as cm
from  shap.plots import colors
from shap.plots._utils import convert_color
from utils.data_and_loading_functions import create_directory

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
            test_comb_name = get_combined_name(curr_test_sims) 
            for dset_name in eval_datasets:
                plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
                create_directory(plot_loc)
                
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)
                X_df = data[feature_columns]
                y_df = data[target_column]
                
        preds = make_preds(client, bst, X_df, y_df, report_name="Report", print_report=False)
        preds = preds.values

        #no_secondary_df = X_df[X_df["c_Scaled_radii"].isna()]
        #no_secondary_df = no_secondary_df.sample(frac=0.01,random_state=rand_seed)
        orb_inloc = (X_df["p_Scaled_radii"] < 0.2) & (X_df["p_Radial_vel"] < 0.6) & (X_df["p_Radial_vel"] > -0.6) 
        orb_outloc = (X_df["p_Scaled_radii"] > 0.5) & (X_df["p_Scaled_radii"] < 1) & (X_df["p_Radial_vel"] < 0.6) & (X_df["p_Radial_vel"] > 0) 
        orb_badloc = (X_df["p_Scaled_radii"] > 0.3) & (X_df["p_Scaled_radii"] < 0.5) & (X_df["p_Radial_vel"] < 0.6) & (X_df["p_Radial_vel"] > -0.6)
        
        good_orb_in_Xdf = X_df[orb_inloc]
        good_orb_out_Xdf = X_df[orb_outloc]
        bad_orb_Xdf = X_df[orb_badloc]
        good_orb_in_ydf = y_df[orb_inloc]
        good_orb_out_ydf = y_df[orb_outloc]
        bad_orb_ydf = y_df[orb_badloc]
        good_orb_in_preds = preds[orb_inloc]
        good_orb_out_preds = preds[orb_outloc]
        bad_orb_preds = preds[orb_badloc]

        new_columns = ["Current $r/R_{\mathrm{200m}}$","Current $v_{\mathrm{r}}/V_{\mathrm{200m}}$","Current $v_{\mathrm{t}}/V_{\mathrm{200m}}$","Past $r/R_{\mathrm{200m}}$","Past $v_{\mathrm{r}}/V_{\mathrm{200m}}$","Past $v_{\mathrm{t}}/V_{\mathrm{200m}}$"]
        col2num = {col: i for i, col in enumerate(new_columns)}
        order = list(map(col2num.get, new_columns))


        bst.set_param({"device": "cuda:0"})
        explainer = shap.TreeExplainer(bst)

        #no_secondary = no_secondary_df.compute()
        good_orb_in_X = good_orb_in_Xdf.compute()
        good_orb_out_X = good_orb_out_Xdf.compute()
        bad_orb_X = bad_orb_Xdf.compute()
        good_orb_in_y = good_orb_in_ydf.compute()
        good_orb_out_y = good_orb_out_ydf.compute()
        bad_orb_y = bad_orb_ydf.compute()

        good_orb_in_X = good_orb_in_X[(good_orb_in_y["Orbit_infall"] == 1) & (good_orb_in_preds == 0)]
        good_orb_out_X = good_orb_out_X[(good_orb_out_y["Orbit_infall"] == 1) & (good_orb_out_preds == 0)]
        bad_orb_X = bad_orb_X[(bad_orb_y["Orbit_infall"] == 1) & (bad_orb_preds == 0)]
        
        # good_orb_in = good_orb_in.sample(frac=0.1,random_state=rand_seed)
        # good_orb_out = good_orb_out.sample(frac=0.1,random_state=rand_seed)
        # bad_orb = bad_orb.sample(frac=0.1,random_state=rand_seed)

        #print("no secondary:",no_secondary.shape)
        print("good in:",good_orb_in_X.shape)
        print("good out:",good_orb_out_X.shape)
        print("bad:", bad_orb_X.shape)
        #no_secondary.columns = new_columns
        good_orb_in_X.columns = new_columns
        good_orb_out_X.columns = new_columns
        bad_orb_X.columns = new_columns

        #no_secondary = no_secondary.reset_index(drop=True)
        good_orb_in_X = good_orb_in_X.reset_index(drop=True)
        good_orb_out_X = good_orb_out_X.reset_index(drop=True)
        bad_orb_X = bad_orb_X.reset_index(drop=True)

        #no_secondary_shap_values = explainer(no_secondary)
        good_orb_in_shap_values = explainer(good_orb_in_X)
        good_orb_out_shap_values = explainer(good_orb_out_X)
        bad_orb_shap_values = explainer(bad_orb_X)

    with timed("SHAP Summary Chart"):
        #ax_no_secondary = shap.plots.beeswarm(no_secondary_shap_values,plot_size=(15,10),show=False,order=order)
        #plt.title("No Secondary Snapshot Population")
        #plt.xlim(-6,10)
        #plt.savefig(gen_plot_save_loc + "no_secondary_beeswarm.png")
        #print("finished no secondary")

        widths = [4,.15]
        heights = [4,4,4]
        fig = plt.figure(constrained_layout=True,figsize=(45,10))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

        ax_good_in = fig.add_subplot(gs[0,0])
        ax_good_out = fig.add_subplot(gs[1,0])
        ax_bad = fig.add_subplot(gs[2,0])

        plt.sca(ax_good_in)
        beeswarm_good_in = shap.plots.beeswarm(good_orb_in_shap_values,plot_size=(15,10),show=False,color_bar=False,order=order)
        ax_good_in.set_title("Well Classified Inner Population")
        ax_good_in.set_xlim(-6,10)
        ax_good_in.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_good_in.set_xlabel('')
       #plt.savefig(gen_plot_save_loc + "good_orb_in_beeswarm.png")
        print("finished good orb in")
        #plt.clf()

        plt.sca(ax_good_out)
        beeswarm_good_out = shap.plots.beeswarm(good_orb_out_shap_values,plot_size=(15,10),show=False,color_bar=False,order=order)
        ax_good_out.set_title("Well Classified Outer Population")
        ax_good_out.set_xlim(-6,10)
        ax_good_out.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_good_out.set_xlabel('')
        #plt.savefig(gen_plot_save_loc + "good_orb_out_beeswarm.png")
        print("finished good orb out")
        #plt.clf()

        plt.sca(ax_bad)
        beeswarm_good_out = shap.plots.beeswarm(good_orb_out_shap_values,plot_size=(15,10),show=False,color_bar=False,order=order)
        beeswarm_bad = shap.plots.beeswarm(bad_orb_shap_values,plot_size=(15,10),show=False,color_bar=False,order=order)
        ax_bad.set_title("Adaquately Classified Between Population")
        ax_bad.set_xlim(-6,10)
        #plt.savefig(gen_plot_save_loc + "bad_orb_beeswarm.png")

        labels = {
        'MAIN_EFFECT': "SHAP main effect value for\n%s",
        'INTERACTION_VALUE': "SHAP interaction value",
        'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
        'VALUE': "SHAP value (impact on model output)",
        'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
        'VALUE_FOR': "SHAP value for\n%s",
        'PLOT_FOR': "SHAP plot for %s",
        'FEATURE': "Feature %s",
        'FEATURE_VALUE': "Feature value",
        'FEATURE_VALUE_LOW': "Low",
        'FEATURE_VALUE_HIGH': "High",
        'JOINT_VALUE': "Joint SHAP value",
        'MODEL_OUTPUT': "Model output value"
        }
        
        color = colors.red_blue
        color = convert_color(color)

        color_bar_label=labels["FEATURE_VALUE"]
        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = plt.colorbar(m, cax=plt.subplot(gs[:,-1]), ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        
        fig.savefig(plot_loc + "all_vr_r_orb_beeswarm.png")