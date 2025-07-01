import os
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import argparse

from src.utils.util_fxns import create_directory, timed, load_config, load_pickle, load_ML_dsets, load_all_sim_cosmols, load_all_tdyn_steps
from src.utils.ML_fxns import setup_client, get_combined_name, make_preds, shap_with_filter, get_model_name, extract_snaps, get_feature_labels

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default=os.getcwd() + "/config.ini", 
    help='Path to config file (default: config.ini)'
)

args = parser.parse_args()
config_params = load_config(args.config)

path_to_models = config_params["PATHS"]["path_to_models"]
ML_dset_path = config_params["PATHS"]["ml_dset_path"]

features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]
model_sims = config_params["TRAIN_MODEL"]["model_sims"]
model_type = config_params["TRAIN_MODEL"]["model_type"]

test_sims = config_params["EVAL_MODEL"]["test_sims"]
eval_test_dsets = config_params["EVAL_MODEL"]["eval_test_dsets"]

if __name__ == '__main__':
    client = setup_client()
    
    with timed("Setup"): 
        comb_model_sims = get_combined_name(model_sims) 
        
        model_name = get_model_name(model_type, model_sims)    
        model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"
        model_save_loc = model_fldr_loc + model_name + ".json"

        # Try loading the model if it can't be thats an error!
        try:
            bst = xgb.Booster()
            bst.load_model(model_save_loc)
            bst.set_param({"device": "cuda:0"})
            print("Loaded Model Trained on:",model_sims)
        except:
            print("Couldn't load Booster Located at: " + model_save_loc)
        
        for curr_test_sims in test_sims:
            all_sim_cosmol_list = load_all_sim_cosmols(curr_test_sims)
            all_tdyn_steps_list = load_all_tdyn_steps(curr_test_sims)

            feature_columns = get_feature_labels(features,all_tdyn_steps_list[0])
            test_comb_name = get_combined_name(curr_test_sims) 
            
            for dset_name in eval_test_dsets:
                plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
                create_directory(plot_loc)
                all_snaps = extract_snaps(curr_test_sims[0])
                data,scale_pos_weight = load_ML_dsets(client,curr_test_sims,dset_name,all_sim_cosmol_list,all_snaps[0])
                
                X_df = data[feature_columns]
                y_df = data[target_column]
                
                preds = make_preds(client, bst, X_df, ret_dask = True)
                X_df = X_df.to_backend('pandas')
                y_df = y_df.to_backend('pandas')

                new_columns = ["Current $r/R_{\mathrm{200m}}$","Current $v_{\mathrm{r}}/V_{\mathrm{200m}}$","Current $v_{\mathrm{t}}/V_{\mathrm{200m}}$","Past $r/R_{\mathrm{200m}}$","Past $v_{\mathrm{r}}/V_{\mathrm{200m}}$","Past $v_{\mathrm{t}}/V_{\mathrm{200m}}$"]
                col2num = {col: i for i, col in enumerate(new_columns)}
                order = list(map(col2num.get, new_columns))
                # order.reverse()
                
                bst.set_param({"device": "cuda:0"})
                explainer = shap.TreeExplainer(bst)
                
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1]
                print(f"Explainer expected value: {expected_value}")
        
                no_second_dict = {
                    'X_filter': {
                        str(all_tdyn_steps_list[0][0])+"_Scaled_radii": ('==',"nan"),
                    },
                    'label_filter': {
                    }
                }
            
                all_ptl_explnr, all_ptl_shap, all_ptl_X = shap_with_filter(explainer,X_df,y_df,preds,col_names=new_columns,max_size=5000)
                no_sec_explnr, no_sec_shap, no_sec_X = shap_with_filter(explainer,X_df,y_df,preds,fltr_dic=no_second_dict,col_names=new_columns,max_size=5000)
                
                with timed("Make SHAP plots"):
                    widths = [4,4]
                    heights = [4]
                    
                    fig = plt.figure(constrained_layout=True)
                    gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

                    ax1 = fig.add_subplot(gs[0])
                    ax2 = fig.add_subplot(gs[1])
                    plt.sca(ax1)
                    ax_all_ptl = shap.plots.beeswarm(all_ptl_explnr,plot_size=(20,10),show=False,order=order,color_bar=False)
                    ax1.set_title("All Particles",fontsize=26)

                    plt.sca(ax2)
                    ax_no_sec = shap.plots.beeswarm(no_sec_explnr,plot_size=(20,10),show=False,order=order,hide_features=True)
                    ax2.set_title("Particles with no Past Snapshot",fontsize=26)
                    
                    # Set the xlim to be the max/min of both axes
                    xlims_ax1 = ax1.get_xlim()
                    xlims_ax2 = ax2.get_xlim()

                    combined_xlim = (min(xlims_ax1[0], xlims_ax2[0]), max(xlims_ax1[1], xlims_ax2[1]))
                    
                    ax1.set_xlim(combined_xlim)
                    ax2.set_xlim(combined_xlim)

                    fig.savefig(plot_loc + "comb_shap.pdf")

    