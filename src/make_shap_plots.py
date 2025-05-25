import os
import matplotlib.pyplot as plt
import xgboost as xgb

import shap

from utils.data_and_loading_functions import create_directory, timed, load_config
from utils.ML_support import setup_client, get_combined_name, load_data, make_preds, shap_with_filter, get_model_name, extract_snaps

config_params = load_config(os.getcwd() + "/config.ini")

path_to_models = config_params["PATHS"]["path_to_models"]

feature_columns = config_params["TRAIN_MODEL"]["feature_columns"]
target_column = config_params["TRAIN_MODEL"]["target_column"]
model_sims = config_params["TRAIN_MODEL"]["model_sims"]
model_type = config_params["TRAIN_MODEL"]["model_type"]

test_sims = config_params["EVAL_MODEL"]["test_sims"]
eval_datasets = config_params["EVAL_MODEL"]["eval_datasets"]



if __name__ == '__main__':
    client = setup_client()
    
    with timed("Setup"): 
        comb_model_sims = get_combined_name(model_sims) 
        
        model_name = get_model_name(model_type, model_sims)    
        model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"
        model_save_loc = model_fldr_loc + model_name + ".json"
        gen_plot_save_loc = model_fldr_loc + "plots/"

        # Try loading the model if it can't be thats an error!
        try:
            bst = xgb.Booster()
            bst.load_model(model_save_loc)
            bst.set_param({"device": "cuda:0"})
            print("Loaded Model Trained on:",model_sims)
        except:
            print("Couldn't load Booster Located at: " + model_save_loc)

        for curr_test_sims in test_sims:
            test_comb_name = get_combined_name(curr_test_sims) 
            for dset_name in eval_datasets:
                plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
                create_directory(plot_loc)
                all_snaps = extract_snaps(model_sims[0])
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,all_snaps[0],limit_files=False)
                X_df = data[feature_columns]
                y_df = data[target_column]
                
        preds = make_preds(client, bst, X_df, dask = True)
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
                "c_Scaled_radii": ('==',"nan"),
            },
            'label_filter': {
            }
        }
        
        orb_in_dict = {
            'X_filter': {
                "p_Scaled_radii": ('<',0.2),
                "p_Radial_vel": ('<',0.6),
                "p_Radial_vel": ('>',-0.6)
            },
            'label_filter': {
                'act':1,
                'pred':0
            }
        }
        
        orb_out_dict = {
            'X_filter': {
                "p_Scaled_radii": ('>',0.5),
                "p_Scaled_radii": ('<',1),
                "p_Radial_vel": ('<',0.6),
                "p_Radial_vel": ('>',0)
            },
            'label_filter': {
                'act':1,
                'pred':0
            }
        }
        
        orb_bad_misclass_dict = {
            'X_filter': {
                "p_Scaled_radii": ('>',0.3),
                "p_Scaled_radii": ('<',0.5),
                "p_Radial_vel": ('<',0.6),
                "p_Radial_vel": ('>',-0.6)
            },
            'label_filter': {
                'act':1,
                'pred':0
            }
        }        
        
        orb_bad_corr_dict = {
            'X_filter': {
                "p_Scaled_radii": ('>',0.3),
                "p_Scaled_radii": ('<',0.5),
                "p_Radial_vel": ('<',0.6),
                "p_Radial_vel": ('>',-0.6)
            },
            'label_filter': {
                'act':1,
                'pred':1
            }
        }       
        
        in_btwn_dict = {
            'X_filter': {
                "p_Scaled_radii": ('>',0.3),
                "p_Scaled_radii": ('<',0.5),
                "p_Radial_vel": ('<',0.6),
                "p_Radial_vel": ('>',-0.6)
            },
        }
        
        test_dict = {
            'label_filter':{
                'pred':1,
                'act':1,
                }
        }
        
        # all_explainer = explainer(X_df.compute())
        # all_shap_values = explainer.shap_values(X_df.compute())
        
        # no_second_X, no_second_fltr = filter_ddf(X_df,y_df,preds,fltr_dic=no_second_dict,col_names=new_columns,max_size=10000)
        # no_second_shap = all_explainer[no_second_fltr]
        # no_second_shap_values = all_shap_values[no_second_fltr]
        # good_orb_in_shap,good_orb_in_shap_values = shap_with_filter(explainer,orb_in_dict,X_df,y_df,preds,new_columns)
        # good_orb_out_shap,good_orb_out_shap_values = shap_with_filter(explainer,orb_out_dict,X_df,y_df,preds,new_columns)
        # test_shap, test_vals, test_X = shap_with_filter(explainer,test_dict,X_df,y_df,preds,new_columns)
        
        # bad_orb_missclass_X,bad_orb_missclass_fltr = filter_ddf(X_df,y_df,preds,fltr_dic = orb_bad_misclass_dict,col_names=new_columns)
        # bad_orb_missclass_shap = all_explainer[bad_orb_missclass_fltr]
        # bad_orb_missclass_shap_values = all_shap_values[bad_orb_missclass_fltr]
        
        # bad_orb_corr_X,bad_orb_corr_fltr = filter_ddf(X_df,y_df,preds,fltr_dic=orb_bad_corr_dict,col_names=new_columns)
        # bad_orb_corr_shap = all_explainer[bad_orb_corr_fltr]
        # bad_orb_corr_shap_values = all_shap_values[bad_orb_corr_fltr]
        # all_shap,all_shap_values, all_X = shap_with_filter(explainer,X=X_df,y=y_df,preds=preds,col_names=new_columns)
        # in_btwn_shap,in_btwn_shap_values = shap_with_filter(explainer,in_btwn_dict,X_df,y_df,preds,new_columns,sample=0.0001)

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
        
        fig.savefig(plot_loc + "comb_shap.png")
        print("finished combined shap plot")
         

        # widths = [4,4]
        # heights = [4,4]
        # fig_width = 75
        # fig_height = 50
        # fig = plt.figure(constrained_layout=True,figsize=(fig_width,fig_height))
        # gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

        # ax1 = fig.add_subplot(gs[0,0])
        # ax2 = fig.add_subplot(gs[0,1])
        # ax3 = fig.add_subplot(gs[1,0])
        # ax4 = fig.add_subplot(gs[1,1])

        # plt.sca(ax1)
        # r=shap.decision_plot(expected_value, bad_orb_corr_shap_values,features=bad_orb_corr_X,feature_names=new_columns,plot_color="viridis",color_bar=False,auto_size_plot=False,show=False,feature_order=None,return_objects=True, hide_bot_label=True)
        # plt.sca(ax2)
        # shap.plots.beeswarm(bad_orb_corr_shap,plot_size=(fig_width/2,fig_height/2),show=False,order=order,hide_features=True, hide_xaxis=True)
        # plt.sca(ax3)
        # shap.decision_plot(expected_value, bad_orb_missclass_shap_values,features=bad_orb_missclass_X,feature_names=new_columns,plot_color="magma",color_bar=False,auto_size_plot=False,show=False,feature_order=None,xlim=r.xlim)
        # plt.sca(ax4)
        # shap.plots.beeswarm(bad_orb_missclass_shap,plot_size=(fig_width/2,fig_height/2),show=False,order=order,hide_features=True)
        
        # ax1.text(-15,0.25,"Orbiting Particles\nCorrectly Labeled",bbox={"facecolor":'white',"alpha":0.9,},fontsize=20)
        # ax3.text(-15,0.25,"Orbiting Particles\nIncorrectly Labeled",bbox={"facecolor":'white',"alpha":0.9,},fontsize=20)
        
        # # Set it so the xlims for both beeswarms are the same
        # xlim2 = ax2.get_xlim()
        # xlim4 = ax4.get_xlim()
        # new_xlim = (min(xlim2[0], xlim4[0]), max(xlim2[1], xlim4[1]))
        # ax2.set_xlim(new_xlim)
        # ax4.set_xlim(new_xlim)
        
        # fig.savefig(plot_loc + "vr_r_orb_middle_decision.png")

    