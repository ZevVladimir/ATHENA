import os
from utils.ML_support import *
import json
import configparser
import matplotlib.pyplot as plt
import matplotlib as mpl
from colossus.cosmology import cosmology
import shap
import matplotlib.cm as cm
from  shap.plots import colors
from shap.plots._utils import convert_color
from utils.data_and_loading_functions import create_directory

config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
sim_cosmol = config["MISC"]["sim_cosmol"]

if __name__ == '__main__':
    if use_gpu:
        mp.set_start_method("spawn")

    if not use_gpu and on_zaratan:
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            print("SLURM_CPUS_PER_TASK is not defined.")
        if use_gpu:
            initialize(local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        else:
            initialize(nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        print("Initialized")
        client = Client()
        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        login_node_address = "zvladimi@login.zaratan.umd.edu" # Change this to the address/domain of your login node

        logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
    else:
        client = get_CUDA_cluster()
    
    with timed("Setup"):

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
                
        preds = make_preds(client, bst, X_df, dask = True, report_name="Report", print_report=False)
        X_df = X_df.to_backend('pandas')
        y_df = y_df.to_backend('pandas')

        new_columns = ["Current $r/R_{\mathrm{200m}}$","Current $v_{\mathrm{r}}/V_{\mathrm{200m}}$","Current $v_{\mathrm{t}}/V_{\mathrm{200m}}$","Past $r/R_{\mathrm{200m}}$","Past $v_{\mathrm{r}}/V_{\mathrm{200m}}$","Past $v_{\mathrm{t}}/V_{\mathrm{200m}}$"]
        col2num = {col: i for i, col in enumerate(new_columns)}
        order = list(map(col2num.get, new_columns))
        order.reverse()
        
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

        no_second_shap,no_second_shap_values, no_second_X = shap_with_filter(explainer,X_df,y_df,preds,fltr_dic=no_second_dict,col_names=new_columns,max_size=10000)
        # good_orb_in_shap,good_orb_in_shap_values = shap_with_filter(explainer,orb_in_dict,X_df,y_df,preds,new_columns)
        # good_orb_out_shap,good_orb_out_shap_values = shap_with_filter(explainer,orb_out_dict,X_df,y_df,preds,new_columns)
        # test_shap, test_vals, test_X = shap_with_filter(explainer,test_dict,X_df,y_df,preds,new_columns)
        bad_orb_missclass_shap,bad_orb_missclass_shap_values, bad_orb_missclass_X = shap_with_filter(explainer,X_df,y_df,preds,fltr_dic = orb_bad_misclass_dict,col_names=new_columns)
        bad_orb_corr_shap,bad_orb_corr_shap_values, bad_orb_corr_X = shap_with_filter(explainer,X_df,y_df,preds,fltr_dic=orb_bad_corr_dict,col_names=new_columns)
        # all_shap,all_shap_values, all_X = shap_with_filter(explainer,X=X_df,y=y_df,preds=preds,col_names=new_columns)
        # in_btwn_shap,in_btwn_shap_values = shap_with_filter(explainer,in_btwn_dict,X_df,y_df,preds,new_columns,sample=0.0001)

    with timed("Make SHAP plots"):
        ax_no_secondary = shap.plots.beeswarm(no_second_shap,plot_size=(15,10),show=False,order=order)
        plt.title("No Secondary Snapshot Population")
        plt.xlim(-6,10)
        plt.savefig(plot_loc + "no_secondary_beeswarm.png")
        print("finished no secondary")

        widths = [4,4]
        heights = [4,4]
        fig_width = 75
        fig_height = 50
        fig = plt.figure(constrained_layout=True,figsize=(fig_width,fig_height))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])

        plt.sca(ax1)
        r=shap.decision_plot(expected_value, bad_orb_corr_shap_values,features=bad_orb_corr_X,feature_names=new_columns,plot_color="viridis",color_bar=False,auto_size_plot=False,show=False,feature_order=None,return_objects=True, hide_bot_label=True)
        plt.sca(ax2)
        shap.plots.beeswarm(bad_orb_corr_shap,plot_size=(fig_width/2,fig_height/2),show=False,order=order,hide_features=True, hide_xaxis=True)
        plt.sca(ax3)
        shap.decision_plot(expected_value, bad_orb_missclass_shap_values,features=bad_orb_missclass_X,feature_names=new_columns,plot_color="magma",color_bar=False,auto_size_plot=False,show=False,feature_order=None,xlim=r.xlim)
        plt.sca(ax4)
        shap.plots.beeswarm(bad_orb_missclass_shap,plot_size=(fig_width/2,fig_height/2),show=False,order=order,hide_features=True)
        
        # Set it so the xlims for both beeswarms are the same
        xlim2 = ax2.get_xlim()
        xlim4 = ax4.get_xlim()
        new_xlim = (min(xlim2[0], xlim4[0]), max(xlim2[1], xlim4[1]))
        ax2.set_xlim(new_xlim)
        ax4.set_xlim(new_xlim)
        
        # labels = {
        # 'MAIN_EFFECT': "SHAP main effect value for\n%s",
        # 'INTERACTION_VALUE': "SHAP interaction value",
        # 'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
        # 'VALUE': "SHAP value (impact on model output)",
        # 'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
        # 'VALUE_FOR': "SHAP value for\n%s",
        # 'PLOT_FOR': "SHAP plot for %s",
        # 'FEATURE': "Feature %s",
        # 'FEATURE_VALUE': "Feature value",
        # 'FEATURE_VALUE_LOW': "Low",
        # 'FEATURE_VALUE_HIGH': "High",
        # 'JOINT_VALUE': "Joint SHAP value",
        # 'MODEL_OUTPUT': "Model output value"
        # }
        
        # color = colors.red_blue
        # color = convert_color(color)

        # color_bar_label=labels["FEATURE_VALUE"]
        # m = cm.ScalarMappable(cmap=color)
        # m.set_array([0, 1])
        # cb = plt.colorbar(m, cax=plt.subplot(gs[:,-1]), ticks=[0, 1], aspect=80)
        # cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        # cb.set_label(color_bar_label, size=12, labelpad=0)
        # cb.ax.tick_params(labelsize=11, length=0)
        # cb.set_alpha(1)
        # cb.outline.set_visible(False)
        
        fig.savefig(plot_loc + "vr_r_orb_middle_decision.png")

    