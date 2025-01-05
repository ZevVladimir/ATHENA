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
        scale_rad=False
        use_weights=False
        if reduce_rad > 0 and reduce_perc > 0:
            scale_rad = True
        if weight_rad > 0 and min_weight > 0:
            use_weights=True    

        model_dir = model_type + "_" + model_comb_name + "nu" + nu_string 

        if scale_rad:
            model_dir += "scl_rad" + str(reduce_rad) + "_" + str(reduce_perc)
        if use_weights:
            model_dir += "wght" + str(weight_rad) + "_" + str(min_weight)
            
        # model_name =  model_dir + model_comb_name

        model_save_loc = path_to_xgboost + model_comb_name + "/" + model_dir + "/"

        gen_plot_save_loc = model_save_loc + "plots/"

        try:
            bst = xgb.Booster()
            bst.load_model(model_save_loc + model_dir + ".json")
            print("Loaded Model Trained on:",model_sims)
        except:
            print("Couldn't load Booster Located at: " + model_save_loc + model_dir + ".json")

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
        ax1.set_title("Sample: All Particles",fontsize=26)

        plt.sca(ax2)
        ax_no_sec = shap.plots.beeswarm(no_sec_explnr,plot_size=(20,10),show=False,order=order,hide_features=True)
        ax2.set_title("Sample: Particles with no Past Snapshot",fontsize=26)
        
        # Set the xlim to be the max/min of both axes
        xlims_ax1 = ax1.get_xlim()
        xlims_ax2 = ax2.get_xlim()

        combined_xlim = (min(xlims_ax1[0], xlims_ax2[0]), max(xlims_ax1[1], xlims_ax2[1]))
        
        ax1.set_xlim(combined_xlim)
        ax2.set_xlim(combined_xlim)
        
        fig.savefig(plot_loc + "comb_shap.png")
        print("finished combiend shap plot")
         

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

    