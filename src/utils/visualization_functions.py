import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg')
import matplotlib.colors as colors
import multiprocessing as mp
import os
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_and_loading_functions import create_directory

plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

num_processes = mp.cpu_count()
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")

curr_sparta_file = config["MISC"]["curr_sparta_file"]
MLOIS_path = config["PATHS"]["MLOIS_path"]

def rv_vs_radius_plot(rad_vel, hubble_vel, start_nu, end_nu, color, ax = None):
    if ax == None:
        ax = plt.gca()
    ax.plot(rad_vel[:,0], rad_vel[:,1], color = color, alpha = 0.7, label = r"${0} < \nu < {1}$".format(str(start_nu), str(end_nu)))
    arr1inds = hubble_vel[:,0].argsort()
    hubble_vel[:,0] = hubble_vel[arr1inds,0]
    hubble_vel[:,1] = hubble_vel[arr1inds,1]
    
    ax.set_title("average radial velocity vs position all particles")
    ax.set_xlabel("position $r/R_{200m}$")
    ax.set_ylabel("average rad vel $v_r/v_{200m}$")
    ax.set_xscale("log")    
    ax.set_ylim([-.5,1])
    ax.set_xlim([0.01,15])
    ax.legend(frameon=False)
    
    return ax.plot(hubble_vel[:,0], hubble_vel[:,1], color = "purple", alpha = 0.5, linestyle = "dashed", label = r"Hubble Flow")

def graph_feature_importance(feature_names, feature_importance, title, plot, save, save_location):
    mpl.rcParams.update({'font.size': 8})
    fig2, (plot1) = plt.subplots(1,1)
    fig2.tight_layout()
    
    import_idxs = np.argsort(feature_importance)
    plot1.barh(feature_names[import_idxs], feature_importance[import_idxs])
    plot1.set_xlabel("XGBoost feature importance")
    plot1.set_title("Feature Importance for model: " + title)
    plot1.set_xlim(0,1)
    
    if plot:
        plt.show()
    if save:
        create_directory(save_location + "feature_importance_plots/")
        fig2.savefig(save_location + "feature_importance_plots/" + title + ".png", bbox_inches="tight")
    plt.close()

def graph_correlation_matrix(data, labels, save_location, show, save):
    mpl.rcParams.update({'font.size': 12})
    masked_data = np.ma.masked_invalid(data)
    corr_mtrx = np.ma.corrcoef(masked_data, rowvar=False)
    heatmap = sns.heatmap(corr_mtrx, annot = True, cbar = True, xticklabels=labels, yticklabels=labels)
    heatmap.set_title("Feature Correlation Heatmap")

    if show:
        plt.show()
    if save:
        fig = heatmap.get_figure()
        fig.set_size_inches(21, 13)
        fig.savefig(save_location + "corr_matrix.png")
    plt.close()
    
def plot_data_dist(data, labels, num_bins, save_location, show, save):
    num_feat = data.shape[1] 
    num_rows = int(np.ceil(np.sqrt(num_feat)))
    num_cols = int(np.ceil(num_feat / num_rows))
    
    fig, axes = plt.subplots(num_rows, num_cols)
    
    axes = axes.flatten()

    for i in range(num_feat, num_rows*num_cols):
        fig.delaxes(axes[i])
        
    for i in range(num_feat):
        axes[i].hist(data[:,i],bins=num_bins)
        axes[i].set_title(labels[i])
        axes[i].set_ylabel("Frequency")
        axes[i].set_yscale('log')

    if show:
        plt.show()
    if save:
        fig.set_size_inches(15, 15)
        fig.savefig(save_location + "data_hist.png")
    plt.close()
        
def feature_dist(features, labels, save_name, plot, save, save_location):
    tot_plts = features.shape[1]
    num_col = 3
    
    num_rows = tot_plts // num_col
    if tot_plts % num_col != 0:
        num_rows += 1
    
    position = np.arange(1, tot_plts + 1)
    
    fig = plt.figure(1)
    fig = plt.figure()
    
    for i in range(tot_plts):
        ax = fig.add_subplot(num_rows, num_col, position[i])
        ax.hist(features[:,i])
        ax.set_title(labels[i])
        ax.set_ylabel("counts")
    
    if plot:
        plt.show()
    if save:
        create_directory(save_location + "feature_dist_hists")
        fig.savefig(save_location + "feature_dist_hists/feature_dist_" + save_name + ".png")
    plt.close()
        
def plot_halo_ptls(pos, act_labels, save_path, pred_labels = None):
    act_inf_ptls = pos[np.where(act_labels == 0)]
    act_orb_ptls = pos[np.where(act_labels == 1)]
    pred_inf_ptls = pos[np.where(pred_labels == 0)]
    pred_orb_ptls = pos[np.where(pred_labels == 1)]
    inc_class = pos[np.where(act_labels != pred_labels)]
    corr_class = pos[np.where(act_labels == pred_labels)]
    plt.rcParams['figure.constrained_layout.use'] = True
    fig, ax = plt.subplots(2)
    ax[0].scatter(act_inf_ptls[:,0], act_inf_ptls[:,1], c='g', label = "Infalling Particles")
    ax[0].scatter(act_orb_ptls[:,0], act_orb_ptls[:,1], c='b', label = "Orbiting Particles")
    ax[0].set_title("Actual Distribution of Orbiting/Infalling Particles")
    ax[0].set_xlabel("X position (kpc)")
    ax[0].set_ylabel("Y position (kpc)")
    ax[0].legend(frameon=False)
    
    ax[1].scatter(pred_inf_ptls[:,0], pred_inf_ptls[:,1], c='g', label = "Infalling Particles")
    ax[1].scatter(pred_orb_ptls[:,0], pred_orb_ptls[:,1], c='b', label = "Orbiting Particles")
    ax[1].set_title("Predicted Distribution of Orbiting/Infalling Particles")
    ax[1].set_xlabel("X position (kpc)")
    ax[1].set_ylabel("Y position (kpc)")
    ax[1].legend(frameon=False)
    fig.savefig(save_path + "plot_of_halo_both_dist.png")

    fig, ax = plt.subplots(1)
    ax.scatter(corr_class[:,0], corr_class[:,1], c='g', label = "Correctly Labeled")
    ax.scatter(inc_class[:,0], inc_class[:,1], c='r', label = "Incorrectly Labeled")
    ax.set_title("Predicted Distribution of Orbiting/Infalling Particles")
    ax.set_xlabel("X position (kpc)")
    ax.set_ylabel("Y position (kpc)")
    ax.legend(frameon=False)
    fig.savefig(save_path + "plot_of_halo_label_dist.png")
    
def halo_plot_3d(ptl_pos, halo_pos, real_labels, preds):
    axis_cut = 2
    
    slice_test_halo_pos = np.where((ptl_pos[:,axis_cut] > 0.9 * halo_pos[axis_cut]) & (ptl_pos[:,axis_cut] < 1.1 * halo_pos[axis_cut]))[0]

    real_inf = np.where(real_labels == 0)[0]
    real_orb = np.where(real_labels == 1)[0]
    pred_inf = np.where(preds == 0)[0]
    pred_orb = np.where(preds == 1)[0]
    
    real_inf_slice = np.intersect1d(slice_test_halo_pos, real_inf)
    real_orb_slice = np.intersect1d(slice_test_halo_pos, real_orb)
    pred_inf_slice = np.intersect1d(slice_test_halo_pos, pred_inf)
    pred_orb_slice = np.intersect1d(slice_test_halo_pos, pred_orb)

    # actually orb labeled inf
    inc_orb = np.where((real_labels == 1) & (preds == 0))[0]
    # actually inf labeled orb
    inc_inf = np.where((real_labels == 0) & (preds == 1))[0]
    inc_orb_slice = np.intersect1d(slice_test_halo_pos, inc_orb)
    inc_inf_slice = np.intersect1d(slice_test_halo_pos, inc_inf)
    
    print(inc_orb.shape[0])
    print(inc_inf.shape[0])
    print(real_inf.shape[0])
    print(real_orb.shape[0])
    print(pred_inf.shape[0])
    print(pred_orb.shape[0])

    axis_fontsize=14
    title_fontsize=24
    
    fig = plt.figure(figsize=(30,10))
    ax1 = fig.add_subplot(131,projection='3d')
    ax1.scatter(ptl_pos[real_inf,0],ptl_pos[real_inf,1],ptl_pos[real_inf,2],c='orange', alpha=0.1)
    ax1.scatter(ptl_pos[real_orb,0],ptl_pos[real_orb,1],ptl_pos[real_orb,2],c='b', alpha=0.1)
    ax1.set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax1.set_ylabel("Y position (kpc/h)",fontsize=axis_fontsize)
    ax1.set_zlabel("Z position (kpc/h)",fontsize=axis_fontsize)
    ax1.set_title("Correctly Labeled Particles", fontsize=title_fontsize)
    ax1.scatter([],[],[],c="orange",label="Infalling Particles")
    ax1.scatter([],[],[],c="b",label="Orbiting Particles")
    ax1.legend(fontsize=axis_fontsize,frameon=False)

    ax2 = fig.add_subplot(132,projection='3d')
    ax2.scatter(ptl_pos[pred_inf,0],ptl_pos[pred_inf,1],ptl_pos[pred_inf,2],c='orange', alpha=0.1)
    ax2.scatter(ptl_pos[pred_orb,0],ptl_pos[pred_orb,1],ptl_pos[pred_orb,2],c='b', alpha=0.1)
    ax2.set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_ylabel("Y position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_zlabel("Z position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_title("Model Predicted Labels", fontsize=title_fontsize)
    ax2.scatter([],[],[],c="orange",label="Infalling Particles")
    ax2.scatter([],[],[],c="b",label="Orbiting Particles")
    ax2.legend(fontsize=axis_fontsize,frameon=False)

    ax3 = fig.add_subplot(133,projection='3d')
    ax3.scatter(ptl_pos[inc_inf,0],ptl_pos[inc_inf,1],ptl_pos[inc_inf,2],c='r', alpha=0.1)
    ax3.scatter(ptl_pos[inc_orb,0],ptl_pos[inc_orb,1],ptl_pos[inc_orb,2],c='k', alpha=0.1)
    ax3.set_xlim(np.min(ptl_pos[:,0]),np.max(ptl_pos[:,0]))
    ax3.set_ylim(np.min(ptl_pos[:,1]),np.max(ptl_pos[:,1]))
    ax3.set_zlim(np.min(ptl_pos[:,2]),np.max(ptl_pos[:,2]))
    ax3.set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax3.set_ylabel("Y position (kpc/h)",fontsize=axis_fontsize)
    ax3.set_zlabel("Z position (kpc/h)",fontsize=axis_fontsize)
    ax3.set_title("Model Incorrect Labels", fontsize=title_fontsize)
    ax3.scatter([],[],[],c="r",label="Pred: Orbiting \n Actual: Infalling")
    ax3.scatter([],[],[],c="k",label="Pred: Inalling \n Actual: Orbiting")
    ax3.legend(fontsize=axis_fontsize,frameon=False)

    fig.subplots_adjust(wspace=0.05)
    
    fig.savefig("/home/zvladimi/MLOIS/Random_figures/3d_one_halo_all.png")

    fig, ax = plt.subplots(1, 3,figsize=(30,10))
    
    alpha = 0.25

    ax[0].scatter(ptl_pos[real_inf_slice,0],ptl_pos[real_inf_slice,1],c='orange', alpha = alpha, label="Inalling ptls")
    ax[0].scatter(ptl_pos[real_orb_slice,0],ptl_pos[real_orb_slice,1],c='b', alpha = alpha, label="Orbiting ptls")
    ax[0].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[0].set_ylabel("Y position (kpc/h)",fontsize=axis_fontsize)
    ax[0].set_title("Particles Labeled by SPARTA",fontsize=title_fontsize)
    ax[0].legend(fontsize=axis_fontsize,frameon=False)
    
    ax[1].scatter(ptl_pos[pred_inf_slice,0],ptl_pos[pred_inf_slice,1],c='orange', alpha = alpha, label="Predicted Inalling ptls")
    ax[1].scatter(ptl_pos[pred_orb_slice,0],ptl_pos[pred_orb_slice,1],c='b', alpha = alpha, label="Predicted Orbiting ptls")
    ax[1].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[1].set_title("Particles Labeled by ML Model",fontsize=title_fontsize)
    ax[1].tick_params(axis='y', which='both',left=False,labelleft=False)
    ax[1].legend(fontsize=axis_fontsize,frameon=False)
    
    ax[2].scatter(ptl_pos[inc_orb_slice,0],ptl_pos[inc_orb_slice,1],c='r', marker='x', label="Pred: Inalling \n Actual: Orbiting")
    ax[2].scatter(ptl_pos[inc_inf_slice,0],ptl_pos[inc_inf_slice,1],c='r', marker='+', label="Pred: Orbiting \n Actual: Infalling")
    ax[2].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[2].set_title("Incorrectly Labeled Particles",fontsize=title_fontsize)
    ax[2].tick_params(axis='y', which='both',left=False,labelleft=False)
    ax[2].legend(fontsize=axis_fontsize,frameon=False)
    
    fig.savefig("/home/zvladimi/MLOIS/Random_figures/one_halo.png")
    
def compute_alpha(num_points, max_alpha=1.0, min_alpha=0.001, scaling_factor=0.5):
    return max(min_alpha, max_alpha * (scaling_factor / (num_points ** 0.5)))    
    
def halo_plot_3d_vec(ptl_pos, ptl_vel, halo_pos, halo_vel, halo_r200m, labels, constraint, halo_idx, v200m_scale):
    inf_ptls = np.where(labels == 0)[0]
    orb_ptls = np.where(labels == 1)[0]
    
    inf_ptls_cnstrn = np.intersect1d(inf_ptls, constraint)
    orb_ptls_cnstrn = np.intersect1d(orb_ptls, constraint)
    
    min_alpha = 0.001
    max_alpha = 1
    all_alpha = compute_alpha(inf_ptls.shape[0])
    cnstrn_alpha = compute_alpha(inf_ptls_cnstrn.shape[0])

    axis_fontsize=14
    title_fontsize=24
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],subplot_titles=["All Particles",'<1.1R200m and >'+str(v200m_scale)+"v200m"])

    
    fig.add_trace(go.Cone(x=ptl_pos[inf_ptls,0], y=ptl_pos[inf_ptls,1], z=ptl_pos[inf_ptls,2],
                          u=ptl_vel[inf_ptls,0], v=ptl_vel[inf_ptls,1], w=ptl_vel[inf_ptls,2],
                          colorscale=[[0, 'green'], [1, 'green']],sizemode='raw',sizeref=1,showscale=False,
                          opacity=all_alpha,name='Infalling'),row=1,col=1)
    fig.add_trace(go.Cone(x=[0], y=[0], z=[0],
                          u=[0], v=[0], w=[0],
                          colorscale=[[0, 'green'], [1, 'green']],sizemode='raw',sizeref=1,showscale=False,
                          opacity=1,name='Infalling',showlegend=True),row=1,col=1)

    fig.add_trace(go.Cone(x=ptl_pos[orb_ptls,0], y=ptl_pos[orb_ptls,1], z=ptl_pos[orb_ptls,2],
                          u=ptl_vel[orb_ptls,0], v=ptl_vel[orb_ptls,1], w=ptl_vel[orb_ptls,2],
                          colorscale=[[0, 'blue'], [1, 'blue']],sizemode='raw',sizeref=1,showscale=False,
                          opacity=all_alpha,name='Orbiting'),row=1,col=1)
    fig.add_trace(go.Cone(x=[0], y=[0], z=[0],
                          u=[0], v=[0], w=[0],
                          colorscale=[[0, 'blue'], [1, 'blue']],sizemode='raw',sizeref=1,showscale=False,
                          opacity=1,name='Orbiting',showlegend=True),row=1,col=1)
    
    fig.add_trace(go.Cone(x=[halo_pos[0]], y=[halo_pos[1]], z=[halo_pos[2]],
                          u=[halo_vel[0]], v=[halo_vel[1]], w=[halo_vel[2]],
                          colorscale=[[0, 'red'], [1, 'red']],sizemode='raw',sizeref=1,showscale=False,
                          opacity=1,name='Halo Center',showlegend=True),row=1,col=1)
    
    if inf_ptls_cnstrn.shape[0] > 0:
        fig.add_trace(go.Cone(x=ptl_pos[inf_ptls_cnstrn,0], y=ptl_pos[inf_ptls_cnstrn,1], z=ptl_pos[inf_ptls_cnstrn,2],
                          u=ptl_vel[inf_ptls_cnstrn,0], v=ptl_vel[inf_ptls_cnstrn,1], w=ptl_vel[inf_ptls_cnstrn,2],
                          colorscale=[[0, 'green'], [1, 'green']],sizemode='raw',sizeref=1,showscale=False,
                          opacity=cnstrn_alpha,name='Infalling'),row=1,col=2)
    if orb_ptls_cnstrn.shape[0] > 0:
        fig.add_trace(go.Cone(x=ptl_pos[orb_ptls_cnstrn,0], y=ptl_pos[orb_ptls_cnstrn,1], z=ptl_pos[orb_ptls_cnstrn,2],
                          u=ptl_vel[orb_ptls_cnstrn,0], v=ptl_vel[orb_ptls_cnstrn,1], w=ptl_vel[orb_ptls_cnstrn,2],
                          colorscale=[[0, 'blue'], [1, 'blue']],sizemode='raw',sizeref=1,showscale=False,
                          opacity=cnstrn_alpha,name='Orbiting'),row=1,col=2)
    fig.add_trace(go.Cone(x=[halo_pos[0]], y=[halo_pos[1]], z=[halo_pos[2]],
                          u=[halo_vel[0]], v=[halo_vel[1]], w=[halo_vel[2]],
                          colorscale=[[0, 'red'], [1, 'red']],sizemode='raw',sizeref=1,showscale=False,
                          opacity=1,name='Halo Center',showlegend=True),row=1,col=2)

    fig.update_scenes(
        xaxis=dict(title='X position (kpc/h)', range=[halo_pos[0] - 10 * halo_r200m,halo_pos[0] + 10 * halo_r200m]),
        yaxis=dict(title='Y position (kpc/h)', range=[halo_pos[1] - 10 * halo_r200m,halo_pos[1] + 10 * halo_r200m]),
        zaxis=dict(title='Z position (kpc/h)', range=[halo_pos[2] - 10 * halo_r200m,halo_pos[2] + 10 * halo_r200m]),
        row=1,col=1)
    fig.update_scenes(
        xaxis=dict(title='X position (kpc/h)', range=[halo_pos[0] - 10 * halo_r200m,halo_pos[0] + 10 * halo_r200m]),
        yaxis=dict(title='Y position (kpc/h)', range=[halo_pos[1] - 10 * halo_r200m,halo_pos[1] + 10 * halo_r200m]),
        zaxis=dict(title='Z position (kpc/h)', range=[halo_pos[2] - 10 * halo_r200m,halo_pos[2] + 10 * halo_r200m]),
        row=1,col=2)
    fig.write_html(MLOIS_path + "/Random_figs/high_vel_halo_idx_" + str(halo_idx) + ".html")
    
def plot_rad_dist(bin_edges,filter_radii,save_path):
    fig,ax = plt.subplots(1,2,figsize=(25,10))
    ax[0].hist(filter_radii)
    ax[0].set_xlabel("Radius $r/R_{200m}$")
    ax[0].set_ylabel("counts")
    ax[1].hist(filter_radii,bins=bin_edges)
    ax[1].set_xlabel("Radius $r/R_{200m}$")
    ax[1].set_xscale("log")
    print("num ptl within 2 R200m", np.where(filter_radii < 2)[0].shape)
    print("num ptl outside 2 R200m", np.where(filter_radii > 2)[0].shape)
    print("ratio in/out", np.where(filter_radii < 2)[0].shape[0] / np.where(filter_radii > 2)[0].shape[0])
    fig.savefig(save_path + "radii_dist.png",bbox_inches="tight")

def plot_orb_inf_dist(num_bins, radii, orb_inf, save_path):
    lin_orb_cnt = np.zeros(num_bins)
    lin_inf_cnt = np.zeros(num_bins)
    log_orb_cnt = np.zeros(num_bins)
    log_inf_cnt = np.zeros(num_bins)

    lin_bins = np.linspace(0, np.max(radii), num_bins + 1)
    log_bins = np.logspace(np.log10(0.1),np.log10(np.max(radii)),num_bins+1)

    # Count particles in each bin
    for i in range(num_bins):
        lin_bin_mask = (radii >= lin_bins[i]) & (radii < lin_bins[i + 1])
        lin_orb_cnt[i] = np.sum((orb_inf == 1) & lin_bin_mask)
        lin_inf_cnt[i] = np.sum((orb_inf == 0) & lin_bin_mask)

        log_bin_mask = (radii >= log_bins[i]) & (radii < log_bins[i + 1])
        log_orb_cnt[i] = np.sum((orb_inf == 1) & log_bin_mask)
        log_inf_cnt[i] = np.sum((orb_inf == 0) & log_bin_mask)
    # Plotting
    bar_width = 0.35  # width of the bars
    index = np.arange(num_bins)  # the label locations

    fig, ax = plt.subplots(1,2,figsize=(35,10))
    ax[0].bar(index, lin_orb_cnt, bar_width, label='Orbiting')
    ax[0].bar(index + bar_width, lin_inf_cnt, bar_width, label='Infalling')
    ax[0].set_xlabel('Radius Bins')
    ax[0].set_ylabel('Number of Particles')
    ax[0].set_title('Number of Orbiting and Infalling Particles by Radius Bin')
    ax[0].set_xticks(index + bar_width / 2)
    ax[0].set_xticklabels([f'{lin_bins[i]:.1f}-{lin_bins[i + 1]:.1f}' for i in range(num_bins)],rotation=90)
    ax[0].legend(frameon=False)
    
    ax[1].bar(index, log_orb_cnt, bar_width, label='Orbiting',log=True)
    ax[1].bar(index + bar_width, log_inf_cnt, bar_width, label='Infalling',log=True)
    ax[1].set_xlabel('Radius Bins')
    ax[1].set_title('Number of Orbiting and Infalling Particles by Radius Bin')
    ax[1].set_xticks(index + bar_width / 2)
    ax[1].set_xticklabels([f'{log_bins[i]:.2f}-{log_bins[i + 1]:.2f}' for i in range(num_bins)],rotation=90)
    ax[1].legend(frameon=False)
    
    fig.savefig(save_path + "orb_inf_dist.png",bbox_inches="tight")

