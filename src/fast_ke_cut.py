import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from scipy.optimize import curve_fit, minimize

plt.rcParams.update({"text.usetex":True, "font.family": "serif", "figure.dpi": 150})
import os
import multiprocessing as mp
from dask.distributed import Client
import json
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import pickle
import scipy.ndimage as ndimage
from sparta_tools import sparta

from utils.calculation_functions import create_stack_mass_prf, filter_prf, calculate_density, calc_mass_acc_rate
from src.utils.vis_fxns import compare_split_prfs
from utils.ML_support import setup_client, get_combined_name, reform_dataset_dfs, parse_ranges, load_sparta_mass_prf, split_sparta_hdf5_name, get_feature_labels, get_model_name, extract_snaps
from utils.data_and_loading_functions import create_directory, timed, save_pickle, load_pickle, load_SPARTA_data, conv_halo_id_spid, load_config, depair_np, set_cosmology
from src.utils.ke_cut_support import load_ke_data, fast_ke_predictor

config_params = load_config(os.getcwd() + "/config.ini")

ML_dset_path = config_params["PATHS"]["ml_dset_path"]
path_to_models = config_params["PATHS"]["path_to_models"]
rockstar_ctlgs_path = config_params["PATHS"]["rockstar_ctlgs_path"]

SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]

features = config_params["TRAIN_MODEL"]["features"]
target_column = config_params["TRAIN_MODEL"]["target_column"]

eval_datasets = config_params["EVAL_MODEL"]["eval_datasets"]

sim_cosmol = config_params["MISC"]["sim_cosmol"]

plt_nu_splits = parse_ranges(config_params["EVAL_MODEL"]["plt_nu_splits"])
plt_macc_splits = parse_ranges(config_params["EVAL_MODEL"]["plt_macc_splits"])

linthrsh = config_params["EVAL_MODEL"]["linthrsh"]
lin_nbin = config_params["EVAL_MODEL"]["lin_nbin"]
log_nbin = config_params["EVAL_MODEL"]["log_nbin"]
lin_rvticks = config_params["EVAL_MODEL"]["lin_rvticks"]
log_rvticks = config_params["EVAL_MODEL"]["log_rvticks"]
lin_tvticks = config_params["EVAL_MODEL"]["lin_tvticks"]
log_tvticks = config_params["EVAL_MODEL"]["log_tvticks"]
lin_rticks = config_params["EVAL_MODEL"]["lin_rticks"]
log_rticks = config_params["EVAL_MODEL"]["log_rticks"]

fast_ke_calib_sims = config_params["KE_CUT"]["fast_ke_calib_sims"]
n_points = config_params["KE_CUT"]["n_points"]
perc = config_params["KE_CUT"]["perc"]
width = config_params["KE_CUT"]["width"]
grad_lims = config_params["KE_CUT"]["grad_lims"]
r_cut_calib = config_params["KE_CUT"]["r_cut_calib"]
r_cut_pred = config_params["KE_CUT"]["r_cut_pred"]
ke_test_sims = config_params["KE_CUT"]["ke_test_sims"]

####################################################################################################################################################################################################################################
    
def gradient_minima(
    r: np.ndarray,
    lnv2: np.ndarray,
    mask_vr: np.ndarray,
    n_points: int,
    r_min: float,
    r_max: float,
):
    """Computes the r-lnv2 gradient and finds the minimum as a function of `r`
    within the interval `[r_min, r_max]`

    Parameters
    ----------
    r : np.ndarray
        Radial separation
    lnv2 : np.ndarray
        Log-kinetic energy
    mask_vr : np.ndarray
        Mask for the selection of radial velocity
    n_points : int
        Number of minima points to compute
    r_min : float
        Minimum radial distance
    r_max : float
        Maximum radial distance

    Returns
    -------
    Tuple[np.ndarray]
        Radial and minima coordinates.
    """
    r_edges_grad = np.linspace(r_min, r_max, n_points + 1)
    grad_r = 0.5 * (r_edges_grad[:-1] + r_edges_grad[1:])
    grad_min = np.zeros(n_points)
    for i in range(n_points):
        r_mask = (r > r_edges_grad[i]) * (r < r_edges_grad[i + 1])
        hist_yv, hist_edges = np.histogram(lnv2[mask_vr * r_mask], bins=200)
        hist_lnv2 = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        hist_lnv2_grad = np.gradient(hist_yv, np.mean(np.diff(hist_edges)))
        lnv2_mask = (1.0 < hist_lnv2) * (hist_lnv2 < r_cut_calib)
        grad_min[i] = hist_lnv2[lnv2_mask][np.argmin(hist_lnv2_grad[lnv2_mask])]

    return grad_r, grad_min

def cost_percentile(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the `target` percentile of paricles is below the line.

    Parameters
    ----------
    b : float
        Fit parameter
    *data : tuple
        A tuple with `[r, lnv2, slope, target]`, where `slope` is the slope of 
        the line and is fixed, and `target` is the desired percentile

    Returns
    -------
    float
    """
    r, lnv2, slope, target = data
    below_line = (lnv2 < (slope * r + b)).sum()
    return np.log((target - below_line / r.shape[0]) ** 2)

def cost_perp_distance(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the perpendicular distance of all points to the line is maximal.

    Parameters
    ----------
    b : float
        Fit parameter
    *data: tuple
        A tuple with `[r, lnv2, slope, width]`, where `slope` is the slope of 
        the line and is fixed, and `width` is the width of a band around the 
        line within which the distance is computed

    Returns
    -------
    float
        _description_
    """
    r, lnv2, slope, width = data
    d = np.abs(lnv2 - slope * r - b) / np.sqrt(1 + slope**2)
    return -np.log(np.mean(d[(d < width)] ** 2))

def calibrate_finder(
    n_points: int = n_points,
    perc: float = perc,
    width: float = width,
    grad_lims: tuple = grad_lims,
):
    """_summary_

    Parameters
    ----------
    n_points : int, optional
        Number of minima points to compute, by default 20
    perc : float, optional
        Target percentile for the positive radial velocity calibration, 
        by default 0.98
    width : float, optional
        Band width for the negattive radial velocity calibration, 
        by default 0.05
    grad_lims : tuple
        Radial interval in which the gradient is to be computed, by default
        [0.2, 0.5]
    """
    # MODIFY this line if needed ======================
    r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ke_data(client,fast_ke_calib_sims,sim_cosmol,snap_list)

    # =================================================

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r < r_cut_calib

    # For vr > 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_pos, n_points, *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * x + b, r_grad, min_grad, p0=[-1, 2])
    m_pos, b01 = popt

    # Find intercept by finding the value that contains 'perc' percent of 
    # particles below the line at fixed slope 'm_pos'
    res = minimize(
        cost_percentile,
        1.1 * b01,
        bounds=((0.8 * b01, 3.0),),
        args=(r[mask_vr_pos * mask_r], lnv2[mask_vr_pos * mask_r], m_pos, perc),
        method='Nelder-Mead',
    )
    b_pos = res.x[0]

    # For vr < 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_neg, n_points, *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * x + b, r_grad, min_grad, p0=[-1, 2])
    m_neg, b02 = popt

    # Find intercept by finding the value that maximizes the perpendicular
    # distance to the line at fixed slope of all points within a perpendicular
    # 'width' distance from the line (ignoring all others).
    res = minimize(
        cost_perp_distance,
        0.75 * b02,
        bounds=((1.2, b02),),
        args=(r[mask_vr_neg], lnv2[mask_vr_neg], m_neg, width),
        method='Nelder-Mead',
    )
    b_neg = res.x[0]

    return (m_pos, b_pos), (m_neg, b_neg)
    
if __name__ == "__main__":
    client = setup_client()
    
    comb_model_sims = get_combined_name(fast_ke_calib_sims) 
    
    model_type = "kinetic_energy_cut"
    model_name = get_model_name(model_type, fast_ke_calib_sims)    
    model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"  
    create_directory(model_fldr_loc)
    
    #TODO make this a loop for all the test sims
    curr_test_sims = ke_test_sims[0]
    test_comb_name = get_combined_name(curr_test_sims) 
    dset_name = eval_datasets[0]
    plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
    create_directory(plot_loc)
    
    dset_params = load_pickle(ML_dset_path + curr_test_sims[0] + "/dset_params.pickle")
    sim_cosmol = dset_params["cosmology"]
    all_tdyn_steps = dset_params["t_dyn_steps"]
    
    feature_columns = get_feature_labels(features,all_tdyn_steps)
    snap_list = extract_snaps(fast_ke_calib_sims[0])
    cosmol = set_cosmology(sim_cosmol)
    
    ####################################################################################################################################################################################################################################
    
    if os.path.exists(model_fldr_loc + "ke_fastparams_dict.pickle"):
        ke_param_dict = load_pickle(model_fldr_loc + "ke_fastparams_dict.pickle")
        m_pos = ke_param_dict["m_pos"]
        b_pos = ke_param_dict["b_pos"]
        m_neg = ke_param_dict["m_neg"]
        b_neg = ke_param_dict["b_neg"]
    else:
        (m_pos, b_pos), (m_neg, b_neg) = calibrate_finder()
          
        ke_param_dict = {
            "m_pos":m_pos,
            "b_pos":b_pos,
            "m_neg":m_neg,
            "b_neg":b_neg
        }
        save_pickle(ke_param_dict, model_fldr_loc + "ke_fastparams_dict.pickle")
    
    print("\nCalibration Params")
    print(m_pos,b_pos,m_neg,b_neg)
    print("\n")
    
    r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ke_data(client, curr_test_sims,sim_cosmol,snap_list)
    
    r_r200m = my_data["p_Scaled_radii"].compute().to_numpy()
    vr = my_data["p_Radial_vel"].compute().to_numpy()
    vt = my_data["p_Tangential_vel"].compute().to_numpy()
    vphys = my_data["p_phys_vel"].compute().to_numpy()
    sparta_labels = my_data["Orbit_infall"].compute().to_numpy()
    hipids = my_data["HIPIDS"].compute().to_numpy()
    all_pids, ptl_halo_idxs = depair_np(hipids)
    lnv2 = np.log(vphys**2)
    
    sparta_orb = np.where(sparta_labels == 1)[0]
    sparta_inf = np.where(sparta_labels == 0)[0]

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    
    fltr_combs = {
    "orb_vr_neg": np.intersect1d(sparta_orb, np.where(mask_vr_neg)[0]),
    "orb_vr_pos": np.intersect1d(sparta_orb, np.where(mask_vr_pos)[0]),
    "inf_vr_neg": np.intersect1d(sparta_inf, np.where(mask_vr_neg)[0]),
    "inf_vr_pos": np.intersect1d(sparta_inf, np.where(mask_vr_pos)[0]),
    }

    # Look for particles vr>0 that are above the phase space line (infalling) that SPARTA says should be below (orbiting)
    wrong_vr_pos_orb = (lnv2[fltr_combs["orb_vr_pos"]] > (m_pos * r_r200m[fltr_combs["orb_vr_pos"]] + b_pos)) & (r_r200m[fltr_combs["orb_vr_pos"]] < r_cut_pred)
    # Look for particles vr>0 that are below the phase space line (orbiting) that SPARTA says should be above (infalling)
    wrong_vr_pos_inf = (lnv2[fltr_combs["inf_vr_pos"]] < (m_pos * r_r200m[fltr_combs["inf_vr_pos"]] + b_pos)) & (r_r200m[fltr_combs["inf_vr_pos"]] < r_cut_pred)

    # Look for particles vr<0 that are above the phase space line (infalling) that SPARTA says should be below (orbiting)
    wrong_vr_neg_orb = (lnv2[fltr_combs["orb_vr_neg"]] > (m_neg * r_r200m[fltr_combs["orb_vr_neg"]] + b_neg)) & (r_r200m[fltr_combs["orb_vr_neg"]] < r_cut_pred)
    # Look for particles vr<0 that are below the phase space line (orbiting) that SPARTA says should be above (infalling)
    wrong_vr_neg_inf = (lnv2[fltr_combs["inf_vr_neg"]] < (m_neg * r_r200m[fltr_combs["inf_vr_neg"]] + b_neg)) & (r_r200m[fltr_combs["inf_vr_neg"]] < r_cut_pred)

    #total num wrong ptls
    wrong_vr_pos_total = wrong_vr_pos_orb.sum() + wrong_vr_pos_inf.sum()
    wrong_vr_neg_total = wrong_vr_neg_orb.sum() + wrong_vr_neg_inf.sum()

    wrong_inf_total = wrong_vr_pos_inf.sum() + wrong_vr_neg_inf.sum()
    wrong_orb_total = wrong_vr_pos_orb.sum() + wrong_vr_neg_orb.sum()

    num_vr_neg = mask_vr_neg.sum()
    num_vr_pos = mask_vr_pos.sum()
    
    print("Fraction of negative radial velocity particles incorrectly classified:", wrong_vr_neg_total / num_vr_neg)
    print("Fraction of positive radial velocity particles incorrectly classified:", wrong_vr_pos_total / num_vr_pos)
    print("Fraction of infalling particles incorrectly classified:", wrong_inf_total / sparta_inf.shape[0])
    print("Fraction of orbiting particles incorrectly classified:", wrong_orb_total / sparta_orb.shape[0])

    split_scale_dict = {
            "linthrsh":linthrsh, 
            "lin_nbin":lin_nbin,
            "log_nbin":log_nbin,
            "lin_rvticks":lin_rvticks,
            "log_rvticks":log_rvticks,
            "lin_tvticks":lin_tvticks,
            "log_tvticks":log_tvticks,
            "lin_rticks":lin_rticks,
            "log_rticks":log_rticks,
    }

    # y_shift = w / np.cos(np.arctan(m_neg))
    x = np.linspace(0, 3, 1000)
    y12 = m_pos * x + b_pos
    y22 = m_neg * x + b_neg

    nbins = 200   
    
    x_range = (0, 3)
    y_range = (-2, 2.5)

    hist1, xedges, yedges = np.histogram2d(r_r200m[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], bins=nbins, range=(x_range, y_range))
    hist2, _, _ = np.histogram2d(r_r200m[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], bins=nbins, range=(x_range, y_range))
    hist3, _, _ = np.histogram2d(r_r200m[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], bins=nbins, range=(x_range, y_range))
    hist4, _, _ = np.histogram2d(r_r200m[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], bins=nbins, range=(x_range, y_range))

    # Combine the histograms to determine the maximum density for consistent color scaling
    combined_hist = np.maximum.reduce([hist1, hist2, hist3, hist4])
    vmax=combined_hist.max()
    
    lin_vmin = 0
    log_vmin = 1

    title_fntsize = 22
    legend_fntsize = 18
    axis_fntsize = 20
    txt_fntsize = 20
    cbar_label_fntsize = 18
    cbar_tick_fntsize = 14

    with timed("SPARTA KE Dist plot"):
        magma_cmap = plt.get_cmap("magma")
        magma_cmap.set_under(color='black')
        magma_cmap.set_bad(color='black') 
        
        widths = [4,4,4,4,.5]
        heights = [0.15,4,4]
        fig = plt.figure(constrained_layout=True, figsize=(28,14))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

        
        fig.suptitle(
            r"Kinetic Energy Distribution of Particles Around Largest Halos at $z=0.03$""\nSimulation: Bolshoi 1000Mpc",fontsize=title_fntsize)
        
        ax1 = fig.add_subplot(gs[1,0])
        ax2 = fig.add_subplot(gs[1,1])
        ax3 = fig.add_subplot(gs[1,2])
        ax4 = fig.add_subplot(gs[1,3])
        ax5 = fig.add_subplot(gs[2,0])
        ax6 = fig.add_subplot(gs[2,1])
        ax7 = fig.add_subplot(gs[2,2])
        ax8 = fig.add_subplot(gs[2,3])
        
        axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
        
        ax1.set_ylabel(r'$\ln(v^2/v_{200m}^2)$',fontsize=axis_fntsize)
        ax5.set_ylabel(r'$\ln(v^2/v_{200m}^2)$',fontsize=axis_fntsize)
        ax5.set_xlabel(r'$r/R_{200m}$',fontsize=axis_fntsize)
        ax6.set_xlabel(r'$r/R_{200m}$',fontsize=axis_fntsize)
        ax7.set_xlabel(r'$r/R_{200m}$',fontsize=axis_fntsize)
        ax8.set_xlabel(r'$r/R_{200m}$',fontsize=axis_fntsize)
        
        ax1.tick_params('x', labelbottom=False,colors="white",direction="in")
        ax2.tick_params('x', labelbottom=False,colors="white",direction="in")
        ax2.tick_params('y', labelleft=False,colors="white",direction="in")
        ax3.tick_params('x', labelbottom=False,colors="white",direction="in")
        ax3.tick_params('y', labelleft=False,colors="white",direction="in")
        ax4.tick_params('x', labelbottom=False,colors="white",direction="in")
        ax4.tick_params('y', labelleft=False,colors="white",direction="in")
        ax6.tick_params('y', labelleft=False,colors="white",direction="in")
        ax7.tick_params('y', labelleft=False,colors="white",direction="in")
        ax8.tick_params('y', labelleft=False,colors="white",direction="in")

        
        for ax in axes:
            ax.text(0.25, -1.4, "Orbiting", fontsize=txt_fntsize, color="r",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.text(1.4, 0.7, "Infalling", fontsize=txt_fntsize, color="b",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.tick_params(axis='both',which='both',labelcolor="black",colors="white",direction="in",pad=5,labelsize=16,length=8,width=2)

        plt.sca(axes[0])
        plt.title("Orbiting Particles: "r'$v_r > 0$',fontsize=title_fntsize)
        plt.hist2d(r_r200m[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], bins=nbins, vmin=lin_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="g",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        plt.ylim(-2.2, 2.5)
        
        plt.sca(axes[1])
        plt.title("Orbiting Particles: "r'$v_r < 0$',fontsize=title_fntsize)
        plt.hist2d(r_r200m[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], bins=nbins, vmin=lin_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="g",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        plt.ylim(-2.2, 2.5)
        
        plt.sca(axes[2])
        plt.title("Infalling Particles: "r'$v_r > 0$',fontsize=title_fntsize)
        plt.hist2d(r_r200m[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], bins=nbins, vmin=lin_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="g",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        plt.ylim(-2.2, 2.5)

        plt.sca(axes[3])
        plt.title("Infalling Particles: "r'$v_r < 0$',fontsize=title_fntsize)
        plt.hist2d(r_r200m[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], bins=nbins, vmin=lin_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="g",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        cbar_lin= plt.colorbar()
        cbar_lin.ax.tick_params(labelsize=cbar_tick_fntsize)
        cbar_lin.set_label(r'$N$ (Counts)', fontsize=cbar_label_fntsize)
        plt.xlim(0, 2)
        plt.ylim(-2.2, 2.5)
        
        plt.sca(axes[4])
        plt.hist2d(r_r200m[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], bins=nbins, norm="log", vmin=log_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="g",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        plt.ylim(-2.2, 2.5)

        plt.sca(axes[5])
        plt.hist2d(r_r200m[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], bins=nbins, norm="log", vmin=log_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="g",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        plt.ylim(-2.2, 2.5)

        plt.sca(axes[6])
        plt.hist2d(r_r200m[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], bins=nbins, norm="log", vmin=log_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="g",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        plt.ylim(-2.2, 2.5)

        plt.sca(axes[7])
        plt.hist2d(r_r200m[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], bins=nbins, norm="log", vmin=log_vmin, vmax=vmax,
                    cmap=magma_cmap, range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="g",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        cbar_log= plt.colorbar()
        cbar_log.ax.tick_params(labelsize=cbar_tick_fntsize)
        cbar_log.set_label(r'$N$ (Counts)', fontsize=cbar_label_fntsize)
        plt.xlim(0, 2)
        plt.ylim(-2.2, 2.5)
    
        plt.savefig(plot_loc + "sparta_KE_dist_cut.png",bbox_inches='tight',dpi=500)
    
    with timed("KE Dist plot"):
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        axes = axes.flatten()
        fig.suptitle(
            r"Kinetic energy distribution of particles around halos at $z=0$""\nSimulation: Bolshoi 1000Mpc",fontsize=16)

        for ax in axes:
            ax.set_xlabel(r'$r/R_{200m}$',fontsize=16)
            ax.set_ylabel(r'$\ln(v^2/v_{200m}^2)$',fontsize=16)
            ax.set_xlim(0, 2)
            ax.set_ylim(-2, 2.5)
            ax.text(0.25, -1.4, "Orbiting", fontsize=16, color="r",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.text(1.5, 0.7, "Infalling", fontsize=16, color="b",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.tick_params(axis='both',which='both',direction="in",labelsize=12,length=8,width=2)

        plt.sca(axes[0])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        plt.hist2d(r_r200m[mask_vr_pos], lnv2[mask_vr_pos], bins=nbins,
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="k",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        

        plt.sca(axes[1])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        plt.hist2d(r_r200m[mask_vr_neg], lnv2[mask_vr_neg], bins=nbins,
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="k",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        plt.sca(axes[2])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        h3 = plt.hist2d(r_r200m[mask_vr_pos], lnv2[mask_vr_pos], bins=nbins, norm="log",
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="k",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(h3[3], label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[3])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        h4 = plt.hist2d(r_r200m[mask_vr_neg], lnv2[mask_vr_neg], bins=nbins, norm="log",
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="k",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(h4[3], label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        mask_vrn = (vr < 0)
        mask_vrp = ~mask_vrn

        # Compute density and gradient.
        # For vr > 0
        hist_zp, hist_xp, hist_yp = np.histogram2d(r_r200m[mask_vrp], lnv2[mask_vrp], 
                                                    bins=nbins, 
                                                    range=((0, 3.), (-2, 2.5)),
                                                    density=True)
        # Bin centres
        hist_xp = 0.5 * (hist_xp[:-1] + hist_xp[1:])
        hist_yp = 0.5 * (hist_yp[:-1] + hist_yp[1:])
        # Bin spacing
        dx = np.mean(np.diff(hist_xp))
        dy = np.mean(np.diff(hist_yp))
        # Generate a 2D grid corresponding to the histogram
        hist_xp, hist_yp = np.meshgrid(hist_xp, hist_yp)
        # Evaluate the gradient at each radial bin
        hist_z_grad = np.zeros_like(hist_zp)
        for i in range(hist_xp.shape[0]):
            hist_z_grad[i, :] = np.gradient(hist_zp[i, :], dy)
        # Apply a gaussian filter to smooth the gradient.
        hist_zp = ndimage.gaussian_filter(hist_z_grad, 2.0)

        # Same for vr < 0
        hist_zn, hist_xn, hist_yn = np.histogram2d(r_r200m[mask_vrn], lnv2[mask_vrn],
                                                    bins=nbins,
                                                    range=((0, 3.), (-2, 2.5)),
                                                    density=True)
        hist_xn = 0.5 * (hist_xn[:-1] + hist_xn[1:])
        hist_yn = 0.5 * (hist_yn[:-1] + hist_yn[1:])
        dy = np.mean(np.diff(hist_yn))
        hist_xn, hist_yn = np.meshgrid(hist_xn, hist_yn)
        hist_z_grad = np.zeros_like(hist_zn)
        for i in range(hist_xn.shape[0]):
            hist_z_grad[i, :] = np.gradient(hist_zn[i, :], dy)
        hist_zn = ndimage.gaussian_filter(hist_z_grad, 2.0)

        #Plot the smoothed gradient
        plt.sca(axes[4])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        plt.contourf(hist_xp, hist_yp, hist_zp.T, levels=80, cmap='terrain')
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label="Smoothed Gradient Magnitude")
        plt.xlim(0, 2)
        
        # Plot the smoothed gradient
        plt.sca(axes[5])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        plt.contourf(hist_xn, hist_yn, hist_zn.T, levels=80, cmap='terrain')
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label="Smoothed Gradient Magnitude")
        plt.xlim(0, 2)
        
        

        plt.tight_layout();
        plt.savefig(plot_loc + "perc_" + str(perc) + "_grd_" + str(grad_lims[0]) + str(grad_lims[1]) + "_KE_dist_cut.png")
        
    
    mask_orb, ke_cut_preds = fast_ke_predictor(ke_param_dict,r_r200m,vr,lnv2,r_cut_pred)
    
    with timed("Phase space dist of ptls plot"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes = axes.flatten()
        fig.suptitle(f'Phase-space distribution of particles around haloes at $z=0$')

        x_range = (0, 3)
        y_range = (-2, 2)

        for ax in axes:
            ax.set_xlabel(r'$r/R_{200}$')
            ax.set_ylabel(r'$v_r/v_{200}$')

        plt.sca(axes[0])
        plt.title('Orbiting')
        plt.hist2d(r_r200m[mask_orb], vr[mask_orb], bins=nbins, range=(x_range, y_range), cmap='terrain')
        plt.colorbar(label=r'$N$ (Counts)')

        plt.sca(axes[1])
        plt.title('Infalling')
        plt.hist2d(r_r200m[~mask_orb], vr[~mask_orb], bins=nbins, range=(x_range, y_range), cmap='terrain')
        plt.colorbar(label=r'$N$ (Counts)')

        plt.tight_layout()
        plt.savefig(plot_loc + "perc_" + str(perc) + "_grd_" + str(grad_lims[0]) + str(grad_lims[1]) + "_ps_dist.png")
    
    
####################################################################################################################################################################################################################################

    with timed("Density profile plot"):
        print("Testing on:", curr_test_sims)
        # Loop through and/or for Train/Test/All datasets and evaluate the model
        
        r_r200m = my_data["p_Scaled_radii"]
        vr = my_data["p_Radial_vel"]
        vphys = my_data["p_phys_vel"]
        lnv2 = np.log(vphys**2)
        
        #TODO rename the different masks and preds
        mask_orb, ke_cut_preds = fast_ke_predictor(ke_param_dict,r_r200m.values.compute(),vr.values.compute(),lnv2.values.compute(),r_cut_pred)
        
        X = my_data[feature_columns]
        y = my_data[target_column]
        halo_first = halo_df["Halo_first"].values
        halo_n = halo_df["Halo_n"].values
        all_idxs = halo_df["Halo_indices"].values

        all_z = []
        all_rhom = []
        # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
        sim_splits = np.where(halo_first == 0)[0]

        use_sims = curr_test_sims

        # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
        # stacked simulations such that they correspond to the larger dataset and not one specific simulation
        if len(use_sims) > 1:
            for i,sim in enumerate(use_sims):
                # The first sim remains the same
                if i == 0:
                    continue
                # Else if it isn't the final sim 
                elif i < len(use_sims) - 1:
                    halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
                # Else if the final sim
                else:
                    halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])

        # Get the redshifts for each simulation's primary snapshot
        for i,sim in enumerate(use_sims):
            with open(ML_dset_path + sim + "/dset_params.pickle", "rb") as file:
                dset_params = pickle.load(file)
                curr_z = dset_params["all_snap_info"]["prime_snap_info"]["red_shift"]
                curr_rho_m = dset_params["all_snap_info"]["prime_snap_info"]["rho_m"]
                all_z.append(curr_z)
                all_rhom.append(curr_rho_m)
                h = dset_params["all_snap_info"]["prime_snap_info"]["h"]

        tot_num_halos = halo_n.shape[0]
        min_disp_halos = int(np.ceil(0.3 * tot_num_halos))

        # Get SPARTA's mass profiles
        act_mass_prf_all, act_mass_prf_orb,all_masses,bins = load_sparta_mass_prf(sim_splits,all_idxs,use_sims)
        act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb

        # Create mass profiles from the model's predictions
        preds = np.zeros(X.shape[0].compute())
        preds[mask_orb] = 1

        calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m = create_stack_mass_prf(sim_splits,radii=X["p_Scaled_radii"].values.compute(), halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds, prf_bins=bins, use_mp=True, all_z=all_z)

        # Halos that get returned with a nan R200m mean that they didn't meet the required number of ptls within R200m and so we need to filter them from our calculated profiles and SPARTA profiles 
        small_halo_fltr = np.isnan(calc_r200m)
        act_mass_prf_all[small_halo_fltr,:] = np.nan
        act_mass_prf_orb[small_halo_fltr,:] = np.nan
        act_mass_prf_inf[small_halo_fltr,:] = np.nan

        # Calculate the density by divide the mass of each bin by the volume of that bin's radius
        calc_dens_prf_all = calculate_density(calc_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        calc_dens_prf_orb = calculate_density(calc_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        calc_dens_prf_inf = calculate_density(calc_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)

        act_dens_prf_all = calculate_density(act_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        act_dens_prf_orb = calculate_density(act_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
        act_dens_prf_inf = calculate_density(act_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)

        # If we want the density profiles to only consist of halos of a specific peak height (nu) bin 

        all_prf_lst = []
        orb_prf_lst = []
        inf_prf_lst = []
        cpy_plt_nu_splits = plt_nu_splits.copy()
        for i,nu_split in enumerate(cpy_plt_nu_splits):
            # Take the second element of the where to filter by the halos (?)
            fltr = np.where((calc_nus > nu_split[0]) & (calc_nus < nu_split[1]))[0]
            if fltr.shape[0] > 25:
                all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
                orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
                inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
            else:
                plt_nu_splits.remove(nu_split)
                
        curr_halos_r200m_list = []
        past_halos_r200m_list = []                
                
        for sim in use_sims:
            dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
            p_snap = dset_params["all_snap_info"]["prime_snap_info"]["ptl_snap"][()]
            curr_z = dset_params["all_snap_info"]["prime_snap_info"]["red_shift"][()]
            # TODO make this generalizable to when the snapshot separation isn't just 1 dynamical time as needed for mass accretion calculation
            # we can just use the secondary snap here because we already chose to do 1 dynamical time for that snap
            past_z = dset_params["all_snap_info"]["comp_" + str(all_tdyn_steps[0]) + "_tdstp_snap_info"]["red_shift"][()] 
            p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"][()]
            c_sparta_snap = dset_params["all_snap_info"]["comp_" + str(all_tdyn_steps[0]) + "_tdstp_snap_info"]["sparta_snap"][()]
            
            sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
            
            curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"
                    
            # Load the halo's positions and radii
            param_paths = [["halos","R200m"],["halos","id"]]
            sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name)

            curr_halos_r200m = sparta_params[sparta_param_names[0]][:,p_sparta_snap]
            curr_halos_ids = sparta_params[sparta_param_names[1]][:,p_sparta_snap]
            
            halo_ddf = reform_dataset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/")
            curr_idxs = halo_ddf["Halo_indices"].values
            
            use_halo_r200m = curr_halos_r200m[curr_idxs]
            use_halo_ids = curr_halos_ids[curr_idxs]
            
            sparta_output = sparta.load(filename=curr_sparta_HDF5_path, halo_ids=use_halo_ids, log_level=0)
            new_idxs = conv_halo_id_spid(use_halo_ids, sparta_output, p_sparta_snap) # If the order changed by sparta re-sort the indices
            
            curr_halos_r200m_list.append(sparta_output['halos']['R200m'][:,p_sparta_snap])
            past_halos_r200m_list.append(sparta_output['halos']['R200m'][:,c_sparta_snap])
            
        curr_halos_r200m = np.concatenate(curr_halos_r200m_list)
        past_halos_r200m = np.concatenate(past_halos_r200m_list)
            
        calc_maccs = calc_mass_acc_rate(curr_halos_r200m,past_halos_r200m,curr_z,past_z)

        cpy_plt_macc_splits = plt_macc_splits.copy()
        for i,macc_split in enumerate(cpy_plt_macc_splits):
            # Take the second element of the where to filter by the halos (?)
            fltr = np.where((calc_maccs > macc_split[0]) & (calc_maccs < macc_split[1]))[0]
            if fltr.shape[0] > 25:
                all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
                orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
                inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
            else:
                plt_macc_splits.remove(macc_split)
                
        compare_split_prfs(plt_nu_splits,len(cpy_plt_nu_splits),all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_loc,title= "perc_" + str(perc) + "_" + "_grd_" + str(grad_lims[0]) + str(grad_lims[1]) + "_ke_cut_dens_",prf_name_0="Kinetic Energy Cut", prf_name_1="SPARTA")
        compare_split_prfs(plt_macc_splits,len(cpy_plt_macc_splits),all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_loc,title= "perc_" + str(perc) + "_" + "_grd_" + str(grad_lims[0]) + str(grad_lims[1]) + "_ke_cut_macc_dens_", split_name="\Gamma", prf_name_0="Kinetic Energy Cut", prf_name_1="SPARTA")