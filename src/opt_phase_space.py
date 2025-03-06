import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.rcParams.update({"text.usetex":True, "font.family": "serif", "figure.dpi": 150})
import os
import multiprocessing as mp
from dask.distributed import Client
import json
import pandas as pd
import matplotlib as mpl
import scipy.ndimage as ndimage
mpl.rcParams.update(mpl.rcParamsDefault)
from colossus.cosmology import cosmology

from utils.ML_support import get_CUDA_cluster, get_combined_name, parse_ranges, create_nu_string, load_sparta_mass_prf
from utils.data_and_loading_functions import create_directory, timed
from utils.ps_cut_support import load_ps_data
from utils.update_vis_fxns import plt_SPARTA_KE_dist, plt_cust_ps_line

import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")

on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")

ML_dset_path = config["PATHS"]["ML_dset_path"]
path_to_models = config["PATHS"]["path_to_models"]
SPARTA_output_path = config["PATHS"]["SPARTA_output_path"]

model_sims = json.loads(config.get("XGBOOST","model_sims"))
dask_task_cpus = config.getint("XGBOOST","dask_task_cpus")
model_type = config["XGBOOST"]["model_type"]
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))
dask_task_cpus = config.getint("XGBOOST","dask_task_cpus")

sim_cosmol = config["MISC"]["sim_cosmol"]
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
    cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
else:
    cosmol = cosmology.setCosmology(sim_cosmol) 
    sim_pat = r"cbol_l(\d+)_n(\d+)"
    
plt_nu_splits = config["XGBOOST"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)

plt_macc_splits = config["XGBOOST"]["plt_macc_splits"]
plt_macc_splits = parse_ranges(plt_macc_splits)

nu_splits = config["XGBOOST"]["nu_splits"]
nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

linthrsh = config.getfloat("XGBOOST","linthrsh")
lin_nbin = config.getint("XGBOOST","lin_nbin")
log_nbin = config.getint("XGBOOST","log_nbin")
lin_rvticks = json.loads(config.get("XGBOOST","lin_rvticks"))
log_rvticks = json.loads(config.get("XGBOOST","log_rvticks"))
lin_tvticks = json.loads(config.get("XGBOOST","lin_tvticks"))
log_tvticks = json.loads(config.get("XGBOOST","log_tvticks"))
lin_rticks = json.loads(config.get("XGBOOST","lin_rticks"))
log_rticks = json.loads(config.get("XGBOOST","log_rticks"))

if on_zaratan:
    from dask_mpi import initialize
    from distributed.scheduler import logger
    import socket
elif not on_zaratan and not use_gpu:
    from dask.distributed import LocalCluster
    
    
def overlap_loss_orb(params, r_bin, lnv2_bin, sparta_labels_bin):
    m, b = params
    line_classification = (lnv2_bin <= (m * r_bin + b)).astype(int)
    return -np.sum(line_classification == sparta_labels_bin)  # Negative for maximization

def overlap_loss_inf(params, r_bin, lnv2_bin, sparta_labels_bin):
    m, b = params
    line_classification = (lnv2_bin >= (m * r_bin + b)).astype(int)
    return -np.sum(line_classification == sparta_labels_bin)  # Negative for maximization

def overlap_loss(params, r_bin, lnv2_bin, sparta_labels_bin):
    m, b = params
    decision_boundary = m * r_bin + b
    line_classif = (lnv2_bin >= decision_boundary).astype(int)  # Orbiting = 1, Infalling = 0
    correct_classif = np.sum(line_classif == sparta_labels_bin)
    return -correct_classif  # Negative for maximization

def opt_func(bins, r, lnv2, sparta_labels, def_m, def_b, orb = True):
    # Assign bin indices based on radius
    bin_indices = np.digitize(r, bins) - 1  
    
    slopes = []
    intercepts = []

    for i in range(bins.shape[0]-1):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            slopes.append(def_m)
            intercepts.append(def_b)
            continue  # Skip empty bins
        
        r_bin = r[mask]
        lnv2_bin = lnv2[mask]
        sparta_labels_bin = sparta_labels[mask]

        # Optimize
        if orb:
            # Initial guess for m and b
            initial_guess = [def_m, np.max(lnv2_bin)]
            result = minimize(overlap_loss_orb, initial_guess, args=(r_bin, lnv2_bin, sparta_labels_bin))
        if orb is False:
            # Initial guess for m and b
            initial_guess = [def_m, np.mean(lnv2_bin)]
            result = minimize(overlap_loss_inf, initial_guess, args=(r_bin, lnv2_bin, sparta_labels_bin))
        else:
            initial_guess = [def_m, np.mean(lnv2_bin)]
            result = minimize(overlap_loss, initial_guess, args=(r_bin, lnv2_bin, sparta_labels_bin))
        slopes.append(result.x[0])
        intercepts.append(result.x[1])
        
    return {"m":slopes, "b":intercepts}
    
if __name__ == "__main__":
    if use_gpu:
        mp.set_start_method("spawn")

    if on_zaratan:            
        if use_gpu:
            initialize(local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        else:
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
            else:
                print("SLURM_CPUS_PER_TASK is not defined.")
            initialize(nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")

        print("Initialized")
        client = Client()
        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        login_node_address = "zvladimi@login.zaratan.umd.edu" # Change this to the address/domain of your login node

        logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
    else:
        if use_gpu:
            client = get_CUDA_cluster()
        else:
            tot_ncpus = mp.cpu_count()
            n_workers = int(np.floor(tot_ncpus / dask_task_cpus))
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=dask_task_cpus,
                memory_limit='5GB'  
            )
            client = Client(cluster)
    
    model_comb_name = get_combined_name(model_sims) 
    model_dir = model_type + "_" + model_comb_name + "nu" + nu_string 
    model_save_loc = path_to_models + model_comb_name + "/" + model_dir + "/"    
    
    curr_test_sims = test_sims[0]
    test_comb_name = get_combined_name(curr_test_sims) 
    dset_name = eval_datasets[0]
    plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
    create_directory(plot_loc)
    
    ps_param_dict = {
        "m_pos": -1.9973747688461672,
        "b_pos": 2.730691113802748,
        "m_neg": -1.601325049968688,
        "b_neg": 1.5101195108968333,
    }
    
    r, vr, lnv2, sparta_labels, my_data, halo_df = load_ps_data(client,test_sims[0])
    
    # r = r.to_numpy()
    # vr = vr.to_numpy()
    # lnv2 = lnv2.to_numpy()
    # sparta_labels = sparta_labels.to_numpy()
    
    r = my_data["p_Scaled_radii"].compute().to_numpy()
    vr = my_data["p_Radial_vel"].compute().to_numpy()
    vt = my_data["p_Tangential_vel"].compute().to_numpy()
    vphys = my_data["p_phys_vel"].compute().to_numpy()
    sparta_labels = my_data["Orbit_infall"].compute().to_numpy()
    lnv2 = np.log(vphys**2)
    
    c_r = my_data["c_Scaled_radii"].compute().to_numpy()
    c_vr = my_data["c_Radial_vel"].compute().to_numpy()
    c_vt = my_data["c_Tangential_vel"].compute().to_numpy()
    
    sparta_orb = np.where(sparta_labels == 1)[0]
    sparta_inf = np.where(sparta_labels == 0)[0]

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r < 1.75
    
    fltr_combs = {
    "orb_vr_neg": np.intersect1d(sparta_orb, np.where(mask_vr_neg)[0]),
    "orb_vr_pos": np.intersect1d(sparta_orb, np.where(mask_vr_pos)[0]),
    "inf_vr_neg": np.intersect1d(sparta_inf, np.where(mask_vr_neg)[0]),
    "inf_vr_pos": np.intersect1d(sparta_inf, np.where(mask_vr_pos)[0]),
    }

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
    
    act_mass_prf_all, act_mass_prf_orb,all_masses,bins = load_sparta_mass_prf(sim_splits,all_idxs,use_sims)
    act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb  
    
    vr_pos = opt_func(bins, r[mask_vr_pos], lnv2[mask_vr_pos], sparta_labels[mask_vr_pos], 0, ps_param_dict["b_pos"], orb = None)
    vr_neg = opt_func(bins, r[mask_vr_neg], lnv2[mask_vr_neg], sparta_labels[mask_vr_neg], 0, ps_param_dict["b_neg"], orb = None)

    
    # opt_param_dict = {
    #     "orb_vr_pos": opt_func(bins, r[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], sparta_labels[fltr_combs["orb_vr_pos"]], ps_param_dict["m_pos"], ps_param_dict["b_pos"], orb = True),
    #     "orb_vr_neg": opt_func(bins, r[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], sparta_labels[fltr_combs["orb_vr_neg"]], ps_param_dict["m_neg"], ps_param_dict["b_neg"], orb = True),
    #     "inf_vr_neg": opt_func(bins, r[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], sparta_labels[fltr_combs["inf_vr_neg"]], ps_param_dict["m_neg"], ps_param_dict["b_neg"], orb = False),
    #     "inf_vr_pos": opt_func(bins, r[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], sparta_labels[fltr_combs["inf_vr_pos"]], ps_param_dict["m_pos"], ps_param_dict["b_pos"], orb = False),
    # }
    
    opt_param_dict = {
        "orb_vr_pos": vr_pos,
        "orb_vr_neg": vr_neg,
        "inf_vr_neg": vr_neg,
        "inf_vr_pos": vr_pos,
    }
    
    width = 0.05
    perc = 0.99
    grad_lims = "0.2_0.5"
    r_cut = 1.75
    
    plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
    
    
    with timed("PS KE Dist plot"):
        
        x = np.linspace(0, 3, 1000)
        m_pos = ps_param_dict["m_pos"]
        b_pos = ps_param_dict["b_pos"]
        m_neg = ps_param_dict["m_neg"]
        b_neg = ps_param_dict["b_neg"]
        y12 = m_pos * x + b_pos
        y22 = m_neg * x + b_neg

        nbins = 200   
        
        x_range = (0, 3)
        y_range = (-2, 2.5)

        hist1, xedges, yedges = np.histogram2d(r[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], bins=nbins, range=(x_range, y_range))
        hist2, _, _ = np.histogram2d(r[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], bins=nbins, range=(x_range, y_range))
        hist3, _, _ = np.histogram2d(r[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], bins=nbins, range=(x_range, y_range))
        hist4, _, _ = np.histogram2d(r[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], bins=nbins, range=(x_range, y_range))

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
        plt.hist2d(r[mask_vr_pos], lnv2[mask_vr_pos], bins=nbins,
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="k",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label=r'$N$ (Counts)')
        plt_cust_ps_line(m = opt_param_dict["orb_vr_pos"]["m"], b = opt_param_dict["orb_vr_pos"]["b"], bins = bins)
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        

        plt.sca(axes[1])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        plt.hist2d(r[mask_vr_neg], lnv2[mask_vr_neg], bins=nbins,
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="k",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label=r'$N$ (Counts)')
        plt_cust_ps_line(m = opt_param_dict["orb_vr_neg"]["m"], b = opt_param_dict["orb_vr_neg"]["b"], bins = bins)
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        plt.sca(axes[2])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        h3 = plt.hist2d(r[mask_vr_pos], lnv2[mask_vr_pos], bins=nbins, norm="log",
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="k",
                label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(h3[3], label=r'$N$ (Counts)')
        plt_cust_ps_line(m = opt_param_dict["orb_vr_pos"]["m"], b = opt_param_dict["orb_vr_pos"]["b"], bins = bins)
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[3])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        h4 = plt.hist2d(r[mask_vr_neg], lnv2[mask_vr_neg], bins=nbins, norm="log",
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="k",
                label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(h4[3], label=r'$N$ (Counts)')
        plt_cust_ps_line(m = opt_param_dict["orb_vr_neg"]["m"], b = opt_param_dict["orb_vr_neg"]["b"], bins = bins)
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        mask_vrn = (vr < 0)
        mask_vrp = ~mask_vrn

        # Compute density and gradient.
        # For vr > 0
        hist_zp, hist_xp, hist_yp = np.histogram2d(r[mask_vrp], lnv2[mask_vrp], 
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
        hist_zn, hist_xn, hist_yn = np.histogram2d(r[mask_vrn], lnv2[mask_vrn],
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
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label="Smoothed Gradient Magnitude")
        plt.xlim(0, 2)
        
        # Plot the smoothed gradient
        plt.sca(axes[5])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        plt.contourf(hist_xn, hist_yn, hist_zn.T, levels=80, cmap='terrain')
        plt.vlines(x=r_cut,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label="Smoothed Gradient Magnitude")
        plt.xlim(0, 2)
        
        

        plt.tight_layout();
        plt.savefig(plot_loc + "bin_fit_perc_" + str(perc) + "_" + grad_lims + "_KE_dist_cut.png")
    
    
    plt_SPARTA_KE_dist(ps_param_dict, fltr_combs, bins, r, lnv2, perc = perc, width = width, r_cut = r_cut, plot_loc = plot_loc, title = "bin_fit_", cust_line_dict = opt_param_dict)
        