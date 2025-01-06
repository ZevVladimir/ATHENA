import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from scipy.optimize import curve_fit, minimize

plt.rcParams.update({"text.usetex":True, "font.family": "serif", "figure.dpi": 150})
cmap = get_cmap('terrain')
import os
import multiprocessing as mp
from dask.distributed import Client
import json
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import pickle
from colossus.cosmology import cosmology
from src.utils.calculation_functions import create_mass_prf, create_stack_mass_prf, filter_prf, calculate_density
from src.utils.ML_support import load_sprta_mass_prf
from src.utils.update_vis_fxns import compare_prfs_nu

    

from src.utils.ML_support import load_data, get_CUDA_cluster, get_combined_name, reform_dataset_dfs, parse_ranges, create_nu_string
from src.utils.data_and_loading_functions import create_directory, timed
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")

on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")

ML_dset_path = config["PATHS"]["ML_dset_path"]
path_to_models = config["PATHS"]["path_to_models"]

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

nu_splits = config["XGBOOST"]["nu_splits"]
nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

if on_zaratan:
    from dask_mpi import initialize
    from distributed.scheduler import logger
    import socket
elif not on_zaratan and not use_gpu:
    from dask.distributed import LocalCluster

####################################################################################################################################################################################################################################
    
def load_your_data():
    curr_test_sims = test_sims[0]
    test_comb_name = get_combined_name(curr_test_sims) 
        
    print("Testing on:", curr_test_sims)
    # Loop through and/or for Train/Test/All datasets and evaluate the model
    dset_name = eval_datasets[0]

    with timed("Model Evaluation on " + dset_name + " dataset"):             
        plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
        create_directory(plot_loc)
        
        # Load the halo information
        halo_files = []
        halo_dfs = []
        if dset_name == "Full":    
            for sim in curr_test_sims:
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + "Train" + "/halo_info/"))
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/"))
        else:
            for sim in curr_test_sims:
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + dset_name + "/halo_info/"))

        halo_df = pd.concat(halo_dfs)
        
        # Load the particle information
        data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)
        nptl = data.shape[0].compute()
        if nptl > 10000000:
            frac = 10_000_000 / nptl
            data = data.sample(frac=frac, random_state=42)
    r = data["p_Scaled_radii"]
    vr = data["p_Radial_vel"]
    vphys = data["p_phys_vel"]
    lnv2 = np.log(vphys**2)
    
    return r, vr, lnv2

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
        lnv2_mask = (1.0 < hist_lnv2) * (hist_lnv2 < 2.0)
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
    n_points: int = 20,
    perc: float = 0.9,
    width: float = 0.05,
    grad_lims: tuple = (0.2, 0.6),
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
    r, vr, lnv2 = load_your_data()

    # =================================================

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r < 2.0

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
    
    ####################################################################################################################################################################################################################################
    
    (m_pos, b_pos), (m_neg, b_neg) = calibrate_finder()
    # These are the values I used for my dataset. Feel free to play around with them
    # to improve results. 'perc' is the percent of particles expected below the line
    # for vr > 0. 'width' is used for 
    width = 0.05
    perc = 0.9

    r, vr, lnv2 = load_your_data()

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r < 2.0

    # y_shift = w / np.cos(np.arctan(m_neg))
    x = np.linspace(0, 3, 1000)
    y12 = m_pos * x + b_pos
    y22 = m_neg * x + b_neg

    x_range = (0, 3)
    y_range = (-2, 2.5)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes = axes.flatten()
    fig.suptitle(
        r"Kinetic energy distribution of particles around halos at $z=0$")

    for ax in axes:
        ax.set_xlabel(r'$r/R_{200}$')
        ax.set_ylabel(r'$\ln(v^2/v_{200}^2)$')
        ax.set_xlim(0, 3)
        ax.set_ylim(-2, 2.5)
        ax.text(0.25, -1.4, "Orbiting", fontsize=16, color="r",
                weight="bold", bbox=dict(facecolor='w', alpha=0.75))
        ax.text(1.5, 0.75, "Infalling", fontsize=16, color="b",
                weight="bold", bbox=dict(facecolor='w', alpha=0.75))

    plt.sca(axes[0])
    plt.title(r'$v_r > 0$')
    plt.hist2d(r[mask_vr_pos], lnv2[mask_vr_pos], bins=200,
                cmap="terrain", range=(x_range, y_range))
    plt.plot(x, y12, lw=2.0, color="k",
            label=fr"$m_p={m_pos:.3f}$"+"\n"+fr"$b_p={b_pos:.3f}$"+"\n"+fr"$p={perc:.3f}$")
    plt.colorbar(label=r'$N$ (Counts)')
    plt.legend()

    plt.sca(axes[1])
    plt.title(r'$v_r < 0$')
    plt.hist2d(r[mask_vr_neg], lnv2[mask_vr_neg], bins=200,
                cmap="terrain", range=(x_range, y_range))
    plt.plot(x, y22, lw=2.0, color="k",
            label=fr"$m_n={m_neg:.3f}$"+"\n"+fr"$b_n={b_neg:.3f}$"+"\n"+fr"$w={width:.3f}$")
    plt.colorbar(label=r'$N$ (Counts)')
    plt.legend()

    plt.tight_layout();
    plt.savefig(plot_loc + "KE_dist_cut.png")
        
    # Orbiting classification for vr > 0
    mask_cut_pos = (lnv2 < (m_pos * r + b_pos)) & (r < 3.0)

    # Orbiting classification for vr < 0
    mask_cut_neg = (lnv2 < (m_neg * r + b_neg)) & (r < 3.0)

    # Particle is infalling if it is below both lines and 2*R00
    mask_orb = \
    (mask_cut_pos & mask_vr_pos) ^ \
    (mask_cut_neg & mask_vr_neg)
    
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
    plt.hist2d(r[mask_orb], vr[mask_orb], bins=200, range=(x_range, y_range), cmap='terrain')
    plt.colorbar(label=r'$N$ (Counts)')

    plt.sca(axes[1])
    plt.title('Infalling')
    plt.hist2d(r[~mask_orb], vr[~mask_orb], bins=200, range=(x_range, y_range), cmap='terrain')
    plt.colorbar(label=r'$N$ (Counts)')

    plt.tight_layout()
    plt.savefig(plot_loc + "ps_dist.png")
    
    
####################################################################################################################################################################################################################################


    feature_columns = ["p_Scaled_radii","p_Radial_vel","p_Tangential_vel","c_Scaled_radii","c_Radial_vel","c_Tangential_vel"]
    target_column = ["Orbit_infall"]

    print("Testing on:", curr_test_sims)
    # Loop through and/or for Train/Test/All datasets and evaluate the model
    
    with timed("Model Evaluation on " + dset_name + " dataset"):                  
        # Load the halo information
        halo_files = []
        halo_dfs = []
        if dset_name == "Full":    
            for sim in curr_test_sims:
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + "Train" + "/halo_info/"))
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + "Test" + "/halo_info/"))
        else:
            for sim in curr_test_sims:
                halo_dfs.append(reform_dataset_dfs(ML_dset_path + sim + "/" + dset_name + "/halo_info/"))

        halo_df = pd.concat(halo_dfs)
        
        # Load the particle information
        data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)
    
    r = data["p_Scaled_radii"]
    vphys = data["p_phys_vel"]
    lnv2 = np.log(vphys**2)
    mask_cut_pos = (lnv2 < (m_pos * r + b_pos)) & (r < 3.0)

    # Orbiting classification for vr < 0
    mask_cut_neg = (lnv2 < (m_neg * r + b_neg)) & (r < 3.0)

    # Particle is infalling if it is below both lines and 2*R00
    mask_orb = \
    (mask_cut_pos & mask_vr_pos) ^ \
    (mask_cut_neg & mask_vr_neg)
    
    X = data[feature_columns]
    y = data[target_column]
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
        with open(ML_dset_path + sim + "/config.pickle", "rb") as file:
            config_dict = pickle.load(file)
            curr_z = config_dict["p_snap_info"]["red_shift"][()]
            all_z.append(curr_z)
            all_rhom.append(cosmol.rho_m(curr_z))
            h = config_dict["p_snap_info"]["h"][()]

    tot_num_halos = halo_n.shape[0]
    min_disp_halos = int(np.ceil(0.3 * tot_num_halos))

    # Get SPARTA's mass profiles
    act_mass_prf_all, act_mass_prf_orb,all_masses,bins = load_sprta_mass_prf(sim_splits,all_idxs,use_sims)
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
    lin_rticks = json.loads(config.get("XGBOOST","lin_rticks"))
    compare_prfs_nu(plt_nu_splits,len(cpy_plt_nu_splits),all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_loc,title="ps_cut_dens_")