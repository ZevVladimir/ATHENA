import numpy as np
import os
import pickle
import xgboost as xgb
from xgboost import dask as dxgb
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
from colossus import cosmology

from dask.distributed import Client
import multiprocessing as mp


from utils.data_and_loading_functions import timed, parse_ranges, create_nu_string, create_directory
from utils.calculation_functions import calculate_density
from utils.ML_support import get_CUDA_cluster, get_combined_name, reform_df, load_data, load_sprta_mass_prf
from sparta_tools import sparta # type: ignore
from colossus.cosmology import cosmology

from scipy.integrate import quad
from scipy.optimize import curve_fit

##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
rand_seed = config.getint("MISC","random_seed")
on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
sim_cosmol = config["MISC"]["sim_cosmol"]
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
    cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
else:
    cosmol = cosmology.setCosmology(sim_cosmol) 
    sim_pat = r"cbol_l(\d+)_n(\d+)"
match = re.search(sim_pat, curr_sparta_file)
if match:
    sparta_name = match.group(0)
path_to_hdf5_file = path_to_SPARTA_data + sparta_name + "/" + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
path_to_xgboost = config["PATHS"]["path_to_xgboost"]
model_sims = json.loads(config.get("XGBOOST","model_sims"))

global p_red_shift
p_red_shift = config.getfloat("SEARCH","p_red_shift")
search_rad = config.getfloat("SEARCH","search_rad")

file_lim = config.getint("XGBOOST","file_lim")
model_type = config["XGBOOST"]["model_type"]
train_rad = config.getint("XGBOOST","training_rad")
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))

reduce_rad = config.getfloat("XGBOOST","reduce_rad")
reduce_perc = config.getfloat("XGBOOST", "reduce_perc")

weight_rad = config.getfloat("XGBOOST","weight_rad")
min_weight = config.getfloat("XGBOOST","min_weight")
weight_exp = config.getfloat("XGBOOST","weight_exp")

hpo_loss = config.get("XGBOOST","hpo_loss")
nu_splits = config["XGBOOST"]["nu_splits"]
nu_splits = parse_ranges(nu_splits)
nu_string = create_nu_string(nu_splits)

plt_nu_splits = config["XGBOOST"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)
plt_nu_string = create_nu_string(plt_nu_splits)

linthrsh = config.getfloat("XGBOOST","linthrsh")
lin_nbin = config.getint("XGBOOST","lin_nbin")
log_nbin = config.getint("XGBOOST","log_nbin")
lin_rvticks = json.loads(config.get("XGBOOST","lin_rvticks"))
log_rvticks = json.loads(config.get("XGBOOST","log_rvticks"))
lin_tvticks = json.loads(config.get("XGBOOST","lin_tvticks"))
log_tvticks = json.loads(config.get("XGBOOST","log_tvticks"))
lin_rticks = json.loads(config.get("XGBOOST","lin_rticks"))
log_rticks = json.loads(config.get("XGBOOST","log_rticks"))

if use_gpu:
    from dask_cuda import LocalCUDACluster
    from cuml.metrics.accuracy import accuracy_score #TODO fix cupy installation??
    from sklearn.metrics import make_scorer
    import dask_ml.model_selection as dcv
    import cudf
    import dask_cudf as dc
elif not use_gpu and on_zaratan:
    from dask_mpi import initialize
    from mpi4py import MPI
    from distributed.scheduler import logger
    import socket
    #from dask_jobqueue import SLURMCluster
elif not on_zaratan:
    from dask_cuda import LocalCUDACluster


def rho_orb_dist(x: float, alpha: float, a: float) -> float:
    """Orbiting profile as a distribution.

    Parameters
    ----------
    x : float
        Radial points scaled by rh
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float

    """
    alpha *= x / (a + x)
    return np.power(x / a, -alpha) * np.exp(-0.5 * x ** 2)


def rho_orb_dens_dist(r: float, r_h: float, alpha: float, a: float) -> float:
    """Orbiting profile density distribution.

    Parameters
    ----------
    r : float
        Radial points
    r_h : float
        Halo radius
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float
        Normalized density distribution
    """
    distr = rho_orb_dist(r/r_h, alpha, a)
    distr /= 4. * np.pi * r_h ** 3 * \
        quad(lambda x, alpha, a: x**2 * rho_orb_dist(x, alpha, a),
             a=0, b=np.inf, args=(alpha, a))[0]
    return distr


def rho_orb_model_with_norm(r: float, log10A: float, r_h: float, alpha: float, a: float) -> float:
    """Orbiting density profile with free normalization constant.

    Parameters
    ----------
    r : float
        Radial points
    log10A : float
        Log 10 value of the normalization constant
    r_h : float
        Halo radius
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float
        Orbiting density profile
    """
    return np.power(10., log10A) * rho_orb_dist(x=r/r_h, alpha=alpha, a=a)


def rho_orb_model(r: float, morb: float, r_h: float, alpha: float, a: float) -> float:
    """Orbiting density profile imposing the constraint

                M_{\rm orb} = \int \rho_{\rm rob} dV

    Parameters
    ----------
    r : float
        Radial points in which to 
    morb : float
        Orbiting mass
    r_h : float
        Halo radius
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float
        Orbiting density profile
    """
    return morb * rho_orb_dens_dist(r=r, r_h=r_h, alpha=alpha, a=a)

def calc_a(a_p, a_s, M_orb, M_p):
    return a_p * (M_orb / M_p)**a_s

def calc_r_h(r_h_p, r_h_s, M_orb, M_p):
    return r_h_p * (M_orb / M_p)**r_h_s

def calc_alpha(alpha_p,alpha_s,M_orb,M_p):
    return alpha_p * (M_orb / M_p)**alpha_s

def cust_rho_orb_model(r: float, morb: float, M_p: float, r_h_p: float, r_h_s: float, alpha_p: float, alpha_s: float, a: float) -> float:
    """Orbiting density profile imposing the constraint

                M_{\rm orb} = \int \rho_{\rm rob} dV

    Parameters
    ----------
    r : float
        Radial points in which to 
    morb : float
        Orbiting mass
    r_h : float
        Halo radius
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float
        Orbiting density profile
    """
    r_h = calc_r_h(r_h_p,r_h_s,morb,M_p)
    alpha = calc_alpha(alpha_p,alpha_s,morb,M_p)
    print(r_h,alpha)
    return morb * rho_orb_dens_dist(r=r, r_h=r_h, alpha=alpha, a=a)

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
                
                halo_files = []
                halo_dfs = []
                if dset_name == "Full":    
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + "Train" + "/halo_info/"))
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + "Test" + "/halo_info/"))
                else:
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + dset_name + "/halo_info/"))

                halo_df = pd.concat(halo_dfs)
                
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)
                X_df = data[feature_columns]
                y_df = data[target_column]
                
        all_masses = []
        halo_first = halo_df["Halo_first"].values
        halo_n = halo_df["Halo_n"].values
        all_idxs = halo_df["Halo_indices"].values

        all_z = []
        all_rhom = []
        # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
        sim_splits = np.where(halo_first == 0)[0]

        # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
        # stacked simulations such that they correspond to the larger dataset and not one specific simulation
        if len(test_sims) > 1:
            for i,sim in enumerate(test_sims):
                # The first sim remains the same
                if i == 0:
                    continue
                # Else if it isn't the final sim 
                elif i < len(test_sims) - 1:
                    halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
                # Else if the final sim
                else:
                    halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
        
    # Get the redshifts for each simulation's primary snapshot
    for curr_test_sims in test_sims:
        for i,sim in enumerate(curr_test_sims):
            with open(path_to_calc_info + sim + "/config.pickle", "rb") as file:
                config_dict = pickle.load(file)
                curr_z = config_dict["p_snap_info"]["red_shift"][()]
                all_z.append(curr_z)
                all_rhom.append(cosmol.rho_m(curr_z))
                h = config_dict["p_snap_info"]["h"][()]
    
        tot_num_halos = halo_n.shape[0]
        
        act_mass_prf_all, act_mass_prf_orb,all_masses,bins,halos_r200m = load_sprta_mass_prf(sim_splits,all_idxs,curr_test_sims,ret_r200m=True)
        act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb

        # act_dens_prf_all = calculate_density(act_mass_prf_all,bins[1:],halos_r200m,sim_splits)
        act_dens_prf_orb = calculate_density(act_mass_prf_orb,bins[1:],halos_r200m,sim_splits)
        # act_dens_prf_inf = calculate_density(act_mass_prf_inf,bins[1:],halos_r200m,sim_splits)
               
        # Define fixed M_p value
        M_p = 1e14
        morb = np.median(act_mass_prf_orb[0,-1])
        print(morb)
        init_guess = [840.3, 0.226, 2.018, -0.05, 0.037] 

        # Fit function call with the fixed M_p passed within the lambda
        params_opt, params_cov = curve_fit(
            lambda r, r_h_p, r_h_s, alpha_p, alpha_s, a: cust_rho_orb_model(r, morb, M_p, r_h_p, r_h_s, alpha_p, alpha_s, a),
            bins[1:]*np.median(halos_r200m), np.median(act_dens_prf_orb,axis=0), p0=init_guess
        )

        # Extract optimized parameters
        r_h_p_opt, r_h_s_opt, alpha_p_opt, alpha_s_opt, a_opt = params_opt
        print("Optimal Parameters:")
        print("r_h_p:", r_h_p_opt, "r_h_s:", r_h_s_opt, "alpha_p:", alpha_p_opt, "alpha_s:", alpha_s_opt, "a:",a_opt)
        
        r_h_opt = calc_r_h(r_h_p_opt,r_h_s_opt,morb,M_p)
        alpha_opt = calc_alpha(alpha_p_opt,alpha_s_opt,morb,M_p)
        opt_fit_prf = rho_orb_model(bins[1:],morb,r_h_opt,alpha_opt,a_opt)
        
        r_h_init = calc_r_h(840.3,0.226,morb,M_p)
        alpha_init = calc_alpha(2.018,-0.05,morb,M_p)
        init_fit_prf = rho_orb_model(bins[1:],morb,r_h_init,alpha_init,0.037)
        
        fig,ax = plt.subplots(1)
        ax.plot(bins[1:],np.median(act_dens_prf_orb,axis=0),label="SPARTA")
        ax.plot(bins[1:],opt_fit_prf,label="Optimal fit")
        ax.plot(bins[1:],init_fit_prf,label="Initial fit")

        ax.set_yscale("log")
        ax.set_xlabel(r"$r/R_{200m}$")
        ax.set_ylabel(r"$\rho_{orb}$")
        ax.legend()
        fig.savefig("/home/zvladimi/MLOIS/Random_figs/test_prf_fit.png")

    client.close()