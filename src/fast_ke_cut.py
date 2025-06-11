import numpy as np
from scipy.optimize import curve_fit, minimize
import os
import pandas as pd

from src.utils.ML_fxns import setup_client, get_combined_name, get_feature_labels, get_model_name, extract_snaps
from src.utils.ke_cut_fxns import load_ke_data, fast_ke_predictor
from src.utils.prfl_fxns import paper_dens_prf_plt
from src.utils.util_fxns import set_cosmology, depair_np, parse_ranges, create_directory, timed, save_pickle, load_pickle, load_config, load_SPARTA_data, split_sparta_hdf5_name
from src.utils.vis_fxns import plt_SPARTA_KE_dist, plt_KE_dist_grad

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
    
    
        
    dset_params = load_pickle(ML_dset_path + fast_ke_calib_sims[0] + "/dset_params.pickle")
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
    
    for curr_test_sims in ke_test_sims:
        test_comb_name = get_combined_name(curr_test_sims) 
        dset_name = eval_datasets[0]
        plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
        create_directory(plot_loc)
        
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
        "mask_vr_pos": mask_vr_pos,
        "mask_vr_neg": mask_vr_neg,
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

        nbins = 200   
        
        x_range = (0, 3)
        y_range = (-2, 2.5)

        halo_first = halo_df["Halo_first"].values
        all_idxs = halo_df["Halo_indices"].values
        # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
        sim_splits = np.where(halo_first == 0)[0]
        
        curr_sparta_file = fast_ke_calib_sims[0]
        sparta_name, sparta_search_name = split_sparta_hdf5_name(curr_sparta_file)
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + curr_sparta_file + ".hdf5"
        param_paths = [["config","anl_prf","r_bins_lin"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, save_data=save_intermediate_data)
        bins = sparta_params[sparta_param_names[0]]

        plt_SPARTA_KE_dist(ke_param_dict, fltr_combs, bins, r_r200m, lnv2, perc, width, r_cut_calib, plot_loc, title="only_fast_", plot_lin_too=True)
        plt_KE_dist_grad(ke_param_dict, fltr_combs, r_r200m, vr, lnv2, nbins, x_range, y_range, r_cut_calib, plot_loc)      
        
    ####################################################################################################################################################################################################################################

        with timed("Density profile plot"):
            r_r200m = my_data["p_Scaled_radii"]
            vr = my_data["p_Radial_vel"]
            vphys = my_data["p_phys_vel"]
            lnv2 = np.log(vphys**2)
            
            mask_orb, ke_cut_preds = fast_ke_predictor(ke_param_dict,r_r200m.values.compute(),vr.values.compute(),lnv2.values.compute(),r_cut_pred)
            
            X = my_data[feature_columns]
            y = my_data[target_column]
            
            paper_dens_prf_plt(X, y, pd.DataFrame(ke_cut_preds), halo_df, curr_test_sims, sim_cosmol, split_scale_dict, plot_loc, split_by_nu=True, split_by_macc=True)

            