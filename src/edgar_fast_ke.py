import numpy as np
from scipy.optimize import curve_fit, minimize
import os
import matplotlib.pyplot as plt
import argparse

from src.utils.ML_fxns import setup_client, get_combined_name, get_model_name, extract_snaps
from src.utils.ke_cut_fxns import load_ke_data, fast_ke_predictor
from src.utils.util_fxns import set_cosmology, create_directory, save_pickle, load_pickle, load_config, load_SPARTA_data, split_sparta_hdf5_name

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default=os.getcwd() + "/config_edgar.ini", 
    help='Path to config file (default: config.ini)'
)

args = parser.parse_args()
config_params = load_config(args.config)

path_to_models = config_params["PATHS"]["path_to_models"]
debug_plt_path = config_params["PATHS"]["debug_plt_path"]

SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]

sim_cosmol = config_params["MISC"]["sim_cosmol"]
save_intermediate_data = config_params["MISC"]["save_intermediate_data"]

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
    #Sets up Dask client
    client = setup_client()
    
    # Combine all the simulations into one name so that models can be differentiated
    comb_model_sims = get_combined_name(fast_ke_calib_sims) 
    
    # We load from any of the SPARTA files the bins (but should just be logarithmic bins so feel free to remove this and use your own binning)
    curr_sparta_file = fast_ke_calib_sims[0]
    sparta_name, sparta_search_name = split_sparta_hdf5_name(curr_sparta_file)
    curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + curr_sparta_file + ".hdf5"
    param_paths = [["config","anl_prf","r_bins_lin"]]
    sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, save_data=save_intermediate_data)
    bins = sparta_params[sparta_param_names[0]]
    bins = np.insert(bins, 0, 0)
    
    # Name the model and save where it should go
    model_type = "kinetic_energy_cut"
    model_name = get_model_name(model_type, fast_ke_calib_sims)    
    model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"  
    create_directory(model_fldr_loc)
    
    ####################################################################################################################################################################################################################################
    # Load in the pickle of the parameters otherwise do the fitting and save the parameters
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
    
    print("Calibration Params")
    print(m_pos,b_pos,m_neg,b_neg)
    
    for curr_test_sims in  ke_test_sims:
        test_comb_name = get_combined_name(curr_test_sims) 
        plot_loc = model_fldr_loc + test_comb_name + "/plots/"
        create_directory(plot_loc)
        
        ############################################################################################################################################################
        # I would suggest replacing the next few lines with your loading method
        # But you just need the radius, radial velocity, physical velocity as expected
        
        snap_list = extract_snaps(fast_ke_calib_sims[0])
        r_samp, vr_samp, lnv2_samp, sparta_labels, samp_data, my_data, halo_df = load_ke_data(client, curr_test_sims, sim_cosmol, snap_list)
        
        r_test = my_data["p_Scaled_radii"].compute().to_numpy()
        vr_test = my_data["p_Radial_vel"].compute().to_numpy()
        vphys_test = my_data["p_phys_vel"].compute().to_numpy()
        lnv2_test = np.log(vphys_test**2)
        ############################################################################################################################################################

        # Generic prediction function I wrote feel free to replace with yours since I don't think mine matches OASIS since I don't use bound radii and only R200m
        ############################################################################################################################################################
        fast_mask_orb, preds_fast_ke = fast_ke_predictor(ke_param_dict, r_test, vr_test, lnv2_test, r_cut_calib)
        ############################################################################################################################################################
             
        bin_indices = np.digitize(r_test, bins) - 1  # subtract 1 to make bins zero-indexed

        # Initialize counters
        num_bins = len(bins) - 1

        fast_ke_orbiting_counts = np.zeros(num_bins)
        fast_ke_infalling_counts = np.zeros(num_bins)
    
        # For each bin find how many orbiting and how many infalling particles are there according to the fast ke cut    
        for i in range(num_bins):
            in_bin = bin_indices == i
            fast_ke_orbiting_counts[i] = np.sum(preds_fast_ke[in_bin] == 1)
            fast_ke_infalling_counts[i] = np.sum(preds_fast_ke[in_bin] == 0)
            
        # Calculate the ratio for each bin
        with np.errstate(divide='ignore', invalid='ignore'):
            fast_ke_ratio = np.where(fast_ke_infalling_counts > 0, fast_ke_orbiting_counts / (fast_ke_infalling_counts + fast_ke_orbiting_counts), np.nan)

        # Plot the ratio
        fig, ax = plt.subplots(1, figsize=(10,5))
        ax.plot(bins[1:], fast_ke_ratio, label='Fast KE Cut Classification')
    
        legend_fntsize = 14
        axis_fntsize = 18
        tick_fntsize = 14
        ax.legend(fontsize=legend_fntsize, loc="upper right")
        ax.set_xlabel(r"$r/R_{\rm 200m}$",fontsize=axis_fntsize)
        ax.set_ylabel(r"$N_{\rm orb}/N_{\rm tot}$",fontsize=axis_fntsize)
        ax.set_xlim(0,3)
        ax.set_ylim(0,1)
        ax.tick_params(axis='both', labelsize=tick_fntsize, length=6,width=2, direction="in")
        
        fig.savefig(debug_plt_path + test_comb_name + "_forb_by_rad.pdf",dpi=400)
        
            