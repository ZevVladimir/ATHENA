import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex":True, "font.family": "serif", "figure.dpi": 150})
import os
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from colossus.cosmology import cosmology
import pickle

from utils.ML_support import setup_client, get_combined_name, parse_ranges, load_sparta_mass_prf, create_stack_mass_prf, get_model_name
from utils.data_and_loading_functions import create_directory, load_pickle, load_config, load_pickle, timed
from utils.ps_cut_support import load_ps_data
from utils.update_vis_fxns import compare_split_prfs_ps
from utils.calculation_functions import calculate_density, filter_prf

config_dict = load_config(os.getcwd() + "/config.ini")

ML_dset_path = config_dict["PATHS"]["ml_dset_path"]
path_to_models = config_dict["PATHS"]["path_to_models"]
SPARTA_output_path = config_dict["SPARTA_DATA"]["sparta_output_path"]

model_sims = config_dict["TRAIN_MODEL"]["model_sims"]
model_type = config_dict["TRAIN_MODEL"]["model_type"]
test_sims = config_dict["EVAL_MODEL"]["test_sims"]
eval_datasets = config_dict["EVAL_MODEL"]["eval_datasets"]

sim_cosmol = config_dict["MISC"]["sim_cosmol"]

plt_nu_splits = parse_ranges(config_dict["EVAL_MODEL"]["plt_nu_splits"])

plt_macc_splits = parse_ranges(config_dict["EVAL_MODEL"]["plt_macc_splits"])

linthrsh = config_dict["EVAL_MODEL"]["linthrsh"]
lin_nbin = config_dict["EVAL_MODEL"]["lin_nbin"]
log_nbin = config_dict["EVAL_MODEL"]["log_nbin"]
lin_rvticks = config_dict["EVAL_MODEL"]["lin_rvticks"]
log_rvticks = config_dict["EVAL_MODEL"]["log_rvticks"]
lin_tvticks = config_dict["EVAL_MODEL"]["lin_tvticks"]
log_tvticks = config_dict["EVAL_MODEL"]["log_tvticks"]
lin_rticks = config_dict["EVAL_MODEL"]["lin_rticks"]
log_rticks = config_dict["EVAL_MODEL"]["log_rticks"]
    
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
    cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
else:
    cosmol = cosmology.setCosmology(sim_cosmol) 
    sim_pat = r"cbol_l(\d+)_n(\d+)"    
    
    
if __name__ == "__main__":
    client = setup_client()
    
    comb_model_sims = get_combined_name(model_sims) 
        
    model_name = get_model_name(model_type, model_sims, hpo_done=config_dict["OPTIMIZE"]["hpo"], opt_param_dict=config_dict["OPTIMIZE"])    
    model_fldr_loc = path_to_models + comb_model_sims + "/" + model_type + "/"
    
    curr_test_sims = test_sims[0]
    test_comb_name = get_combined_name(curr_test_sims) 
    dset_name = eval_datasets[0]
    plot_loc = model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
    create_directory(plot_loc)
    
    if os.path.isfile(model_fldr_loc + "bin_fit_ps_cut_params.pickle"):
        print("Loading parameters from saved file")
        opt_param_dict = load_pickle(model_fldr_loc + "bin_fit_ps_cut_params.pickle")
        
        with timed("Loading Testing Data"):
            r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ps_data(client,curr_test_sims=curr_test_sims)
            r_test = my_data["p_Scaled_radii"].compute().to_numpy()
            vr_test = my_data["p_Radial_vel"].compute().to_numpy()
            vphys_test = my_data["p_phys_vel"].compute().to_numpy()
            sparta_labels_test = my_data["Orbit_infall"].compute().to_numpy()
            lnv2_test = np.log(vphys_test**2)
            
            halo_first = halo_df["Halo_first"].values
            halo_n = halo_df["Halo_n"].values
            all_idxs = halo_df["Halo_indices"].values
            # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
            sim_splits = np.where(halo_first == 0)[0]
        
            sparta_orb = np.where(sparta_labels_test == 1)[0]
            sparta_inf = np.where(sparta_labels_test == 0)[0]
    else:
        raise FileNotFoundError(f"Expected to find optimized parameters at {os.path.join(model_fldr_loc, 'bin_fit_ps_cut_params.pickle')}")

            
    act_mass_prf_all, act_mass_prf_orb, all_masses, bins = load_sparta_mass_prf(sim_splits,all_idxs,curr_test_sims)
    act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb 
    

#######################################################################################################################################    
    all_z = []
    all_rhom = []
    with timed("Density Profile Comparison"):
        # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
        # stacked simulations such that they correspond to the larger dataset and not one specific simulation
        if len(curr_test_sims) > 1:
            for i,sim in enumerate(curr_test_sims):
                # The first sim remains the same
                if i == 0:
                    continue
                # Else if it isn't the final sim 
                elif i < len(curr_test_sims) - 1:
                    halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
                # Else if the final sim
                else:
                    halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
        # Get the redshifts for each simulation's primary snapshot
        for i,sim in enumerate(curr_test_sims):
            with open(ML_dset_path + sim + "/config.pickle", "rb") as file:
                config_dict = pickle.load(file)
                curr_z = config_dict["p_snap_info"]["red_shift"][()]
                all_z.append(curr_z)
                all_rhom.append(cosmol.rho_m(curr_z))
                h = config_dict["p_snap_info"]["h"][()]

        tot_num_halos = halo_n.shape[0]
        min_disp_halos = int(np.ceil(0.3 * tot_num_halos))
        
        bin_indices = np.digitize(r_test, bins) - 1  
        preds_fit_ps = np.zeros(r_test.shape[0])
        
        mask_vr_neg = (vr_test < 0)
        mask_vr_pos = ~mask_vr_neg

        #TODO load this from pickle
        ps_param_dict = {
        "m_pos": -1.9973747688461672,
        "b_pos": 2.730691113802748,
        "m_neg": -1.601325049968688,
        "b_neg": 1.5101195108968333,
        }   
        
        mask_cut_pos = (lnv2_test < (ps_param_dict["m_pos"] * r_test + ps_param_dict["b_pos"])) & (r_test < 3.0)

        # Orbiting classification for vr < 0
        mask_cut_neg = (lnv2_test < (ps_param_dict["m_neg"] * r_test + ps_param_dict["b_neg"])) & (r_test < 3.0)

        # Particle is infalling if it is below both lines and 2*R00
        mask_orb = \
        (mask_cut_pos & mask_vr_pos) ^ \
        (mask_cut_neg & mask_vr_neg)
        
        preds_simp_ps = np.zeros(r_test.shape[0])
        preds_simp_ps[mask_orb] = 1
        
        for i in range(bins.shape[0]-1):
            if bins[i] <= 3.0:
                mask_pos = (bin_indices == i) & (vr_test > 0) & (lnv2_test <= opt_param_dict["inf_vr_pos"]["b"][i])
                mask_neg = (bin_indices == i) & (vr_test < 0) & (lnv2_test <= opt_param_dict["inf_vr_neg"]["b"][i])
            
                preds_fit_ps[mask_pos] = 1
                preds_fit_ps[mask_neg] = 1

        fit_calc_mass_prf_all, fit_calc_mass_prf_orb, fit_calc_mass_prf_inf, fit_calc_nus, fit_calc_r200m = create_stack_mass_prf(sim_splits,radii=r_test, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds_fit_ps, prf_bins=bins, use_mp=True, all_z=all_z)
        simp_calc_mass_prf_all, simp_calc_mass_prf_orb, simp_calc_mass_prf_inf, simp_calc_nus, simp_calc_r200m = create_stack_mass_prf(sim_splits,radii=r_test, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds_simp_ps, prf_bins=bins, use_mp=True, all_z=all_z)

        # Halos that get returned with a nan R200m mean that they didn't meet the required number of ptls within R200m and so we need to filter them from our calculated profiles and SPARTA profiles 
        fit_small_halo_fltr = np.isnan(fit_calc_r200m)
        act_mass_prf_orb[fit_small_halo_fltr,:] = np.nan
        act_mass_prf_inf[fit_small_halo_fltr,:] = np.nan

        # Calculate the density by divide the mass of each bin by the volume of that bin's radius
        fit_calc_dens_prf_all = calculate_density(fit_calc_mass_prf_all*h,bins[1:],fit_calc_r200m*h,sim_splits,all_rhom)
        fit_calc_dens_prf_orb = calculate_density(fit_calc_mass_prf_orb*h,bins[1:],fit_calc_r200m*h,sim_splits,all_rhom)
        fit_calc_dens_prf_inf = calculate_density(fit_calc_mass_prf_inf*h,bins[1:],fit_calc_r200m*h,sim_splits,all_rhom)
        
        simp_calc_dens_prf_all = calculate_density(simp_calc_mass_prf_all*h,bins[1:],simp_calc_r200m*h,sim_splits,all_rhom)
        simp_calc_dens_prf_orb = calculate_density(simp_calc_mass_prf_orb*h,bins[1:],simp_calc_r200m*h,sim_splits,all_rhom)
        simp_calc_dens_prf_inf = calculate_density(simp_calc_mass_prf_inf*h,bins[1:],simp_calc_r200m*h,sim_splits,all_rhom)

        act_dens_prf_all = calculate_density(act_mass_prf_all*h,bins[1:],fit_calc_r200m*h,sim_splits,all_rhom)
        act_dens_prf_orb = calculate_density(act_mass_prf_orb*h,bins[1:],fit_calc_r200m*h,sim_splits,all_rhom)
        act_dens_prf_inf = calculate_density(act_mass_prf_inf*h,bins[1:],fit_calc_r200m*h,sim_splits,all_rhom)

        # If we want the density profiles to only consist of halos of a specific peak height (nu) bin 

        fit_orb_prf_lst = []
        fit_inf_prf_lst = []
        
        simp_orb_prf_lst = []
        simp_inf_prf_lst = []
        cpy_plt_nu_splits = plt_nu_splits.copy()
        
        for i,nu_split in enumerate(cpy_plt_nu_splits):
            # Take the second element of the where to filter by the halos (?)
            fit_fltr = np.where((fit_calc_nus > nu_split[0]) & (fit_calc_nus < nu_split[1]))[0]
            simp_fltr = np.where((simp_calc_nus > nu_split[0]) & (simp_calc_nus < nu_split[1]))[0]
            if fit_fltr.shape[0] > 25:
                fit_orb_prf_lst.append(filter_prf(fit_calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fit_fltr))
                fit_inf_prf_lst.append(filter_prf(fit_calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fit_fltr))
            if simp_fltr.shape[0] > 25:
                simp_orb_prf_lst.append(filter_prf(simp_calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,simp_fltr))
                simp_inf_prf_lst.append(filter_prf(simp_calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,simp_fltr))
            if fit_fltr.shape[0] < 25 and simp_fltr.shape[0] < 25:
                plt_nu_splits.remove(nu_split)        
        compare_split_prfs_ps(plt_nu_splits,len(cpy_plt_nu_splits),fit_orb_prf_lst,fit_inf_prf_lst,simp_orb_prf_lst,simp_inf_prf_lst,bins[1:],lin_rticks,plot_loc)
