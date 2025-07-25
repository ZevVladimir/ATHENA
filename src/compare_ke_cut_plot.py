import numpy as np
import os
import pickle
import argparse 

from src.utils.ML_fxns import get_combined_name, extract_snaps, get_feature_labels
from src.utils.util_fxns import create_directory, load_pickle, load_config, load_pickle, timed, load_sparta_mass_prf,load_all_sim_cosmols,load_all_tdyn_steps
from src.utils.ke_cut_fxns import load_ke_data, fast_ke_predictor, opt_ke_predictor
from src.utils.calc_fxns import calc_rho
from src.utils.util_fxns import set_cosmology, parse_ranges
from src.utils.prfl_fxns import create_stack_mass_prf, filter_prf, compare_split_prfs_ke

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default=os.getcwd() + "/config.ini", 
    help='Path to config file (default: config.ini)'
)

args = parser.parse_args()
config_params = load_config(args.config)

ML_dset_path = config_params["PATHS"]["ml_dset_path"]
path_to_models = config_params["PATHS"]["path_to_models"]
SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]

features = config_params["TRAIN_MODEL"]["features"]

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

ke_test_dsets = config_params["KE_CUT"]["ke_test_dsets"]
fast_ke_calib_sims = config_params["KE_CUT"]["fast_ke_calib_sims"]
opt_ke_calib_sims = config_params["KE_CUT"]["opt_ke_calib_sims"]
r_cut_calib = config_params["KE_CUT"]["r_cut_calib"]
ke_test_sims = config_params["KE_CUT"]["ke_test_sims"]
    
    
if __name__ == "__main__":
    model_type = "kinetic_energy_cut"
    
    comb_fast_model_sims = get_combined_name(fast_ke_calib_sims) 
    comb_opt_model_sims = get_combined_name(opt_ke_calib_sims)   
      
    fast_model_fldr_loc = path_to_models + comb_fast_model_sims + "/" + model_type + "/"
    opt_model_fldr_loc = path_to_models + comb_opt_model_sims + "/" + model_type + "/" 
    
    for curr_test_sims in ke_test_sims:
        for dset_name in ke_test_dsets:
            all_sim_cosmol_list = load_all_sim_cosmols(curr_test_sims)
            all_tdyn_steps_list = load_all_tdyn_steps(curr_test_sims)
            
            test_comb_name = get_combined_name(curr_test_sims) 

            plot_loc = opt_model_fldr_loc + dset_name + "_" + test_comb_name + "/plots/"
            create_directory(plot_loc)
            
            
            feature_columns = get_feature_labels(features,all_tdyn_steps_list[0])
            snap_list = extract_snaps(curr_test_sims[0])
            
            if os.path.isfile(opt_model_fldr_loc + "ke_optparams_dict.pickle"):
                print("Loading parameters from saved file")
                opt_param_dict = load_pickle(opt_model_fldr_loc + "ke_optparams_dict.pickle")
                
                with timed("Loading Testing Data"):
                    r, vr, lnv2, sparta_labels, samp_data, my_data, halo_df = load_ke_data(curr_sims=curr_test_sims,sim_cosmol_list=all_sim_cosmol_list,snap_list=snap_list, dset_name=dset_name)
                    r_test = my_data["p_Scaled_radii"].values
                    vr_test = my_data["p_Radial_vel"].values
                    vphys_test = my_data["p_phys_vel"].values
                    sparta_labels_test = my_data["Orbit_infall"].values
                    lnv2_test = np.log(vphys_test**2)
                    
                    halo_first = halo_df["Halo_first"].values
                    halo_n = halo_df["Halo_n"].values
                    all_idxs = halo_df["Halo_indices"].values
                    # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
                    sim_splits = np.where(halo_first == 0)[0]
                
                    sparta_orb = np.where(sparta_labels_test == 1)[0]
                    sparta_inf = np.where(sparta_labels_test == 0)[0]
            else:
                raise FileNotFoundError(f"Expected to find optimized parameters at {os.path.join(opt_model_fldr_loc, 'ke_optparams_dict.pickle')}")

                    
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
                    with open(ML_dset_path + sim + "/dset_params.pickle", "rb") as file:
                        dset_params = pickle.load(file)
                        curr_z = dset_params["all_snap_info"]["prime_snap_info"]["red_shift"]
                        curr_rho_m = dset_params["all_snap_info"]["prime_snap_info"]["rho_m"]
                        all_z.append(curr_z)
                        all_rhom.append(curr_rho_m)
                        h = dset_params["all_snap_info"]["prime_snap_info"]["h"][()]

                tot_num_halos = halo_n.shape[0]
                min_disp_halos = int(np.ceil(0.3 * tot_num_halos))
                
                mask_vr_neg = (vr_test < 0)
                mask_vr_pos = ~mask_vr_neg

                ke_fastparam_dict = load_pickle(fast_model_fldr_loc + "ke_fastparams_dict.pickle")

                fast_mask_orb, preds_fast_ke = fast_ke_predictor(ke_fastparam_dict,r_test,vr_test,lnv2_test,r_cut_calib)
                
                preds_opt_ke = opt_ke_predictor(opt_param_dict, bins, r_test, vr_test, lnv2_test, r_cut_calib)
                
                opt_calc_mass_prf_all, opt_calc_mass_prf_orb, opt_calc_mass_prf_inf, opt_calc_nus, opt_calc_r200m = create_stack_mass_prf(sim_splits,radii=r_test, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds_opt_ke, prf_bins=bins, sim_cosmol_list=all_sim_cosmol_list, use_mp=True, all_z=all_z)
                fast_calc_mass_prf_all, fast_calc_mass_prf_orb, fast_calc_mass_prf_inf, fast_calc_nus, fast_calc_r200m = create_stack_mass_prf(sim_splits,radii=r_test, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds_fast_ke, prf_bins=bins, sim_cosmol_list=all_sim_cosmol_list, use_mp=True, all_z=all_z)

                # Halos that get returned with a nan R200m mean that they didn't meet the required number of ptls within R200m and so we need to filter them from our calculated profiles and SPARTA profiles 
                opt_small_halo_fltr = np.isnan(opt_calc_r200m)
                act_mass_prf_orb[opt_small_halo_fltr,:] = np.nan
                act_mass_prf_inf[opt_small_halo_fltr,:] = np.nan

                # Calculate the density by divide the mass of each bin by the volume of that bin's radius
                opt_calc_dens_prf_all = calc_rho(opt_calc_mass_prf_all*h,bins[1:],opt_calc_r200m*h,sim_splits,all_rhom)
                opt_calc_dens_prf_orb = calc_rho(opt_calc_mass_prf_orb*h,bins[1:],opt_calc_r200m*h,sim_splits,all_rhom)
                opt_calc_dens_prf_inf = calc_rho(opt_calc_mass_prf_inf*h,bins[1:],opt_calc_r200m*h,sim_splits,all_rhom)
                
                fast_calc_dens_prf_all = calc_rho(fast_calc_mass_prf_all*h,bins[1:],fast_calc_r200m*h,sim_splits,all_rhom)
                fast_calc_dens_prf_orb = calc_rho(fast_calc_mass_prf_orb*h,bins[1:],fast_calc_r200m*h,sim_splits,all_rhom)
                fast_calc_dens_prf_inf = calc_rho(fast_calc_mass_prf_inf*h,bins[1:],fast_calc_r200m*h,sim_splits,all_rhom)

                act_dens_prf_all = calc_rho(act_mass_prf_all*h,bins[1:],opt_calc_r200m*h,sim_splits,all_rhom)
                act_dens_prf_orb = calc_rho(act_mass_prf_orb*h,bins[1:],opt_calc_r200m*h,sim_splits,all_rhom)
                act_dens_prf_inf = calc_rho(act_mass_prf_inf*h,bins[1:],opt_calc_r200m*h,sim_splits,all_rhom)

                # If we want the density profiles to only consist of halos of a specific peak height (nu) bin 

                opt_orb_prf_lst = []
                opt_inf_prf_lst = []
                
                fast_orb_prf_lst = []
                fast_inf_prf_lst = []
                cpy_plt_nu_splits = plt_nu_splits.copy()
                
                for i,nu_split in enumerate(cpy_plt_nu_splits):
                    # Take the second element of the where to filter by the halos (?)
                    opt_fltr = np.where((opt_calc_nus > nu_split[0]) & (opt_calc_nus < nu_split[1]))[0]
                    fast_fltr = np.where((fast_calc_nus > nu_split[0]) & (fast_calc_nus < nu_split[1]))[0]
                    if opt_fltr.shape[0] > 25:
                        opt_orb_prf_lst.append(filter_prf(opt_calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,opt_fltr))
                        opt_inf_prf_lst.append(filter_prf(opt_calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,opt_fltr))
                    if fast_fltr.shape[0] > 25:
                        fast_orb_prf_lst.append(filter_prf(fast_calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fast_fltr))
                        fast_inf_prf_lst.append(filter_prf(fast_calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fast_fltr))
                    if opt_fltr.shape[0] < 25 and fast_fltr.shape[0] < 25:
                        plt_nu_splits.remove(nu_split)     

                compare_split_prfs_ke(plt_nu_splits,len(cpy_plt_nu_splits),opt_orb_prf_lst,opt_inf_prf_lst,fast_orb_prf_lst,fast_inf_prf_lst,bins[1:],lin_rticks,plot_loc)
