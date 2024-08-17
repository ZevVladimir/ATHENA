import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import pickle
mpl.use('agg')
from utils.calculation_functions import calculate_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report
from colossus.halo import mass_so
import matplotlib.colors as colors
import multiprocessing as mp
from itertools import repeat
from sparta_tools import sparta # type: ignore
import os
from contextlib import contextmanager
import h5py
from pairing import depair
from scipy.spatial import cKDTree
import sys
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_and_loading_functions import check_pickle_exist_gadget, create_directory, find_closest_z, load_or_pickle_ptl_data, timed
from utils.calculation_functions import calc_v200m, calculate_density, create_mass_prf

num_processes = mp.cpu_count()

##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
rand_seed = config.getint("MISC","random_seed")
on_zaratan = config.getboolean("MISC","on_zaratan")
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
sim_cosmol = config["MISC"]["sim_cosmol"]
if sim_cosmol == "planck13-nbody":
    sim_pat = r"cpla_l(\d+)_n(\d+)"
else:
    sim_pat = r"cbol_l(\d+)_n(\d+)"
match = re.search(sim_pat, curr_sparta_file)
if match:
    sparta_name = match.group(0)
snap_loc = path_to_snaps + sparta_name + "/"
path_to_hdf5_file = path_to_SPARTA_data + sparta_name + "/" + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
curr_chunk_size = config.getint("SEARCH","chunk_size")

if not on_zaratan:
    import ipyvolume as ipv

sys.path.insert(1, path_to_pygadgetreader)  
from pygadgetreader import readsnap, readheader # type: ignore

def split_into_bins(num_bins, radial_vel, scaled_radii, particle_radii, halo_r200_per_part, red_shift, hubble_constant, little_h):
    start_bin_val = 0.001
    finish_bin_val = np.max(scaled_radii)
    
    bins = np.logspace(np.log10(start_bin_val), np.log10(finish_bin_val), num_bins)
    
    bin_start = 0
    average_val_part = np.zeros((num_bins,2), dtype = np.float32)
    average_val_hubble = np.zeros((num_bins,2), dtype = np.float32)
    
    # For each bin
    for i in range(num_bins - 1):
        bin_end = bins[i]
        
        # Find which particle belong in that bin
        indices_in_bin = np.where((scaled_radii >= bin_start) & (scaled_radii < bin_end))[0]
 
        if indices_in_bin.size != 0:
            # Get all the scaled radii within this bin and average it
            use_scaled_particle_radii = scaled_radii[indices_in_bin]
            average_val_part[i, 0] = np.average(use_scaled_particle_radii)
            
            # Get all the radial velocities within this bin and average it
            use_vel_rad = radial_vel[indices_in_bin]
            average_val_part[i, 1] = np.average(use_vel_rad)
            
            # get all the radii within this bin
            hubble_radius = particle_radii[indices_in_bin]

            # Find the median value and then the median value for the corresponding R200m values
            median_hubble_radius = np.median(hubble_radius)
            median_hubble_r200 = np.median(halo_r200_per_part[indices_in_bin])
            median_scaled_hubble = median_hubble_radius/median_hubble_r200
            
            # Calculate the v200m value for the corresponding R200m value found
            average_val_hubble[i,0] = median_scaled_hubble
            corresponding_hubble_m200m = mass_so.R_to_M(median_hubble_r200, red_shift, "200c")
            average_val_hubble[i,1] = (median_hubble_radius * hubble_constant)/calc_v200m(corresponding_hubble_m200m, median_hubble_r200)
            
        bin_start = bin_end
    
    return average_val_part, average_val_hubble 

def comb_prf(prf, num_halo, dtype):
    if num_halo > 1:
        prf = np.stack(prf, axis=0)
    else:
        prf = np.asarray(prf)
        prf = np.reshape(prf, (1,prf.size))
    
    prf = prf.astype(dtype)

    return prf
    
def compare_density_prf(splits, radii, halo_first, halo_n, act_mass_prf_all, act_mass_prf_orb, mass, orbit_assn, prf_bins, title, save_location, use_mp = False, show_graph = False, save_graph = True):
    # Shape of profiles should be (num halo,num bins)
    # EX: 10 halos, 80 bins (10,80)

    with timed("Finished Density Profile Plot"):
        print("Starting Density Profile Plot")
        act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb
        tot_num_halos = halo_first.shape[0]
        if tot_num_halos > 5:
            min_disp_halos = int(np.ceil(0.3 * tot_num_halos))
        else:
            min_disp_halos = 0
        num_bins = prf_bins.size
        
        med_act_mass_prf_all = 0
        med_act_mass_prf_orb = 0
        med_act_mass_prf_inf = 0
        med_act_dens_prf_all = 0
        med_act_dens_prf_orb = 0
        med_act_dens_prf_inf = 0
        
        med_calc_mass_prf_all = 0
        med_calc_mass_prf_orb = 0
        med_calc_mass_prf_inf = 0
        med_calc_dens_prf_all = 0
        med_calc_dens_prf_orb = 0
        med_calc_dens_prf_inf = 0
        
        calc_mass_prf_orb_lst = []
        calc_mass_prf_inf_lst = []   
        calc_mass_prf_all_lst = []   
        calc_dens_prf_orb_lst = []   
        calc_dens_prf_inf_lst = []   
        calc_dens_prf_all_lst = []   

        # Get density profiles by dividing the mass profiles by the volume of each bin
        act_dens_prf_all = calculate_density(act_mass_prf_all, prf_bins[1:])
        act_dens_prf_orb = calculate_density(act_mass_prf_orb, prf_bins[1:])
        act_dens_prf_inf = calculate_density(act_mass_prf_inf, prf_bins[1:])

        for i in range(splits.size):
            if i < splits.size - 1:
                curr_num_halos = act_mass_prf_all[splits[i]:splits[i+1]].shape[0]
            else:
                curr_num_halos = act_mass_prf_all[splits[i]:].shape[0]

            # for each halo get the corresponding mass and density profile that the model predicts for it
            # Can do this using either multiprocessing or a for loop
            if use_mp:
                num_processes = mp.cpu_count()
                with mp.Pool(processes=num_processes) as p:
                    calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf = zip(*p.starmap(create_mass_prf, 
                                                zip((radii[halo_first[n]:halo_first[n]+halo_n[n]] for n in range(splits[i],splits[i]+curr_num_halos)),
                                                    (orbit_assn[halo_first[n]:halo_first[n]+halo_n[n]] for n in range(splits[i],splits[i]+curr_num_halos)),
                                                    repeat(prf_bins),repeat(mass[i])),
                                                chunksize=100))
                p.close()
                p.join()        

            else:
                calc_mass_prf_orb = []
                calc_mass_prf_inf = []
                calc_mass_prf_all = []
                calc_dens_prf_orb = []
                calc_dens_prf_inf = []
                calc_dens_prf_all = []
                
                for j in range(curr_num_halos):
                    halo_mass_prf_all, halo_mass_prf_orb, halo_mass_prf_inf = create_mass_prf(radii[halo_first[splits[i]+j]:halo_first[splits[i]+j]+halo_n[splits[i]+j]], orbit_assn[halo_first[splits[i]+j]:halo_first[splits[i]+j]+halo_n[splits[i]+j]], prf_bins,mass[i])
                    calc_mass_prf_orb.append(np.array(halo_mass_prf_orb))
                    calc_mass_prf_inf.append(np.array(halo_mass_prf_inf))
                    calc_mass_prf_all.append(np.array(halo_mass_prf_all))
                    calc_dens_prf_orb.append(calculate_density(np.array(calc_mass_prf_orb),prf_bins[1:]))
                    calc_dens_prf_inf.append(calculate_density(np.array(calc_mass_prf_inf),prf_bins[1:]))
                    calc_dens_prf_all.append(calculate_density(np.array(calc_mass_prf_all),prf_bins[1:]))
            
            
            # For each profile combine all halos for each bin
            # calc_mass_prf_xxx has shape (num_halo, num_bins)
            curr_calc_mass_prf_orb = comb_prf(calc_mass_prf_orb, curr_num_halos, np.float32)
            curr_calc_mass_prf_inf = comb_prf(calc_mass_prf_inf, curr_num_halos, np.float32)
            curr_calc_mass_prf_all = comb_prf(calc_mass_prf_all, curr_num_halos, np.float32)
            # Calculate the density by divide the mass of each bin by the volume of that bin's radius
            curr_calc_dens_prf_orb = calculate_density(curr_calc_mass_prf_orb, prf_bins[1:])
            curr_calc_dens_prf_inf = calculate_density(curr_calc_mass_prf_inf, prf_bins[1:])
            curr_calc_dens_prf_all = calculate_density(curr_calc_mass_prf_all, prf_bins[1:]) 

            med_calc_mass_prf_all += np.median(curr_calc_mass_prf_all,axis=0) 
            med_calc_mass_prf_orb += np.median(curr_calc_mass_prf_orb,axis=0) 
            med_calc_mass_prf_inf += np.median(curr_calc_mass_prf_inf,axis=0) 
            med_calc_dens_prf_all += np.median(curr_calc_dens_prf_all,axis=0) 
            med_calc_dens_prf_orb += np.median(curr_calc_dens_prf_orb,axis=0) 
            med_calc_dens_prf_inf += np.median(curr_calc_dens_prf_inf,axis=0)
            
            med_act_mass_prf_all += np.median(act_mass_prf_all[splits[i]:splits[i]+curr_num_halos], axis=0)
            med_act_mass_prf_orb += np.median(act_mass_prf_orb[splits[i]:splits[i]+curr_num_halos], axis=0)
            med_act_mass_prf_inf += np.median(act_mass_prf_inf[splits[i]:splits[i]+curr_num_halos], axis=0)
            med_act_dens_prf_all += np.median(act_dens_prf_all[splits[i]:splits[i]+curr_num_halos], axis=0)
            med_act_dens_prf_orb += np.median(act_dens_prf_orb[splits[i]:splits[i]+curr_num_halos], axis=0)
            med_act_dens_prf_inf += np.median(act_dens_prf_inf[splits[i]:splits[i]+curr_num_halos], axis=0)
            
            
            calc_mass_prf_orb_lst.append(curr_calc_mass_prf_orb)
            calc_mass_prf_inf_lst.append(curr_calc_mass_prf_inf)
            calc_mass_prf_all_lst.append(curr_calc_mass_prf_all)
            calc_dens_prf_orb_lst.append(curr_calc_dens_prf_orb)
            calc_dens_prf_inf_lst.append(curr_calc_dens_prf_inf)
            calc_dens_prf_all_lst.append(curr_calc_dens_prf_all)
            
        calc_mass_prf_orb = np.vstack(calc_mass_prf_orb_lst)
        calc_mass_prf_inf = np.vstack(calc_mass_prf_inf_lst)
        calc_mass_prf_all = np.vstack(calc_mass_prf_all_lst)
        calc_dens_prf_orb = np.vstack(calc_dens_prf_orb_lst)
        calc_dens_prf_inf = np.vstack(calc_dens_prf_inf_lst)
        calc_dens_prf_all = np.vstack(calc_dens_prf_all_lst)
            
        # for each bin checking how many halos have particles there
        # if there are less than half the total number of halos then just treat that bin as having 0
        # for i in range(calc_mass_prf_orb.shape[1]):
        #     if np.where(calc_mass_prf_orb[:,i] > 0)[0].shape[0] < min_disp_halos:
        #         calc_mass_prf_orb[:,i] = np.NaN
        #         act_mass_prf_orb[:,i] = np.NaN
        #         med_calc_mass_prf_orb[i] = np.NaN
        #         med_act_mass_prf_orb[i] = np.NaN
        #     if np.where(calc_mass_prf_inf[:,i] > 0)[0].shape[0] < min_disp_halos:
        #         calc_mass_prf_inf[:,i] = np.NaN
        #         act_mass_prf_inf[:,i] = np.NaN
        #         med_calc_mass_prf_inf[i] = np.NaN
        #         med_act_mass_prf_inf[i] = np.NaN
        #     if np.where(calc_mass_prf_all[:,i] > 0)[0].shape[0] < min_disp_halos:
        #         calc_mass_prf_all[:,i] = np.NaN
        #         act_mass_prf_all[:,i] = np.NaN
        #         med_calc_mass_prf_all[i] = np.NaN
        #         med_act_mass_prf_all[i] = np.NaN
        #     if np.where(calc_dens_prf_orb[:,i] > 0)[0].shape[0] < min_disp_halos:
        #         calc_dens_prf_orb[:,i] = np.NaN
        #         act_dens_prf_orb[:,i] = np.NaN
        #         med_calc_dens_prf_orb[i] = np.NaN
        #         med_act_dens_prf_orb[i] = np.NaN
        #     if np.where(calc_dens_prf_inf[:,i] > 0)[0].shape[0] < min_disp_halos:
        #         calc_dens_prf_inf[:,i] = np.NaN
        #         act_dens_prf_inf[:,i] = np.NaN
        #         med_calc_dens_prf_inf[i] = np.NaN
        #         med_act_dens_prf_inf[i] = np.NaN
        #     if np.where(calc_dens_prf_all[:,i] > 0)[0].shape[0] < min_disp_halos:
        #         calc_dens_prf_all[:,i] = np.NaN
        #         act_dens_prf_all[:,i] = np.NaN
        #         med_calc_dens_prf_all[i] = np.NaN
        #         med_act_dens_prf_all[i] = np.NaN
        
        # Get the ratio of the calculated profile with the actual profile
        with np.errstate(divide='ignore', invalid='ignore'):
            all_dens_ratio = np.divide(calc_dens_prf_all,act_dens_prf_all) - 1
            inf_dens_ratio = np.divide(calc_dens_prf_inf,act_dens_prf_inf) - 1
            orb_dens_ratio = np.divide(calc_dens_prf_orb,act_dens_prf_orb) - 1
        
        inf_dens_ratio[np.isinf(inf_dens_ratio)] = 0

        
        # Find the upper and lower bound for scatter for calculated profiles
        # Want shape to be (1,80)
        # upper_calc_mass_prf_orb = np.nanpercentile(calc_mass_prf_orb, q=84.1, axis=0)
        # lower_calc_mass_prf_orb = np.nanpercentile(calc_mass_prf_orb, q=15.9, axis=0)
        # upper_calc_mass_prf_inf = np.nanpercentile(calc_mass_prf_inf, q=84.1, axis=0)
        # lower_calc_mass_prf_inf = np.nanpercentile(calc_mass_prf_inf, q=15.9, axis=0)
        # upper_calc_mass_prf_all = np.nanpercentile(calc_mass_prf_all, q=84.1, axis=0)
        # lower_calc_mass_prf_all = np.nanpercentile(calc_mass_prf_all, q=15.9, axis=0)
        # upper_calc_dens_prf_orb = np.nanpercentile(calc_dens_prf_orb, q=84.1, axis=0)
        # lower_calc_dens_prf_orb = np.nanpercentile(calc_dens_prf_orb, q=15.9, axis=0)
        # upper_calc_dens_prf_inf = np.nanpercentile(calc_dens_prf_inf, q=84.1, axis=0)
        # lower_calc_dens_prf_inf = np.nanpercentile(calc_dens_prf_inf, q=15.9, axis=0)
        # upper_calc_dens_prf_all = np.nanpercentile(calc_dens_prf_all, q=84.1, axis=0)
        # lower_calc_dens_prf_all = np.nanpercentile(calc_dens_prf_all, q=15.9, axis=0)
        
        # Same for actual profiles    
        upper_orb_dens_ratio = np.percentile(orb_dens_ratio, q=84.1, axis=0)
        lower_orb_dens_ratio = np.percentile(orb_dens_ratio, q=15.9, axis=0)
        upper_inf_dens_ratio = np.percentile(inf_dens_ratio, q=84.1, axis=0)
        lower_inf_dens_ratio = np.percentile(inf_dens_ratio, q=15.9, axis=0)
        upper_all_dens_ratio = np.percentile(all_dens_ratio, q=84.1, axis=0)
        lower_all_dens_ratio = np.percentile(all_dens_ratio, q=15.9, axis=0)
        # print(calc_dens_prf_inf[40])
        # print(act_dens_prf_inf[40])
        # print(inf_dens_ratio[40])
        # print(lower_inf_dens_ratio)
        # Take the median value of the ratios
        med_all_ratio = np.median(all_dens_ratio, axis=0)
        med_inf_ratio = np.median(inf_dens_ratio, axis=0)
        med_orb_ratio = np.median(orb_dens_ratio, axis=0)


        middle_bins = (prf_bins[1:] + prf_bins[:-1]) / 2
        
        # fig, ax = plt.subplots(1,3,figsize=(15,30))
        # ax[0].scatter(middle_bins,diff_n_inf_ptls,c="g")
        # ax[1].scatter(middle_bins,diff_n_orb_ptls,c="b")
        # ax[2].scatter(middle_bins,np.round(diff_n_all_ptls,5),c="r")
        # ax[0].set_xscale("log")
        # ax[1].set_xscale("log")
        # ax[2].set_xscale("log")
        # fig.set_size_inches(50, 25)
        # fig.savefig(save_location + title + "num_ptls_wrng.png", bbox_inches='tight')

        fig, ax = plt.subplots(1,3, figsize=(15,30))
        titlefntsize=26
        axisfntsize=22
        tickfntsize=20
        legendfntsize=18
        fill_alpha = 0.2
        
        # Get rid of the jump from 0 to the first occupied bin by setting them to nan
        med_calc_mass_prf_all[med_calc_mass_prf_all == 0] = np.NaN
        med_calc_mass_prf_orb[med_calc_mass_prf_orb == 0] = np.NaN
        med_calc_mass_prf_inf[med_calc_mass_prf_inf == 0] = np.NaN
        med_calc_dens_prf_all[med_calc_dens_prf_all == 0] = np.NaN
        med_calc_dens_prf_orb[med_calc_dens_prf_orb == 0] = np.NaN
        med_calc_dens_prf_inf[med_calc_dens_prf_inf == 0] = np.NaN
        med_act_mass_prf_all[med_act_mass_prf_all == 0] = np.NaN
        med_act_mass_prf_orb[med_act_mass_prf_orb == 0] = np.NaN
        med_act_mass_prf_inf[med_act_mass_prf_inf == 0] = np.NaN
        med_act_dens_prf_all[med_act_dens_prf_all == 0] = np.NaN
        med_act_dens_prf_orb[med_act_dens_prf_orb == 0] = np.NaN
        med_act_dens_prf_inf[med_act_dens_prf_inf == 0] = np.NaN
        
        ax[0].plot(middle_bins, med_calc_mass_prf_all, 'r-', label = "ML mass profile all ptls")
        ax[0].plot(middle_bins, med_calc_mass_prf_orb, 'b-', label = "ML mass profile orb ptls")
        ax[0].plot(middle_bins, med_calc_mass_prf_inf, 'g-', label = "ML mass profile inf ptls")
        ax[0].plot(middle_bins, med_act_mass_prf_all, 'r--', label = "SPARTA mass profile all ptls")
        ax[0].plot(middle_bins, med_act_mass_prf_orb, 'b--', label = "SPARTA mass profile orb ptls")
        ax[0].plot(middle_bins, med_act_mass_prf_inf, 'g--', label = "SPARTA mass profile inf ptls")

        
        ax[0].set_title("ML Predicted vs Actual Mass Profile",fontsize=titlefntsize)
        ax[0].set_xlabel("Radius $r/R_{200m}$", fontsize=axisfntsize)
        ax[0].set_ylabel("Mass $M_\odot$", fontsize=axisfntsize)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_box_aspect(1)
        ax[0].tick_params(axis='both',which='both',labelsize=tickfntsize)
        ax[0].legend(fontsize=legendfntsize)
        
        ax[1].plot(middle_bins, med_calc_dens_prf_all, 'r-', label = "ML density profile all ptls")
        ax[1].plot(middle_bins, med_calc_dens_prf_orb, 'b-', label = "ML density profile orb ptls")
        ax[1].plot(middle_bins, med_calc_dens_prf_inf, 'g-', label = "ML density profile inf ptls")
        ax[1].plot(middle_bins, med_act_dens_prf_all, 'r--', label = "SPARTA density profile all ptls")
        ax[1].plot(middle_bins, med_act_dens_prf_orb, 'b--', label = "SPARTA density profile orb ptls")
        ax[1].plot(middle_bins, med_act_dens_prf_inf, 'g--', label = "SPARTA density profile inf ptls")

        ax[1].set_title("ML Predicted vs Actual Density Profile",fontsize=titlefntsize)
        ax[1].set_xlabel("Radius $r/R_{200m}$", fontsize=axisfntsize)
        ax[1].set_ylabel("Density $M_\odot/kpc^3$", fontsize=axisfntsize)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_box_aspect(1)
        ax[1].tick_params(axis='both',which='both',labelsize=tickfntsize)
        ax[1].tick_params(axis='both',which='both',labelsize=tickfntsize)
        ax[1].legend(fontsize=legendfntsize)
        
        ax[2].plot(middle_bins, med_all_ratio, 'r', label = "(ML density profile / SPARTA density profile all) - 1")
        ax[2].plot(middle_bins, med_orb_ratio, 'b', label = "(ML density profile / SPARTA density profile orb) - 1")
        ax[2].plot(middle_bins, med_inf_ratio, 'g', label = "(ML density profile / SPARTA density profile inf) - 1")
        
        if tot_num_halos > 5:
            ax[2].fill_between(middle_bins, lower_all_dens_ratio, upper_all_dens_ratio, color='r', alpha=fill_alpha)
            ax[2].fill_between(middle_bins, lower_inf_dens_ratio, upper_inf_dens_ratio, color='g', alpha=fill_alpha)
            ax[2].fill_between(middle_bins, lower_orb_dens_ratio, upper_orb_dens_ratio, color='b', alpha=fill_alpha)    
            
        ax[2].set_title("(ML Predicted / Actual Density Profile) - 1",fontsize=titlefntsize)
        ax[2].set_xlabel("Radius $r/R_{200m}$", fontsize=axisfntsize)
        ax[2].set_ylabel("(ML Dens Prf / Act Dens Prf) - 1", fontsize=axisfntsize)
        #ax[2].set_ylim(0,8)
        
        all_ticks = [0]
        if tot_num_halos > 5:
            all_ticks.append(np.round(np.nanmax(upper_orb_dens_ratio),4))
            all_ticks.append(np.round(np.nanmin(lower_orb_dens_ratio),4))
            all_ticks.append(np.round(np.nanmax(upper_inf_dens_ratio),4))
            all_ticks.append(np.round(np.nanmin(lower_inf_dens_ratio),4))

            # ax[2].vlines(middle_bins[np.where(med_orb_ratio > 0)[0][0]],ymin=np.min([all_ticks[2],all_ticks[4]]),ymax=np.max([all_ticks[1],all_ticks[3]]),colors="black",label="orb ratio > 0: " + str(np.round(middle_bins[np.where(med_orb_ratio > 0)[0][0]],3)) + "R200m")
        all_ticks.append(np.round(np.nanmax(med_orb_ratio),4))
        all_ticks.append(np.round(np.nanmin(med_orb_ratio),4))
        all_ticks.append(np.round(np.nanmax(med_inf_ratio),4))
        all_ticks.append(np.round(np.nanmin(med_inf_ratio),4))

        ax[2].set_xscale("log")
        # ax[2].set_yscale("symlog")
        ax[2].set_box_aspect(1)
        ax[2].tick_params(axis='both',which='both',labelsize=tickfntsize)
        # ax[2].set_yticks(all_ticks)
        ax[2].legend(fontsize=legendfntsize)    
        
        if save_graph:
            fig.set_size_inches(50, 25)
            fig.savefig(save_location + title + "dens_prfl_rat.png", bbox_inches='tight')
        if show_graph:
            plt.show()
        plt.close()

    return middle_bins

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
    ax.legend()
    
    return ax.plot(hubble_vel[:,0], hubble_vel[:,1], color = "purple", alpha = 0.5, linestyle = "dashed", label = r"Hubble Flow")

def histogram(x,y,n_bins,range,min_ptl,set_ptl,use_bins=None,split_yscale_dict=None):
    if use_bins != None:
        x_bins = use_bins[0]
        y_bins = use_bins[1]
    else:
        x_bins = n_bins
        y_bins = n_bins
        
    if split_yscale_dict != None:
        linthrsh = split_yscale_dict["linthrsh"]
        lin_nbin = split_yscale_dict["lin_nbin"]
        log_nbin = split_yscale_dict["log_nbin"]
        
        y_range = range[1]
        # if the y axis goes to the negatives
        if y_range[0] < 0:
            lin_bins = np.linspace(-linthrsh,linthrsh,lin_nbin,endpoint=False)
            neg_log_bins = -np.logspace(np.log10(-y_range[0]),np.log10(linthrsh),log_nbin,endpoint=False)
            pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(y_range[1]),log_nbin)
            y_bins = np.concatenate([neg_log_bins,lin_bins,pos_log_bins])
            
        else:
            lin_bins = np.linspace(y_range[0],linthrsh,lin_nbin,endpoint=False)
            pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(y_range[1]),log_nbin)
            y_bins = np.concatenate([lin_bins,pos_log_bins])
    
    hist = np.histogram2d(x, y, bins=[x_bins,y_bins], range=range)
    hist[0][hist[0] < min_ptl] = set_ptl
    
    return hist
  
def split_orb_inf(data, labels):
    infall = data[np.where(labels == 0)[0]]
    orbit = data[np.where(labels == 1)[0]]
    return infall, orbit
 
def phase_plot(ax, x, y, min_ptl, max_ptl, range, num_bins, cmap, x_label="", y_label="", norm = "log", xrange=None, yrange=None, split_yscale_dict = None, hide_xticks=False, hide_yticks=False,text="", axisfontsize=20, title=""):
    bins = [num_bins,num_bins]
    if split_yscale_dict != None:
        linthrsh = split_yscale_dict["linthrsh"]
        lin_nbin = split_yscale_dict["lin_nbin"]
        log_nbin = split_yscale_dict["log_nbin"]
        
        x_range = range[0]
        y_range = range[1]
        use_yrange = y_range
        # if the y axis goes to the negatives
        if y_range[0] < 0:
            # keep the plot symmetric around 0
            if y_range[0] > y_range[1]:
                use_yrange[1] = -y_range[0]
            elif y_range[1] > y_range[0]:
                use_yrange[0] = -y_range[1]
            
            lin_bins = np.linspace(-linthrsh,linthrsh,lin_nbin,endpoint=False)
            neg_log_bins = -np.logspace(np.log10(-use_yrange[0]),np.log10(linthrsh),log_nbin,endpoint=False)
            pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(use_yrange[1]),log_nbin)
            y_bins = np.concatenate([neg_log_bins,lin_bins,pos_log_bins])            
            
            bins = [num_bins,y_bins]
            
            ax.hist2d(x[np.where((y >= -linthrsh) & (y <= linthrsh))], y[np.where((y >= -linthrsh) & (y <= linthrsh))], bins=bins, density=False, weights=None, cmin=min_ptl, cmap=cmap, norm=norm, vmin=min_ptl, vmax=max_ptl)
            ax.set_yscale('linear')
            ax.set_ylim(-linthrsh,linthrsh)
            ax.set_xlim(x_range[0],x_range[1])
            ax.spines[["bottom","top"]].set_visible(False)
            ax.get_xaxis().set_visible(False)
            
            divider = make_axes_locatable(ax)
            axposlog = divider.append_axes("top", size="100%", pad=0, sharex=ax)
            axposlog.hist2d(x[np.where(y >= linthrsh)[0]], y[np.where(y >= linthrsh)[0]], bins=bins, density=False, weights=None, cmin=min_ptl, cmap=cmap, norm=norm, vmin=min_ptl, vmax=max_ptl)
            axposlog.set_yscale('symlog',linthresh=linthrsh)
            axposlog.set_ylim((linthrsh,use_yrange[1]))
            axposlog.set_xlim(x_range[0],x_range[1])
            axposlog.spines[["bottom"]].set_visible(False)
            axposlog.get_xaxis().set_visible(False)
            
            axneglog = divider.append_axes("bottom", size="100%", pad=0, sharex=ax)            
            axneglog.hist2d(x[np.where(y < -linthrsh)[0]], y[np.where(y < -linthrsh)[0]], bins=bins, density=False, weights=None, cmin=min_ptl, cmap=cmap, norm=norm, vmin=min_ptl, vmax=max_ptl)
            axneglog.set_yscale('symlog',linthresh=linthrsh)
            axneglog.set_ylim((use_yrange[0],-linthrsh))
            axneglog.set_xlim(x_range[0],x_range[1])
            axneglog.spines[["top"]].set_visible(False)
            
        else:
            lin_bins = np.linspace(use_yrange[0],linthrsh,lin_nbin,endpoint=False)
            pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(use_yrange[1]),log_nbin)
            y_bins = np.concatenate([lin_bins,pos_log_bins])
            
            bins = [num_bins,y_bins]
            
            ax.hist2d(x[np.where((y <= linthrsh))], y[np.where(y <= linthrsh)], bins=bins, density=False, weights=None, cmin=min_ptl, cmap=cmap, norm=norm, vmin=min_ptl, vmax=max_ptl)
            ax.set_yscale('linear')
            ax.set_ylim(0,linthrsh)
            ax.set_xlim(x_range[0],x_range[1])
            ax.spines[["top"]].set_visible(False)
            
            divider = make_axes_locatable(ax)
            axposlog = divider.append_axes("top", size="100%", pad=0, sharex=ax)
            axposlog.hist2d(x[np.where(y >= linthrsh)[0]], y[np.where(y >= linthrsh)[0]], bins=bins, density=False, weights=None, cmin=min_ptl, cmap=cmap, norm=norm, vmin=min_ptl, vmax=max_ptl)
            axposlog.set_yscale('symlog',linthresh=linthrsh)
            axposlog.set_ylim((linthrsh,use_yrange[1]))
            axposlog.set_xlim(x_range[0],x_range[1])
            axposlog.spines[["bottom"]].set_visible(False)
            axposlog.get_xaxis().set_visible(False)
    else:
        ax.hist2d(x, y, bins=bins, range=range, density=False, weights=None, cmin=min_ptl, cmap=cmap, norm=norm, vmin=min_ptl, vmax=max_ptl)
    
    if text != "":
        if split_yscale_dict != None and y_range[0] < 0:
            axneglog.text(.01,.03, text, ha="left", va="bottom", transform=axneglog.transAxes, fontsize=18, bbox={"facecolor":'white',"alpha":.9,})
        else:
            ax.text(.01,.03, text, ha="left", va="bottom", transform=ax.transAxes, fontsize=18, bbox={"facecolor":'white',"alpha":.9,})
    if title != "":
        if split_yscale_dict != None:
            axposlog.set_title(title,fontsize=24)
        else:
            ax.set_title(title,fontsize=24)
    if x_label != "":
        if split_yscale_dict != None and y_range[0] < 0:
            axneglog.set_xlabel(x_label,fontsize=axisfontsize)
        else:
            ax.set_xlabel(x_label,fontsize=axisfontsize)
    if y_label != "":
        ax.set_ylabel(y_label,fontsize=axisfontsize)
    if hide_xticks:
        if split_yscale_dict != None and y_range[0] < 0:
            ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
            axposlog.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
            axneglog.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
        elif split_yscale_dict != None and y_range[0] >= 0:
            ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
            axposlog.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
        else:
            ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False) 
    else:
        if split_yscale_dict != None and y_range[0] < 0:
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='x', which='minor', labelsize=14)
            axposlog.tick_params(axis='x', which='major', labelsize=16)
            axposlog.tick_params(axis='x', which='minor', labelsize=14)
            axneglog.tick_params(axis='x', which='major', labelsize=16)
            axneglog.tick_params(axis='x', which='minor', labelsize=14) 
        elif split_yscale_dict != None and y_range[0] >= 0:
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='x', which='minor', labelsize=14)
            axposlog.tick_params(axis='x', which='major', labelsize=16)
            axposlog.tick_params(axis='x', which='minor', labelsize=14)
        else:
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='x', which='minor', labelsize=14)

    if hide_yticks:
        if split_yscale_dict != None and y_range[0] < 0:
            ax.tick_params(axis='y', which='both',left=False,labelleft=False) 
            axposlog.tick_params(axis='y', which='both',left=False,labelleft=False) 
            axneglog.tick_params(axis='y', which='both',left=False,labelleft=False) 
        elif split_yscale_dict != None and y_range[0] >= 0:
            ax.tick_params(axis='y', which='both',left=False,labelleft=False) 
            axposlog.tick_params(axis='y', which='both',left=False,labelleft=False) 
        else:
            ax.tick_params(axis='y', which='both',left=False,labelleft=False) 
    else:
        if split_yscale_dict != None and y_range[0] < 0:
            ax.tick_params(axis='y', which='major', labelsize=16)
            ax.tick_params(axis='y', which='minor', labelsize=14)
            axposlog.tick_params(axis='y', which='major', labelsize=16)
            axposlog.tick_params(axis='y', which='minor', labelsize=14)
            axneglog.tick_params(axis='y', which='major', labelsize=16)
            axneglog.tick_params(axis='y', which='minor', labelsize=14) 
        elif split_yscale_dict != None and y_range[0] >= 0:
            ax.tick_params(axis='y', which='major', labelsize=16)
            ax.tick_params(axis='y', which='minor', labelsize=14)
            axposlog.tick_params(axis='y', which='major', labelsize=16)
            axposlog.tick_params(axis='y', which='minor', labelsize=14)
        else:
            ax.tick_params(axis='y', which='major', labelsize=16)
            ax.tick_params(axis='y', which='minor', labelsize=14)

    # if xrange != None:
    #     ax.set_xlim(xrange)
    # if yrange != None:
    #     ax.set_ylim(yrange)    
                  
def imshow_plot(ax, img, extent, x_label="", y_label="", text="", title="", split_yscale_dict = None, return_img=False, hide_xticks=False, hide_yticks=False, axisfontsize=20, kwargs={}):
    if split_yscale_dict != None:
        linthrsh = split_yscale_dict["linthrsh"]
        lin_nbin = split_yscale_dict["lin_nbin"]
        log_nbin = split_yscale_dict["log_nbin"]

        x_range = extent[:2]
        y_range = extent[2:]
        use_range = y_range
        # if the y axis goes to the negatives
        if y_range[0] < 0:
            # lin_bins = np.linspace(-linthrsh,linthrsh,lin_nbin,endpoint=False)
            # neg_log_bins = -np.logspace(np.log10(-y_range[0]),np.log10(linthrsh),log_nbin,endpoint=False)
            # pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(y_range[1]),log_nbin)
            # y_bins = np.concatenate([neg_log_bins,lin_bins,pos_log_bins])            
            
            # keep the plot symmetric around 0
            if y_range[0] > y_range[1]:
                use_range[1] = -y_range[0]
            elif y_range[1] > y_range[0]:
                use_range[0] = -y_range[1]
                
            ret_img=ax.imshow(img, interpolation="none", **kwargs)
            ax.set_yscale('linear')
            ax.set_ylim(-linthrsh,linthrsh)
            ax.set_xlim(x_range[0],x_range[1])
            ax.spines[["bottom","top"]].set_visible(False)
            ax.get_xaxis().set_visible(False)
            
            divider = make_axes_locatable(ax)
            axposlog = divider.append_axes("top", size="100%", pad=0, sharex=ax)
            axposlog.imshow(img, interpolation="none", **kwargs)
            axposlog.set_yscale('symlog',linthresh=linthrsh)
            axposlog.set_ylim((linthrsh,use_range[1]))
            axposlog.set_xlim(x_range[0],x_range[1])
            axposlog.spines[["bottom"]].set_visible(False)
            axposlog.get_xaxis().set_visible(False)
            
            axneglog = divider.append_axes("bottom", size="100%", pad=0, sharex=ax)            
            axneglog.imshow(img, interpolation="none", **kwargs)
            axneglog.set_yscale('symlog',linthresh=linthrsh)
            axneglog.set_ylim((use_range[0],-linthrsh))
            axneglog.set_xlim(x_range[0],x_range[1])
            axneglog.spines[["top"]].set_visible(False)
            
        else:
            # lin_bins = np.linspace(y_range[0],linthrsh,lin_nbin,endpoint=False)
            # pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(y_range[1]),log_nbin)
            # y_bins = np.concatenate([lin_bins,pos_log_bins])
            
            ret_img=ax.imshow(img, interpolation="none", **kwargs)
            ax.set_yscale('linear')
            ax.set_ylim(0,linthrsh)
            ax.set_xlim(x_range[0],x_range[1])
            ax.spines[["top"]].set_visible(False)
            
            divider = make_axes_locatable(ax)
            axposlog = divider.append_axes("top", size="100%", pad=0, sharex=ax)
            axposlog.imshow(img, interpolation="none", **kwargs)
            axposlog.set_yscale('symlog',linthresh=linthrsh)
            axposlog.set_ylim((linthrsh,use_range[1]))
            axposlog.set_xlim(x_range[0],x_range[1])
            axposlog.spines[["bottom"]].set_visible(False)
            axposlog.get_xaxis().set_visible(False)
    else:
        ret_img=ax.imshow(img, interpolation="none", extent = extent, **kwargs)
        
    if text != "":
        if split_yscale_dict != None and y_range[0] < 0:
            axneglog.text(.01,.03, text, ha="left", va="bottom", transform=axneglog.transAxes, fontsize=18, bbox={"facecolor":'white',"alpha":0.9,})
        else:
            ax.text(.01,.03, text, ha="left", va="bottom", transform=ax.transAxes, fontsize=18, bbox={"facecolor":'white',"alpha":0.9,})
        
    if title != "":
        if split_yscale_dict != None:
            axposlog.set_title(title,fontsize=24)
        else:
            ax.set_title(title,fontsize=24)
    if x_label != "":
        if split_yscale_dict != None and y_range[0] < 0:
            axneglog.set_xlabel(x_label,fontsize=axisfontsize)
        else:
            ax.set_xlabel(x_label,fontsize=axisfontsize)
    if y_label != "":
        ax.set_ylabel(y_label,fontsize=axisfontsize)
    if hide_xticks:
        if split_yscale_dict != None and y_range[0] < 0:
            ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
            axposlog.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
            axneglog.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
        elif split_yscale_dict != None and y_range[0] >= 0:
            ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
            axposlog.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
        else:
            ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
    else:
        if split_yscale_dict != None and y_range[0] < 0:
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='x', which='minor', labelsize=14)
            axposlog.tick_params(axis='x', which='major', labelsize=16)
            axposlog.tick_params(axis='x', which='minor', labelsize=14)
            axneglog.tick_params(axis='x', which='major', labelsize=16)
            axneglog.tick_params(axis='x', which='minor', labelsize=14) 
        elif split_yscale_dict != None and y_range[0] >= 0:
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='x', which='minor', labelsize=14)
            axposlog.tick_params(axis='x', which='major', labelsize=16)
            axposlog.tick_params(axis='x', which='minor', labelsize=14)
        else:
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='x', which='minor', labelsize=14)
         
    if hide_yticks:
        if split_yscale_dict != None and y_range[0] < 0:
            ax.tick_params(axis='y', which='both',left=False,labelleft=False)
            axposlog.tick_params(axis='y', which='both',left=False,labelleft=False)
            axneglog.tick_params(axis='y', which='both',left=False,labelleft=False)
        elif split_yscale_dict != None and y_range[0] >= 0:
            ax.tick_params(axis='y', which='both',left=False,labelleft=False)
            axposlog.tick_params(axis='y', which='both',left=False,labelleft=False)
        else:
            ax.tick_params(axis='y', which='both',left=False,labelleft=False)
    else:
        if split_yscale_dict != None and y_range[0] < 0:
            ax.tick_params(axis='y', which='major', labelsize=16)
            ax.tick_params(axis='y', which='minor', labelsize=14)
            axposlog.tick_params(axis='y', which='major', labelsize=16)
            axposlog.tick_params(axis='y', which='minor', labelsize=14)
            axneglog.tick_params(axis='y', which='major', labelsize=16)
            axneglog.tick_params(axis='y', which='minor', labelsize=14) 
        elif split_yscale_dict != None and y_range[0] >= 0:
            ax.tick_params(axis='y', which='major', labelsize=16)
            ax.tick_params(axis='y', which='minor', labelsize=14)
            axposlog.tick_params(axis='y', which='major', labelsize=16)
            axposlog.tick_params(axis='y', which='minor', labelsize=14)
        else:
            ax.tick_params(axis='y', which='major', labelsize=16)
            ax.tick_params(axis='y', which='minor', labelsize=14)
           
    if return_img:
        return ret_img

def update_miss_class(img, miss_class, act, miss_class_min, act_min):
    # Where there are no misclassified particles but there are actual particles set to 0
    img = np.where((miss_class < 1) & (act >= act_min), miss_class_min, img)
    # Where there are miss classified particles but they won't show up on the image, set them to the min
    img = np.where((miss_class >= 1) & (img < miss_class_min) & (act >= act_min), miss_class_min, img)
    return img.T

def create_hist_max_ptl(min_ptl, set_ptl, inf_r, orb_r, inf_rv, orb_rv, inf_tv, orb_tv, num_bins, r_range, rv_range, tv_range, split_yscale_dict = None, bin_r_rv = None, bin_r_tv = None, bin_rv_tv = None):
    if bin_r_rv == None:
        orb_r_rv = histogram(orb_r, orb_rv, n_bins=num_bins, range=[r_range,rv_range],min_ptl=min_ptl,set_ptl=set_ptl,split_yscale_dict=split_yscale_dict)
        orb_r_tv = histogram(orb_r, orb_tv, n_bins=num_bins, range=[r_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl,split_yscale_dict=split_yscale_dict)
        orb_rv_tv = histogram(orb_rv, orb_tv, n_bins=num_bins, range=[rv_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl,split_yscale_dict=split_yscale_dict)
        inf_r_rv = histogram(inf_r, inf_rv, n_bins=num_bins, range=[r_range,rv_range],min_ptl=min_ptl,set_ptl=set_ptl,split_yscale_dict=split_yscale_dict)
        inf_r_tv = histogram(inf_r, inf_tv, n_bins=num_bins, range=[r_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl,split_yscale_dict=split_yscale_dict)
        inf_rv_tv = histogram(inf_rv, inf_tv, n_bins=num_bins, range=[rv_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl,split_yscale_dict=split_yscale_dict)
    else:
        orb_r_rv = histogram(orb_r, orb_rv,n_bins=num_bins,range=[r_range,rv_range],min_ptl=min_ptl,set_ptl=set_ptl,use_bins=bin_r_rv,split_yscale_dict=split_yscale_dict)
        orb_r_tv = histogram(orb_r, orb_tv,n_bins=num_bins,range=[r_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl,use_bins=bin_r_tv,split_yscale_dict=split_yscale_dict)
        orb_rv_tv = histogram(orb_rv, orb_tv,n_bins=num_bins,range=[rv_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl,use_bins=bin_rv_tv,split_yscale_dict=split_yscale_dict)
        inf_r_rv = histogram(inf_r, inf_rv,n_bins=num_bins,range=[r_range,rv_range],min_ptl=min_ptl,set_ptl=set_ptl,use_bins=bin_r_rv,split_yscale_dict=split_yscale_dict)
        inf_r_tv = histogram(inf_r, inf_tv,n_bins=num_bins,range=[r_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl,use_bins=bin_r_tv,split_yscale_dict=split_yscale_dict)
        inf_rv_tv = histogram(inf_rv, inf_tv,n_bins=num_bins,range=[rv_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl,use_bins=bin_rv_tv,split_yscale_dict=split_yscale_dict)

    max_ptl = np.max(np.array([np.max(orb_r_rv[0]),np.max(orb_r_tv[0]),np.max(orb_rv_tv[0]),np.max(inf_r_rv[0]),np.max(inf_r_tv[0]),np.max(inf_rv_tv[0]),]))
    
    return max_ptl, orb_r_rv, orb_r_tv, orb_rv_tv, inf_r_rv, inf_r_tv, inf_rv_tv

def percent_error(pred, act):
    return ((pred - act)/act) * 100

def calc_misclassified(correct_labels, ml_labels, r, rv, tv, r_range, rv_range, tv_range, num_bins, split_yscale_dict, model_info,dataset_name): 
    min_ptl = 1e-4
    act_min_ptl = 10

    inc_inf = np.where((ml_labels == 1) & (correct_labels == 0))[0]
    num_orb = np.where(correct_labels == 1)[0].shape[0]
    inc_orb = np.where((ml_labels == 0) & (correct_labels == 1))[0]
    num_inf = np.where(correct_labels == 0)[0].shape[0]
    tot_num_inc = inc_orb.shape[0] + inc_inf.shape[0]
    tot_num_ptl = num_orb + num_inf

    misclass_dict = {
        "Total Num of Particles": tot_num_ptl,
        "Num Incorrect Infalling Particles": str(inc_inf.shape[0])+", "+str(np.round(((inc_inf.shape[0]/num_inf)*100),2))+"% of infalling ptls",
        "Num Incorrect Orbiting Particles": str(inc_orb.shape[0])+", "+str(np.round(((inc_orb.shape[0]/num_orb)*100),2))+"% of orbiting ptls",
        "Num Incorrect All Particles": str(tot_num_inc)+", "+str(np.round(((tot_num_inc/tot_num_ptl)*100),2))+"% of all ptls",
    }
    
    inc_orb_r = r[inc_orb]
    inc_inf_r = r[inc_inf]
    inc_orb_rv = rv[inc_orb]
    inc_inf_rv = rv[inc_inf]
    inc_orb_tv = tv[inc_orb]
    inc_inf_tv = tv[inc_inf]

    act_inf_r, act_orb_r = split_orb_inf(r, correct_labels)
    act_inf_rv, act_orb_rv = split_orb_inf(rv, correct_labels)
    act_inf_tv, act_orb_tv = split_orb_inf(tv, correct_labels)

    max_all_ptl, act_orb_r_rv, act_orb_r_tv, act_orb_rv_tv, act_inf_r_rv, act_inf_r_tv, act_inf_rv_tv = create_hist_max_ptl(act_min_ptl, 0, act_inf_r, act_orb_r, act_inf_rv, act_orb_rv, act_inf_tv, act_orb_tv, num_bins, r_range, rv_range, tv_range, split_yscale_dict=split_yscale_dict)    
    max_ptl, inc_orb_r_rv, inc_orb_r_tv, inc_orb_rv_tv, inc_inf_r_rv, inc_inf_r_tv, inc_inf_rv_tv = create_hist_max_ptl(min_ptl, min_ptl, inc_inf_r, inc_orb_r, inc_inf_rv, inc_orb_rv, inc_inf_tv, inc_orb_tv, num_bins, r_range, rv_range, tv_range, bin_r_rv=act_orb_r_rv[1:], bin_r_tv=act_orb_r_tv[1:],bin_rv_tv=act_orb_rv_tv[1:],split_yscale_dict=split_yscale_dict)

    all_inc_r_rv = (inc_orb_r_rv[0] + inc_inf_r_rv[0])
    all_inc_r_tv = (inc_orb_r_tv[0] + inc_inf_r_tv[0])
    all_inc_rv_tv = (inc_orb_rv_tv[0] + inc_inf_rv_tv[0])
    all_act_r_rv = (act_orb_r_rv[0] + act_inf_r_rv[0])
    all_act_r_tv = (act_orb_r_tv[0] + act_inf_r_tv[0])
    all_act_rv_tv = (act_orb_rv_tv[0] + act_inf_rv_tv[0])

    scaled_orb_r_rv = (np.divide(inc_orb_r_rv[0],act_orb_r_rv[0],out=np.zeros_like(inc_orb_r_rv[0]), where=act_orb_r_rv[0]!=0))
    scaled_orb_r_tv = (np.divide(inc_orb_r_tv[0],act_orb_r_tv[0],out=np.zeros_like(inc_orb_r_tv[0]), where=act_orb_r_tv[0]!=0))
    scaled_orb_rv_tv = (np.divide(inc_orb_rv_tv[0],act_orb_rv_tv[0],out=np.zeros_like(inc_orb_rv_tv[0]), where=act_orb_rv_tv[0]!=0))
    scaled_inf_r_rv = (np.divide(inc_inf_r_rv[0],act_inf_r_rv[0],out=np.zeros_like(inc_inf_r_rv[0]), where=act_inf_r_rv[0]!=0))
    scaled_inf_r_tv = (np.divide(inc_inf_r_tv[0],act_inf_r_tv[0],out=np.zeros_like(inc_inf_r_tv[0]), where=act_inf_r_tv[0]!=0))
    scaled_inf_rv_tv = (np.divide(inc_inf_rv_tv[0],act_inf_rv_tv[0],out=np.zeros_like(inc_inf_rv_tv[0]), where=act_inf_rv_tv[0]!=0))
    scaled_all_r_rv = (np.divide(all_inc_r_rv,all_act_r_rv,out=np.zeros_like(all_inc_r_rv), where=all_act_r_rv!=0))
    scaled_all_r_tv = (np.divide(all_inc_r_tv,all_act_r_tv,out=np.zeros_like(all_inc_r_tv), where=all_act_r_tv!=0))
    scaled_all_rv_tv = (np.divide(all_inc_rv_tv,all_act_rv_tv,out=np.zeros_like(all_inc_rv_tv), where=all_act_rv_tv!=0))

    # For any spots that have no missclassified particles but there are particles there set it to the minimum amount so it still shows up in the plot.
    scaled_orb_r_rv = update_miss_class(scaled_orb_r_rv, inc_orb_r_rv[0], act_orb_r_rv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_orb_r_tv = update_miss_class(scaled_orb_r_tv, inc_orb_r_tv[0], act_orb_r_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_orb_rv_tv = update_miss_class(scaled_orb_rv_tv, inc_orb_rv_tv[0], act_orb_rv_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_inf_r_rv = update_miss_class(scaled_inf_r_rv, inc_inf_r_rv[0], act_inf_r_rv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_inf_r_tv = update_miss_class(scaled_inf_r_tv, inc_inf_r_tv[0], act_inf_r_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_inf_rv_tv = update_miss_class(scaled_inf_rv_tv, inc_inf_rv_tv[0], act_inf_rv_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_all_r_rv = update_miss_class(scaled_all_r_rv, all_inc_r_rv, all_act_r_rv, miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_all_r_tv = update_miss_class(scaled_all_r_tv, all_inc_r_tv, all_act_r_tv, miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_all_rv_tv = update_miss_class(scaled_all_rv_tv, all_inc_rv_tv, all_act_rv_tv, miss_class_min=min_ptl, act_min=act_min_ptl)
    
    all_inc_inf_r_rv = update_miss_class(inc_inf_r_rv[0], inc_inf_r_rv[0], act_inf_r_rv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    all_inc_inf_r_tv = update_miss_class(inc_inf_r_tv[0], inc_inf_r_tv[0], act_inf_r_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    all_inc_inf_rv_tv = update_miss_class(inc_inf_rv_tv[0], inc_inf_rv_tv[0], act_inf_rv_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    all_inc_orb_r_rv = update_miss_class(inc_orb_r_rv[0], inc_orb_r_rv[0], act_orb_r_rv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    all_inc_orb_r_tv = update_miss_class(inc_orb_r_tv[0], inc_orb_r_tv[0], act_orb_r_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    all_inc_orb_rv_tv = update_miss_class(inc_orb_rv_tv[0], inc_orb_rv_tv[0], act_orb_rv_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    
    max_diff = np.max(np.array([np.max(scaled_orb_r_rv),np.max(scaled_orb_r_tv),np.max(scaled_orb_rv_tv),
                                np.max(scaled_inf_r_rv),np.max(scaled_inf_r_tv),np.max(scaled_inf_rv_tv),
                                np.max(scaled_all_r_rv),np.max(scaled_all_r_tv),np.max(scaled_all_rv_tv)]))
    
    return misclass_dict, min_ptl, max_diff, max_all_ptl, all_inc_r_rv, all_inc_r_tv, all_inc_rv_tv, scaled_inf_r_rv, scaled_inf_r_tv, scaled_inf_rv_tv, scaled_orb_r_rv, scaled_orb_r_tv, scaled_orb_rv_tv, scaled_all_r_rv, scaled_all_r_tv, scaled_all_rv_tv, all_inc_inf_r_rv,all_inc_inf_r_tv,all_inc_inf_rv_tv,all_inc_orb_r_rv,all_inc_orb_r_tv,all_inc_orb_rv_tv
    
def plot_misclassified(p_corr_labels, p_ml_labels, p_r, p_rv, p_tv, c_r, c_rv, c_tv, title, num_bins,save_location,model_info,dataset_name):
    if "Results" not in model_info:
        model_info["Results"] = {}
    
    with timed("Finished Misclassified Particle Plot"):
        print("Starting Misclassified Particle Plot")

        max_r = np.max(p_r)
        max_rv = np.max(p_rv)
        min_rv = np.min(p_rv)
        max_tv = np.max(p_tv)
        min_tv = np.min(p_tv)
        
        c_max_r = np.nanmax(c_r)
        c_max_rv = np.nanmax(c_rv)
        c_min_rv = np.nanmin(c_rv)
        c_max_tv = np.nanmax(c_tv)
        c_min_tv = np.nanmin(c_tv)
        
        r_range = [0, max_r]
        rv_range = [min_rv, max_rv]
        tv_range = [min_tv, max_tv]

        c_corr_labels = np.copy(p_corr_labels)
        c_ml_labels = np.copy(p_ml_labels)

        c_corr_labels[np.argwhere(np.isnan(c_r)).flatten()] = -99
        c_ml_labels[np.argwhere(np.isnan(c_r)).flatten()] = -99
        
        split_yscale_dict = {
            "linthrsh":0.05, 
            "lin_nbin":30,
            "log_nbin":15,
        }   
        
        print("Primary Snap Misclassification")
        p_misclass_dict, p_min_ptl, p_max_diff, p_max_all_ptl, p_all_inc_r_rv, p_all_inc_r_tv, p_all_inc_rv_tv, p_scaled_inf_r_rv, p_scaled_inf_r_tv, p_scaled_inf_rv_tv, p_scaled_orb_r_rv, p_scaled_orb_r_tv, p_scaled_orb_rv_tv, p_scaled_all_r_rv, p_scaled_all_r_tv, p_scaled_all_rv_tv,p_all_inc_inf_r_rv,p_all_inc_inf_r_tv,p_all_inc_inf_rv_tv,p_all_inc_orb_r_rv,p_all_inc_orb_r_tv,p_all_inc_orb_rv_tv = calc_misclassified(p_corr_labels, p_ml_labels, p_r, p_rv, p_tv, r_range, rv_range, tv_range, num_bins=num_bins, split_yscale_dict=split_yscale_dict,model_info=model_info,dataset_name=dataset_name)
        print("Secondary Snap Misclassification")
        c_misclass_dict, c_min_ptl, c_max_diff, c_max_all_ptl, c_all_inc_r_rv, c_all_inc_r_tv, c_all_inc_rv_tv, c_scaled_inf_r_rv, c_scaled_inf_r_tv, c_scaled_inf_rv_tv, c_scaled_orb_r_rv, c_scaled_orb_r_tv, c_scaled_orb_rv_tv, c_scaled_all_r_rv, c_scaled_all_r_tv, c_scaled_all_rv_tv,c_all_inc_inf_r_rv,c_all_inc_inf_r_tv,c_all_inc_inf_rv_tv,c_all_inc_orb_r_rv,c_all_inc_orb_r_tv,c_all_inc_orb_rv_tv = calc_misclassified(c_corr_labels, c_ml_labels, c_r, c_rv, c_tv, r_range, rv_range, tv_range, num_bins=num_bins, split_yscale_dict=split_yscale_dict,model_info=model_info,dataset_name=dataset_name)
        
        if dataset_name not in model_info["Results"]:
            model_info["Results"][dataset_name]={}
        model_info["Results"][dataset_name]["Primary Snap"] = p_misclass_dict
        model_info["Results"][dataset_name]["Secondary Snap"] = c_misclass_dict
        
        cividis_cmap = plt.get_cmap("cividis_r")
        cividis_cmap.set_under(color='white')   
        magma_cmap = plt.get_cmap("magma_r")
        magma_cmap.set_under(color='white') 
        cmap = plt.get_cmap("magma")
        test_cmap = plt.get_cmap("viridis")
        
        scale_miss_class_args = {
            "vmin":p_min_ptl,
            "vmax":p_max_diff,
            "norm":"log",
            "origin":"lower",
            "aspect":"auto",
            "cmap":magma_cmap,
        }

        all_miss_class_args = {
            "vmin":1,
            "vmax":p_max_all_ptl,
            "norm":"log",
            "origin":"lower",
            "aspect":"auto",
            "cmap":cividis_cmap,
        }

        widths = [4,4,4,4,.5]
        heights = [0.05,4,4,4]
        
        ptl_distr_fig = plt.figure(constrained_layout=True, figsize=(25,25))
        gs = ptl_distr_fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
                
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[1,0]), c_all_inc_r_rv.T, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",hide_xticks=True,text="All Misclassified",kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[1,1]), p_all_inc_r_rv.T, extent=[0,max_r,min_rv,max_rv],hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="All Misclassified",kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[1,2]), p_all_inc_r_tv.T, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[1,3]), p_all_inc_rv_tv.T, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=all_miss_class_args)
        
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[2,0]), c_all_inc_inf_r_rv, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",hide_xticks=True,kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[2,1]), p_all_inc_inf_r_rv, extent=[0,max_r,min_rv,max_rv],hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[2,2]), p_all_inc_inf_r_tv, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[2,3]), p_all_inc_inf_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=all_miss_class_args)
        
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[3,0]), c_all_inc_orb_r_rv, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",hide_xticks=True,kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[3,1]), p_all_inc_orb_r_rv, extent=[0,max_r,min_rv,max_rv],hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[3,2]), p_all_inc_orb_r_tv, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=all_miss_class_args)
        # imshow_plot(scal_miss_class_fig.add_subplot(gs[3,3]), p_all_inc_orb_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=all_miss_class_args)
        
        inf_loc = np.where(p_corr_labels == 0)[0]
        orb_loc = np.where(p_corr_labels == 1)[0]
        
        phase_plot(ptl_distr_fig.add_subplot(gs[1,0]), p_r, p_rv, min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,rv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="All\nParticles")
        phase_plot(ptl_distr_fig.add_subplot(gs[1,1]), p_r, p_tv, min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,tv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,hide_xticks=True,y_label="$v_t/v_{200m}$")
        phase_plot(ptl_distr_fig.add_subplot(gs[1,2]), p_rv, p_tv, min_ptl=10, max_ptl=p_max_all_ptl, range=[rv_range,tv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,hide_xticks=True,hide_yticks=True)
        phase_plot(ptl_distr_fig.add_subplot(gs[1,3]), c_r, c_rv, min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,rv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,hide_xticks=True,y_label="$v_r/v_{200m}$", text="All\nParticles")
        
        phase_plot(ptl_distr_fig.add_subplot(gs[2,0]), p_r[inf_loc], p_rv[inf_loc], min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,rv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="Infalling\nParticles")
        phase_plot(ptl_distr_fig.add_subplot(gs[2,1]), p_r[inf_loc], p_tv[inf_loc], min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,tv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,hide_xticks=True,y_label="$v_t/v_{200m}$")
        phase_plot(ptl_distr_fig.add_subplot(gs[2,2]), p_rv[inf_loc], p_tv[inf_loc], min_ptl=10, max_ptl=p_max_all_ptl, range=[rv_range,tv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,hide_xticks=True,hide_yticks=True)
        phase_plot(ptl_distr_fig.add_subplot(gs[2,3]), c_r[inf_loc], c_rv[inf_loc], min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,rv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,hide_xticks=True,y_label="$v_r/v_{200m}$", text="Infalling\nParticles")

        phase_plot(ptl_distr_fig.add_subplot(gs[3,0]), p_r[orb_loc], p_rv[orb_loc], min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,rv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,x_label="$r/R_{200m}$",hide_yticks=False,y_label="$v_r/v_{200m}$",text="Orbiting\nParticles")
        phase_plot(ptl_distr_fig.add_subplot(gs[3,1]), p_r[orb_loc], p_tv[orb_loc], min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,tv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$")
        phase_plot(ptl_distr_fig.add_subplot(gs[3,2]), p_rv[orb_loc], p_tv[orb_loc], min_ptl=10, max_ptl=p_max_all_ptl, range=[rv_range,tv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,x_label="$v_r/v_{200m}$",hide_yticks=True)
        phase_plot(ptl_distr_fig.add_subplot(gs[3,3]), c_r[orb_loc], c_rv[orb_loc], min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,rv_range],num_bins=num_bins,cmap=cividis_cmap,split_yscale_dict=split_yscale_dict,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$", text="Orbiting\nParticles")

        
        phase_plt_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=1, vmax=p_max_all_ptl),cmap=cividis_cmap), cax=plt.subplot(gs[1:,-1]))
        phase_plt_color_bar.ax.tick_params(labelsize=14)
        phase_plt_color_bar.set_label("Number of Particles",fontsize=16)
        
        ptl_distr_fig.savefig(save_location + title + "ptl_distr.png")

        widths = [4,4,4,4,.5]
        heights = [0.12,4,4,4]
        
        scal_miss_class_fig = plt.figure(constrained_layout=True,figsize=(30,25))
        gs = scal_miss_class_fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

        imshow_plot(scal_miss_class_fig.add_subplot(gs[1,0]), p_scaled_inf_r_rv, extent=[0,max_r,min_rv,max_rv],split_yscale_dict=split_yscale_dict,hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",kwargs=scale_miss_class_args, title="Primary Snap")
        imshow_plot(scal_miss_class_fig.add_subplot(gs[1,1]), p_scaled_inf_r_tv, extent=[0,max_r,min_tv,max_tv],split_yscale_dict=split_yscale_dict,y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=scale_miss_class_args)
        imshow_plot(scal_miss_class_fig.add_subplot(gs[1,2]), p_scaled_inf_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],split_yscale_dict=split_yscale_dict,hide_xticks=True,hide_yticks=True,kwargs=scale_miss_class_args)
        imshow_plot(scal_miss_class_fig.add_subplot(gs[1,3]), c_scaled_inf_r_rv, extent=[0,max_r,min_rv,max_rv],split_yscale_dict=split_yscale_dict,y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",hide_xticks=True,kwargs=scale_miss_class_args, title="Secondary Snap")
        
        imshow_plot(scal_miss_class_fig.add_subplot(gs[2,0]), p_scaled_orb_r_rv, extent=[0,max_r,min_rv,max_rv],split_yscale_dict=split_yscale_dict,hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",kwargs=scale_miss_class_args)
        imshow_plot(scal_miss_class_fig.add_subplot(gs[2,1]), p_scaled_orb_r_tv, extent=[0,max_r,min_tv,max_tv],split_yscale_dict=split_yscale_dict,y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=scale_miss_class_args)
        imshow_plot(scal_miss_class_fig.add_subplot(gs[2,2]), p_scaled_orb_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],split_yscale_dict=split_yscale_dict,hide_xticks=True,hide_yticks=True,kwargs=scale_miss_class_args)
        imshow_plot(scal_miss_class_fig.add_subplot(gs[2,3]), c_scaled_orb_r_rv, extent=[0,max_r,min_rv,max_rv],split_yscale_dict=split_yscale_dict,y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",hide_xticks=True,kwargs=scale_miss_class_args)
        
        imshow_plot(scal_miss_class_fig.add_subplot(gs[3,0]), p_scaled_all_r_rv, extent=[0,max_r,min_rv,max_rv],split_yscale_dict=split_yscale_dict,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",text="All Misclassified\nScaled",kwargs=scale_miss_class_args)
        imshow_plot(scal_miss_class_fig.add_subplot(gs[3,1]), p_scaled_all_r_tv, extent=[0,max_r,min_tv,max_tv],split_yscale_dict=split_yscale_dict,x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$",kwargs=scale_miss_class_args)
        imshow_plot(scal_miss_class_fig.add_subplot(gs[3,2]), p_scaled_all_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],split_yscale_dict=split_yscale_dict,x_label="$v_r/v_{200m}$",hide_yticks=True,kwargs=scale_miss_class_args)
        imshow_plot(scal_miss_class_fig.add_subplot(gs[3,3]), c_scaled_all_r_rv, extent=[0,max_r,min_rv,max_rv],split_yscale_dict=split_yscale_dict,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",text="All Misclassified\nScaled",kwargs=scale_miss_class_args)

        scal_misclas_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=p_min_ptl, vmax=p_max_diff),cmap=magma_cmap), cax=plt.subplot(gs[1:,-1]))
        scal_misclas_color_bar.set_label("Num Incorrect Particles (inf/orb) / Total Particles (inf/orb)",fontsize=16)
        scal_misclas_color_bar.ax.tick_params(labelsize=14)
        
        scal_miss_class_fig.savefig(save_location + title + "scaled_miss_class.png")

def plot_r_rv_tv_graph(orb_inf, r, rv, tv, correct_orb_inf, title, num_bins, save_location):
    with timed("Finished particle phase space plots"):
        print("Starting particle phase space plots")
        
        plt.rcParams['figure.constrained_layout.use'] = True

        min_ptl = 1e-3

        max_r = np.max(r)
        max_rv = np.max(rv)
        min_rv = np.min(rv)
        max_tv = np.max(tv)
        min_tv = np.min(tv)
        
        r_range = [0, max_r]
        rv_range = [min_rv, max_rv]
        tv_range = [min_tv, max_tv]
        
        ml_inf_r, ml_orb_r = split_orb_inf(r, orb_inf)
        ml_inf_rv, ml_orb_rv = split_orb_inf(rv, orb_inf)
        ml_inf_tv, ml_orb_tv = split_orb_inf(tv, orb_inf)
        
        act_inf_r, act_orb_r = split_orb_inf(r, correct_orb_inf)
        act_inf_rv, act_orb_rv = split_orb_inf(rv, correct_orb_inf)
        act_inf_tv, act_orb_tv = split_orb_inf(tv, correct_orb_inf)

        ml_max_ptl, ml_orb_r_rv, ml_orb_r_tv, ml_orb_rv_tv, ml_inf_r_rv, ml_inf_r_tv, ml_inf_rv_tv = create_hist_max_ptl(min_ptl,min_ptl, ml_inf_r, ml_orb_r, ml_inf_rv, ml_orb_rv, ml_inf_tv, ml_orb_tv, num_bins, r_range, rv_range, tv_range)
        act_max_ptl, act_orb_r_rv, act_orb_r_tv, act_orb_rv_tv, act_inf_r_rv, act_inf_r_tv, act_inf_rv_tv = create_hist_max_ptl(min_ptl,min_ptl, act_inf_r, act_orb_r, act_inf_rv, act_orb_rv, act_inf_tv, act_orb_tv, num_bins, r_range, rv_range, tv_range, bin_r_rv=ml_orb_r_rv[1:], bin_r_tv=ml_orb_r_tv[1:],bin_rv_tv=ml_orb_rv_tv[1:])    
        
        floor = 200
        per_err_1 = percent_error(ml_orb_r_rv[0], act_orb_r_rv[0]).T
        per_err_2 = percent_error(ml_orb_r_tv[0], act_orb_r_tv[0]).T
        per_err_3 = percent_error(ml_orb_rv_tv[0], act_orb_rv_tv[0]).T
        per_err_4 = percent_error(ml_inf_r_rv[0], act_inf_r_rv[0]).T
        per_err_5 = percent_error(ml_inf_r_tv[0], act_inf_r_tv[0]).T
        per_err_6 = percent_error(ml_inf_rv_tv[0], act_inf_rv_tv[0]).T

        max_err = np.max(np.array([np.max(per_err_1),np.max(per_err_2),np.max(per_err_3),np.max(per_err_4),np.max(per_err_5),np.max(per_err_6)]))
        min_err = np.min(np.array([np.min(per_err_1),np.min(per_err_2),np.min(per_err_3),np.min(per_err_4),np.min(per_err_5),np.min(per_err_6)]))

        if ml_max_ptl > act_max_ptl:
            max_ptl = ml_max_ptl
        else:
            max_ptl = act_max_ptl
    
        cmap = plt.get_cmap("magma")
        per_err_cmap = plt.get_cmap("cividis")

        widths = [4,4,4,.5]
        heights = [4,4]
        
        inf_fig = plt.figure()
        inf_fig.suptitle("Infalling Particles: " + title)
        gs = inf_fig.add_gridspec(2,4,width_ratios = widths, height_ratios = heights)
        
        phase_plot(inf_fig.add_subplot(gs[0,0]), ml_inf_r, ml_inf_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$",axisfontsize=12)
        phase_plot(inf_fig.add_subplot(gs[0,1]), ml_inf_r, ml_inf_tv, min_ptl, max_ptl, range=[r_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="ML Predictions",axisfontsize=12)
        phase_plot(inf_fig.add_subplot(gs[0,2]), ml_inf_rv, ml_inf_tv, min_ptl, max_ptl, range=[rv_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$",axisfontsize=12)
        phase_plot(inf_fig.add_subplot(gs[1,0]), act_inf_r, act_inf_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$",axisfontsize=12)
        phase_plot(inf_fig.add_subplot(gs[1,1]), act_inf_r, act_inf_tv, min_ptl, max_ptl, range=[r_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="Actual Distribution",axisfontsize=12)
        phase_plot(inf_fig.add_subplot(gs[1,2]), act_inf_rv, act_inf_tv, min_ptl, max_ptl, range=[rv_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$",axisfontsize=12)
        
        inf_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]))
        
        inf_fig.savefig(save_location + title + "ptls_inf.png")
        
    #########################################################################################################################################################
        
        orb_fig = plt.figure()
        orb_fig.suptitle("Orbiting Particles: " + title)
        gs = orb_fig.add_gridspec(2,4,width_ratios = widths, height_ratios = heights)
        
        phase_plot(orb_fig.add_subplot(gs[0,0]), ml_orb_r, ml_orb_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$",axisfontsize=12)
        phase_plot(orb_fig.add_subplot(gs[0,1]), ml_orb_r, ml_orb_tv, min_ptl, max_ptl, range=[r_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="ML Predictions",axisfontsize=12)
        phase_plot(orb_fig.add_subplot(gs[0,2]), ml_orb_rv, ml_orb_tv, min_ptl, max_ptl, range=[rv_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$",axisfontsize=12)
        phase_plot(orb_fig.add_subplot(gs[1,0]), act_orb_r, act_orb_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$",axisfontsize=12)
        phase_plot(orb_fig.add_subplot(gs[1,1]), act_orb_r, act_orb_tv, min_ptl, max_ptl, range=[r_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="Actual Distribution",axisfontsize=12)
        phase_plot(orb_fig.add_subplot(gs[1,2]), act_orb_rv, act_orb_tv, min_ptl, max_ptl, range=[rv_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$",axisfontsize=12)
        
        orb_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]), pad = 0.1)
        
        orb_fig.savefig(save_location + title + "ptls_orb.png")    
        
    #########################################################################################################################################################
        
        only_r_rv_widths = [4,4,.5]
        only_r_rv_heights = [4,4]
        only_r_rv_fig = plt.figure()
        only_r_rv_fig.suptitle("Radial Velocity Versus Radius: " + title)
        gs = only_r_rv_fig.add_gridspec(2,3,width_ratios = only_r_rv_widths, height_ratios = only_r_rv_heights)
        
        phase_plot(only_r_rv_fig.add_subplot(gs[0,0]), ml_orb_r, ml_orb_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="ML Predicted Orbiting Particles")
        phase_plot(only_r_rv_fig.add_subplot(gs[0,1]), ml_inf_r, ml_inf_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="ML Predicted Infalling Particles")
        phase_plot(only_r_rv_fig.add_subplot(gs[1,0]), act_orb_r, act_orb_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="Actual Orbiting Particles")
        phase_plot(only_r_rv_fig.add_subplot(gs[1,1]), act_inf_r, act_inf_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="Actual Infalling Particles")

        
        only_r_rv_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]))
        only_r_rv_fig.savefig(save_location + title + "only_r_rv.png")
        
    #########################################################################################################################################################

        err_fig = plt.figure(figsize=(30,15))
        err_fig.suptitle("Percent Error " + title)
        gs = err_fig.add_gridspec(2,4,width_ratios = widths, height_ratios = heights)
        
        err_fig_kwargs = {
            "norm":colors.CenteredNorm(),
            "origin":"lower",
            "aspect":"auto",
            "cmap":per_err_cmap,
        }
        
        imshow_plot(err_fig.add_subplot(gs[0,0]), per_err_1, extent=[0,max_r,min_rv,max_rv], x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$",kwargs=err_fig_kwargs)
        imshow_plot(err_fig.add_subplot(gs[0,1]), per_err_2, extent=[0,max_r,min_tv,max_tv], x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="Orbiting Ptls",kwargs=err_fig_kwargs)
        imshow_plot(err_fig.add_subplot(gs[0,2]), per_err_3, extent=[min_rv,max_rv,min_tv,max_tv], x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$",kwargs=err_fig_kwargs)
        imshow_plot(err_fig.add_subplot(gs[1,0]), per_err_4, extent=[0,max_r,min_rv,max_rv], x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$",kwargs=err_fig_kwargs)
        imshow_plot(err_fig.add_subplot(gs[1,1]), per_err_5, extent=[0,max_r,min_tv,max_tv], x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="Infalling Ptls",kwargs=err_fig_kwargs)
        perr_imshow_img=imshow_plot(err_fig.add_subplot(gs[1,2]), per_err_6, extent=[min_rv,max_rv,min_tv,max_tv], x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$", return_img=True,kwargs=err_fig_kwargs)
        
        perr_color_bar = plt.colorbar(perr_imshow_img, cax=plt.subplot(gs[:,-1]), pad = 0.1)
        
        err_fig.savefig(save_location + title + "percent_error.png")  
    
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
    
def plot_per_err(bins,parameter,act_labels,pred_labels,save_location,x_label,save_param):
    all_err_inf = []
    all_err_orb = []
    num_wrong_inf = []
    num_wrong_orb = []
    scale_wrong_inf = []
    scale_wrong_orb = []
    log_num_wrong_inf = []
    log_num_wrong_orb = []

    for i in range(bins.size - 1):
        curr_act_inf = np.where((parameter > bins[i])&(parameter < bins[i+1])&(act_labels==0))[0].size
        curr_act_orb = np.where((parameter > bins[i])&(parameter < bins[i+1])&(act_labels==1))[0].size
        curr_pred_inf = np.where((parameter > bins[i])&(parameter < bins[i+1])&(pred_labels==0))[0].size
        curr_pred_orb = np.where((parameter > bins[i])&(parameter < bins[i+1])&(pred_labels==1))[0].size
        
        all_err_inf.append(0 if curr_act_inf == 0 else ((curr_pred_inf - curr_act_inf)/curr_act_inf) * 100)
        all_err_orb.append(0 if curr_act_orb == 0 else ((curr_pred_orb - curr_act_orb)/curr_act_orb) * 100)
        num_wrong_inf.append(np.where((parameter > bins[i])&(parameter < bins[i+1])&(act_labels==0)&(pred_labels==1))[0].size)
        num_wrong_orb.append(np.where((parameter > bins[i])&(parameter < bins[i+1])&(act_labels==1)&(pred_labels==0))[0].size)
        scale_wrong_inf.append(0 if curr_act_inf == 0 else np.where((parameter > bins[i])&(parameter < bins[i+1])&(act_labels==0)&(pred_labels==1))[0].size/curr_act_inf)
        scale_wrong_orb.append(0 if curr_act_orb == 0 else np.where((parameter > bins[i])&(parameter < bins[i+1])&(act_labels==1)&(pred_labels==0))[0].size/curr_act_orb)
        
    
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0,vmax=1)
    inf_colors = cmap(norm(scale_wrong_inf))
    orb_colors = cmap(norm(scale_wrong_orb))
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    cumsum_inf = np.cumsum(num_wrong_inf)
    cumsum_orb = np.cumsum(num_wrong_orb)

    widths = [4,4,.2]
    heights = [4,4]
    bar_width = 0.75
    num_bins = bins.size - 1
    index = np.arange(0,num_bins)
    
    r200m_loc = np.where(bins > 1)[0][0]
    
    fig = plt.figure(figsize=(40,20))
    gs = fig.add_gridspec(2,3,width_ratios = widths, height_ratios = heights)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    
    ax1.bar(index,num_wrong_inf,width=bar_width,align='center',color=inf_colors,log=True)
    ax1.vlines(r200m_loc,ymin=0,ymax=np.max(num_wrong_inf),colors='r')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Number of Misclassified")
    ax1.set_title("Infalling Particles")
    ax1.set_xticks(index)
    ax1.set_xticklabels([f'{bins[i]:.3f}-{bins[i + 1]:.3f}' for i in range(num_bins)],rotation=90)
    
    ax2.bar(index,num_wrong_orb,width=bar_width,align='center',color=orb_colors,log=True)
    ax2.vlines(r200m_loc,ymin=0,ymax=np.max(num_wrong_orb),colors='r')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Number of Misclassified")
    ax2.set_title("Orbiting Particles")
    ax2.set_xticks(index)
    ax2.set_xticklabels([f'{bins[i]:.3f}-{bins[i + 1]:.3f}' for i in range(num_bins)],rotation=90)
    
    ax3.plot(index,cumsum_inf,color='r',marker='o')
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Number of Misclassified")
    ax3.set_ylim(np.min([cumsum_inf[0],cumsum_orb[0]])-1000,np.max([cumsum_inf[-1],cumsum_orb[-1]])+1000)
    ax3.set_xticks(index)
    ax3.set_xticklabels([f'{bins[i]:.3f}-{bins[i + 1]:.3f}' for i in range(num_bins)],rotation=90)
    
    ax4.plot(index,cumsum_orb,color='r',marker='o')
    ax4.set_xlabel(x_label)
    ax4.set_ylabel("Number of Misclassified")
    ax4.set_ylim(np.min([cumsum_inf[0],cumsum_orb[0]])-1000,np.max([cumsum_inf[-1],cumsum_orb[-1]])+1000)
    ax4.set_xticks(index)
    ax4.set_xticklabels([f'{bins[i]:.3f}-{bins[i + 1]:.3f}' for i in range(num_bins)],rotation=90)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=plt.subplot(gs[0,-1]))
    cbar.set_label("Number of Misclassified Particles / Number of Actual Particles")

    fig.savefig(save_location + save_param + "_per_err_by_bin.png",bbox_inches="tight")
        
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
    ax[0].legend()
    
    ax[1].scatter(pred_inf_ptls[:,0], pred_inf_ptls[:,1], c='g', label = "Infalling Particles")
    ax[1].scatter(pred_orb_ptls[:,0], pred_orb_ptls[:,1], c='b', label = "Orbiting Particles")
    ax[1].set_title("Predicted Distribution of Orbiting/Infalling Particles")
    ax[1].set_xlabel("X position (kpc)")
    ax[1].set_ylabel("Y position (kpc)")
    ax[1].legend()
    fig.savefig(save_path + "plot_of_halo_both_dist.png")

    fig, ax = plt.subplots(1)
    ax.scatter(corr_class[:,0], corr_class[:,1], c='g', label = "Correctly Labeled")
    ax.scatter(inc_class[:,0], inc_class[:,1], c='r', label = "Incorrectly Labeled")
    ax.set_title("Predicted Distribution of Orbiting/Infalling Particles")
    ax.set_xlabel("X position (kpc)")
    ax.set_ylabel("Y position (kpc)")
    ax.legend()
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
    ax1.legend(fontsize=axis_fontsize)

    ax2 = fig.add_subplot(132,projection='3d')
    ax2.scatter(ptl_pos[pred_inf,0],ptl_pos[pred_inf,1],ptl_pos[pred_inf,2],c='orange', alpha=0.1)
    ax2.scatter(ptl_pos[pred_orb,0],ptl_pos[pred_orb,1],ptl_pos[pred_orb,2],c='b', alpha=0.1)
    ax2.set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_ylabel("Y position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_zlabel("Z position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_title("Model Predicted Labels", fontsize=title_fontsize)
    ax2.scatter([],[],[],c="orange",label="Infalling Particles")
    ax2.scatter([],[],[],c="b",label="Orbiting Particles")
    ax2.legend(fontsize=axis_fontsize)

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
    ax3.legend(fontsize=axis_fontsize)

    fig.subplots_adjust(wspace=0.05)
    
    fig.savefig("/home/zvladimi/MLOIS/Random_figures/3d_one_halo_all.png")

    fig, ax = plt.subplots(1, 3,figsize=(30,10))
    
    alpha = 0.25

    ax[0].scatter(ptl_pos[real_inf_slice,0],ptl_pos[real_inf_slice,1],c='orange', alpha = alpha, label="Inalling ptls")
    ax[0].scatter(ptl_pos[real_orb_slice,0],ptl_pos[real_orb_slice,1],c='b', alpha = alpha, label="Orbiting ptls")
    ax[0].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[0].set_ylabel("Y position (kpc/h)",fontsize=axis_fontsize)
    ax[0].set_title("Particles Labeled by SPARTA",fontsize=title_fontsize)
    ax[0].legend(fontsize=axis_fontsize)
    
    ax[1].scatter(ptl_pos[pred_inf_slice,0],ptl_pos[pred_inf_slice,1],c='orange', alpha = alpha, label="Predicted Inalling ptls")
    ax[1].scatter(ptl_pos[pred_orb_slice,0],ptl_pos[pred_orb_slice,1],c='b', alpha = alpha, label="Predicted Orbiting ptls")
    ax[1].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[1].set_title("Particles Labeled by ML Model",fontsize=title_fontsize)
    ax[1].tick_params(axis='y', which='both',left=False,labelleft=False)
    ax[1].legend(fontsize=axis_fontsize)
    
    ax[2].scatter(ptl_pos[inc_orb_slice,0],ptl_pos[inc_orb_slice,1],c='r', marker='x', label="Pred: Inalling \n Actual: Orbiting")
    ax[2].scatter(ptl_pos[inc_inf_slice,0],ptl_pos[inc_inf_slice,1],c='r', marker='+', label="Pred: Orbiting \n Actual: Infalling")
    ax[2].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[2].set_title("Incorrectly Labeled Particles",fontsize=title_fontsize)
    ax[2].tick_params(axis='y', which='both',left=False,labelleft=False)
    ax[2].legend(fontsize=axis_fontsize)
    
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
    fig.write_html(path_to_MLOIS + "/Random_figs/high_vel_halo_idx_" + str(halo_idx) + ".html")
    
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
    ax[0].legend()
    
    ax[1].bar(index, log_orb_cnt, bar_width, label='Orbiting',log=True)
    ax[1].bar(index + bar_width, log_inf_cnt, bar_width, label='Infalling',log=True)
    ax[1].set_xlabel('Radius Bins')
    ax[1].set_title('Number of Orbiting and Infalling Particles by Radius Bin')
    ax[1].set_xticks(index + bar_width / 2)
    ax[1].set_xticklabels([f'{log_bins[i]:.2f}-{log_bins[i + 1]:.2f}' for i in range(num_bins)],rotation=90)
    ax[1].legend()
    
    fig.savefig(save_path + "orb_inf_dist.png",bbox_inches="tight")
    
def mov_3d_plt(ptl_pos,ptl_vel,labels):
    inf_mask = np.where(labels == 0)[0]
    orb_msk = np.where(labels == 1)[0]
    inf_ptls = ipv.quiver(ptl_pos[:,0,0],ptl_pos[:,1,0],ptl_pos[:,2,0],ptl_vel[:,0,0],ptl_vel[:,1,0],ptl_vel[:,2,0])
   
def plot_log_vel(phys_vel,radii,label,save_loc,v200m):
    if v200m == -1:
        title = "no_cut"
    else:
        title = str(v200m) + "v200m"
    log_phys_vel = np.log10(phys_vel)
    
    orb_loc = np.where(label == 1)[0]
    inf_loc = np.where(label == 0)[0]
    
    r_range = [0,np.max(radii)]
    pv_range = [np.min(log_phys_vel),np.max(log_phys_vel)]
    plot_range = [r_range,pv_range]
    
    num_bins = 500
    min_ptl = 1
    
    all_hist = histogram(radii,log_phys_vel,num_bins,plot_range,min_ptl,min_ptl)    
    
    max_all_ptl = np.max(all_hist[0])
    
    magma_cmap = plt.get_cmap("magma_r")
    widths = [4,4,4,.5]
    heights = [4,4]
    
    fig = plt.figure(constrained_layout=True, figsize=(25,20))
    fig.suptitle(title,fontsize=32)
    gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0])
    ax5 = fig.add_subplot(gs[1,1])
    ax6 = fig.add_subplot(gs[1,2])
    
    
    phase_plot(ax1, radii, log_phys_vel, min_ptl=min_ptl, max_ptl=max_all_ptl, range=plot_range,num_bins=num_bins,cmap=magma_cmap,y_label="$log_{10}(v_{phys}/v_{200m})$", norm="linear", hide_xticks=True, title="All Particles",axisfontsize=26)
    phase_plot(ax2, radii[inf_loc], log_phys_vel[inf_loc], min_ptl=min_ptl, max_ptl=max_all_ptl, range=plot_range,num_bins=num_bins,cmap=magma_cmap,norm="linear",hide_xticks=True,hide_yticks=True,title="Infalling Particles",axisfontsize=26)
    phase_plot(ax3, radii[orb_loc], log_phys_vel[orb_loc], min_ptl=min_ptl, max_ptl=max_all_ptl, range=plot_range,num_bins=num_bins,cmap=magma_cmap,norm="linear",hide_xticks=True,hide_yticks=True,title="Orbiting Particles",axisfontsize=26)
    if v200m > 0:
        ax1.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
        ax2.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
        ax3.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
    
    lin_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min_ptl, vmax=max_all_ptl),cmap=magma_cmap), cax=plt.subplot(gs[0,-1]))
    lin_color_bar.set_label("Number of Particles")
    
    phase_plot(ax4, radii, log_phys_vel, min_ptl=min_ptl, max_ptl=max_all_ptl, range=plot_range,num_bins=num_bins,cmap=magma_cmap,x_label="$r/R_{200}$",y_label="$log_{10}(v_{phys}/v_{200m})$",axisfontsize=26)
    phase_plot(ax5, radii[inf_loc], log_phys_vel[inf_loc], min_ptl=min_ptl, max_ptl=max_all_ptl, range=plot_range,num_bins=num_bins,cmap=magma_cmap,x_label="$r/R_{200}$",hide_yticks=True,axisfontsize=26)
    phase_plot(ax6, radii[orb_loc], log_phys_vel[orb_loc], min_ptl=min_ptl, max_ptl=max_all_ptl, range=plot_range,num_bins=num_bins,cmap=magma_cmap,x_label="$r/R_{200}$",hide_yticks=True,axisfontsize=26)
    if v200m > 0:
        ax4.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
        ax5.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
        ax6.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
    log_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_all_ptl),cmap=magma_cmap), cax=plt.subplot(gs[1,-1]))
    log_color_bar.set_label("Number of Particles")
    
    fig.savefig(save_loc + "log_phys_vel_" + title + ".png")
    