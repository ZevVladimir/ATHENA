import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
mpl.use('agg')
from utils.calculation_functions import calculate_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report
from colossus.halo import mass_so
import matplotlib.colors as colors
import multiprocessing as mp
from itertools import repeat
import time
from sparta_tools import sparta
import os
from contextlib import contextmanager
import h5py
from pairing import depair
from scipy.spatial import cKDTree
import sys
from matplotlib.animation import FuncAnimation
import seaborn as sns

from utils.data_and_loading_functions import check_pickle_exist_gadget, create_directory, find_closest_z, load_or_pickle_ptl_data
from utils.calculation_functions import calc_v200m, calculate_density

num_processes = mp.cpu_count()

##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
rand_seed = config.getint("MISC","random_seed")
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
path_to_hdf5_file = path_to_SPARTA_data + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
snap_format = config["MISC"]["snap_format"]
curr_chunk_size = config.getint("SEARCH","chunk_size")

sys.path.insert(1, path_to_pygadgetreader)  
from pygadgetreader import readsnap, readheader

@contextmanager
def timed(txt):
    t0 = time.time()
    yield
    t1 = time.time()
    print("%32s time:  %8.5f" % (txt, t1 - t0))

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

def update_density_prf(calc_prf, diff_n_ptl, radii, idx, start_bin, end_bin, mass, act_prf):
    radii_within_range = np.where((radii >= start_bin) & (radii < end_bin))[0]
        
    # If there are particles in this bin and its not the first bin
    # Then add the mass of prior bin to the mass of this bin
    if radii_within_range.size != 0 and idx != 0:
        calc_prf[idx] = calc_prf[idx - 1] + radii_within_range.size * mass
        diff_n_ptl[idx] = ((act_prf[idx] - act_prf[idx-1])/mass) - radii_within_range.size
    # If there are particles in this bin and its  the first bin
    # Then simply the mass of this bin
    elif radii_within_range.size != 0 and idx == 0:
        calc_prf[idx] = radii_within_range.size * mass
        diff_n_ptl[idx] = act_prf[idx]/mass - radii_within_range.size
    # If there are  no particles in this bin and its not the first bin
    # Then use the mass of prior bin 
    elif radii_within_range.size == 0 and idx != 0:
        calc_prf[idx] = calc_prf[idx-1]
        diff_n_ptl[idx] = diff_n_ptl[idx-1]   
    # If there are  no particles in this bin and its the first bin
    # Then use the mass of this bin 
    else:
        calc_prf[idx] = calc_prf[idx-1]
        diff_n_ptl[idx] = (act_prf[idx])/mass
    
    return calc_prf, diff_n_ptl

def create_dens_prf(radii, orbit_assn, act_mass_prf_all, act_mass_prf_1halo, prf_bins, mass):
    act_mass_prf_inf = act_mass_prf_all - act_mass_prf_1halo
    # Create bins for the density profile calculation
    num_prf_bins = act_mass_prf_all.shape[0]

    calc_mass_prf_orb = np.zeros(num_prf_bins)
    calc_mass_prf_inf = np.zeros(num_prf_bins)
    calc_mass_prf_all = np.zeros(num_prf_bins)
    diff_n_inf_ptls = np.zeros(num_prf_bins)
    diff_n_orb_ptls = np.zeros(num_prf_bins)
    diff_n_all_ptls = np.zeros(num_prf_bins)
    
    # determine which radii correspond to orbiting and which to infalling
    orbit_radii = radii[np.where(orbit_assn == 1)[0]]
    infall_radii = radii[np.where(orbit_assn == 0)[0]]

    # loop through each bin's radii range and get the mass of each type of particle
    for i in range(num_prf_bins):
        start_bin = prf_bins[i]
        end_bin = prf_bins[i+1]  
        
        calc_mass_prf_orb, diff_n_orb_ptls = update_density_prf(calc_mass_prf_orb, diff_n_orb_ptls, orbit_radii, i, start_bin, end_bin, mass, act_mass_prf_1halo)      
        calc_mass_prf_inf, diff_n_inf_ptls = update_density_prf(calc_mass_prf_inf, diff_n_inf_ptls, infall_radii, i, start_bin, end_bin, mass, act_mass_prf_inf)      
        calc_mass_prf_all, diff_n_all_ptls  = update_density_prf(calc_mass_prf_all, diff_n_all_ptls, radii, i, start_bin, end_bin, mass, act_mass_prf_all)    
 
    # Calculate the density by divide the mass of each bin by the volume of that bin's radius
    calc_dens_prf_orb = calculate_density(calc_mass_prf_orb, prf_bins[1:])
    calc_dens_prf_inf = calculate_density(calc_mass_prf_inf, prf_bins[1:])
    calc_dens_prf_all = calculate_density(calc_mass_prf_all, prf_bins[1:])
    
    return calc_mass_prf_orb, calc_mass_prf_inf, calc_mass_prf_all, calc_dens_prf_orb, calc_dens_prf_inf, calc_dens_prf_all, diff_n_orb_ptls, diff_n_inf_ptls, diff_n_all_ptls 

def med_prf(prf, num_halo, dtype):
    if num_halo > 1:
        prf = np.stack(prf, axis=0)
    else:
        prf = np.asarray(prf)
        prf = np.reshape(prf, (prf.size,1))
    
    prf = prf.astype(dtype)

    # For median want to have one value for each bin (1,80)
    med_prf = np.median(prf,axis=0)

    return prf, med_prf
    
def compare_density_prf(radii, halo_first, halo_n, act_mass_prf_all, act_mass_prf_orb, mass, orbit_assn, prf_bins, title, save_location, use_mp = False, show_graph = False, save_graph = False):
    # Shape of profiles should be (num halo,num bins)
    # EX: 10 halos, 80 bins (10,80)
    
    t1 = time.time()
    print("Starting Density Profile Plot")
    act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb
    create_directory(save_location + "plots/")
    curr_num_halos = halo_first.shape[0]
    min_disp_halos = int(np.ceil(0.3 * curr_num_halos))
    num_bins = prf_bins.size
    
    # for each halo get the corresponding mass and density profile that the model predicts for it
    # Can do this using either multiprocessing or a for loop
    if use_mp:
        num_processes = mp.cpu_count()
        with mp.Pool(processes=num_processes) as p:
            calc_mass_prf_orb, calc_mass_prf_inf, calc_mass_prf_all, calc_dens_prf_orb, calc_dens_prf_inf, calc_dens_prf_all, diff_n_orb_ptls, diff_n_inf_ptls, diff_n_all_ptls = zip(*p.starmap(create_dens_prf, 
                                        zip((radii[halo_first[i]:halo_first[i]+halo_n[i]] for i in range(curr_num_halos)),
                                            (orbit_assn[halo_first[i]:halo_first[i]+halo_n[i]] for i in range(curr_num_halos)),
                                            (act_mass_prf_all[j] for j in range(curr_num_halos)),
                                            (act_mass_prf_orb[j] for j in range(curr_num_halos)),
                                            repeat(prf_bins),repeat(mass)),
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
        diff_n_orb_ptls = []
        diff_n_inf_ptls = []
        diff_n_all_ptls = []
        
        for i in range(curr_num_halos):
            halo_mass_prf_orb, halo_mass_prf_inf, halo_mass_prf_all, halo_dens_prf_orb, halo_dens_prf_inf, halo_dens_prf_all, halo_diff_n_orb_ptls, halo_diff_n_inf_ptls, halo_diff_n_all_ptls = create_dens_prf(radii[halo_first[i]:halo_first[i]+halo_n[i]], orbit_assn[halo_first[i]:halo_first[i]+halo_n[i]], act_mass_prf_all[i], act_mass_prf_orb[i],prf_bins,mass)
            calc_mass_prf_orb.append(np.array(halo_mass_prf_orb))
            calc_mass_prf_inf.append(np.array(halo_mass_prf_inf))
            calc_mass_prf_all.append(np.array(halo_mass_prf_all))
            calc_dens_prf_orb.append(np.array(halo_dens_prf_orb))
            calc_dens_prf_inf.append(np.array(halo_dens_prf_inf))
            calc_dens_prf_all.append(np.array(halo_dens_prf_all))
            diff_n_orb_ptls.append(np.array(halo_diff_n_orb_ptls))
            diff_n_inf_ptls.append(np.array(halo_diff_n_inf_ptls))
            diff_n_all_ptls.append(np.array(halo_diff_n_all_ptls))

    # For each profile combine all halos and obtain their median values for each bin
    # calc_mass_prf_xxx has shape (num_halo, num_bins)
    calc_mass_prf_orb, med_calc_mass_prf_orb = med_prf(calc_mass_prf_orb, curr_num_halos, np.float32)
    calc_mass_prf_inf, med_calc_mass_prf_inf = med_prf(calc_mass_prf_inf, curr_num_halos, np.float32)
    calc_mass_prf_all, med_calc_mass_prf_all = med_prf(calc_mass_prf_all, curr_num_halos, np.float32)
    calc_dens_prf_orb, med_calc_dens_prf_orb = med_prf(calc_dens_prf_orb, curr_num_halos, np.float32)
    calc_dens_prf_inf, med_calc_dens_prf_inf = med_prf(calc_dens_prf_inf, curr_num_halos, np.float32)
    calc_dens_prf_all, med_calc_dens_prf_all = med_prf(calc_dens_prf_all, curr_num_halos, np.float32)
 
    # Get density profiles by dividing the mass profiles by the volume of each bin
    act_dens_prf_all = calculate_density(act_mass_prf_all, prf_bins[1:])
    act_dens_prf_orb = calculate_density(act_mass_prf_orb, prf_bins[1:])
    act_dens_prf_inf = calculate_density(act_mass_prf_inf, prf_bins[1:])
    
    # Get the median value of the actual mass and density profiles
    med_act_mass_prf_all = np.median(act_mass_prf_all, axis=0)
    med_act_mass_prf_orb = np.median(act_mass_prf_orb, axis=0)
    med_act_mass_prf_inf = np.median(act_mass_prf_inf, axis=0)

    med_act_dens_prf_all = np.median(act_dens_prf_all, axis=0)
    med_act_dens_prf_orb = np.median(act_dens_prf_orb, axis=0)
    med_act_dens_prf_inf = np.median(act_dens_prf_inf, axis=0)
    
    # for each bin checking how many halos have particles there
    # if there are less than half the total number of halos then just treat that bin as having 0
    for i in range(calc_mass_prf_orb.shape[1]):
        if np.where(calc_mass_prf_orb[:,i] > 0)[0].shape[0] < min_disp_halos:
            calc_mass_prf_orb[:,i] = np.NaN
            act_mass_prf_orb[:,i] = np.NaN
            med_calc_mass_prf_orb[i] = np.NaN
            med_act_mass_prf_orb[i] = np.NaN
        if np.where(calc_mass_prf_inf[:,i] > 0)[0].shape[0] < min_disp_halos:
            calc_mass_prf_inf[:,i] = np.NaN
            act_mass_prf_inf[:,i] = np.NaN
            med_calc_mass_prf_inf[i] = np.NaN
            med_act_mass_prf_inf[i] = np.NaN
        if np.where(calc_mass_prf_all[:,i] > 0)[0].shape[0] < min_disp_halos:
            calc_mass_prf_all[:,i] = np.NaN
            act_mass_prf_all[:,i] = np.NaN
            med_calc_mass_prf_all[i] = np.NaN
            med_act_mass_prf_all[i] = np.NaN
        if np.where(calc_dens_prf_orb[:,i] > 0)[0].shape[0] < min_disp_halos:
            calc_dens_prf_orb[:,i] = np.NaN
            act_dens_prf_orb[:,i] = np.NaN
            med_calc_dens_prf_orb[i] = np.NaN
            med_act_dens_prf_orb[i] = np.NaN
        if np.where(calc_dens_prf_inf[:,i] > 0)[0].shape[0] < min_disp_halos:
            calc_dens_prf_inf[:,i] = np.NaN
            act_dens_prf_inf[:,i] = np.NaN
            med_calc_dens_prf_inf[i] = np.NaN
            med_act_dens_prf_inf[i] = np.NaN
        if np.where(calc_dens_prf_all[:,i] > 0)[0].shape[0] < min_disp_halos:
            calc_dens_prf_all[:,i] = np.NaN
            act_dens_prf_all[:,i] = np.NaN
            med_calc_dens_prf_all[i] = np.NaN
            med_act_dens_prf_all[i] = np.NaN
    
    # Get the ratio of the calculated profile with the actual profile
    with np.errstate(divide='ignore', invalid='ignore'):
        all_mass_ratio = np.divide(calc_mass_prf_all,act_mass_prf_all) - 1
        inf_mass_ratio = np.divide(calc_mass_prf_inf,act_mass_prf_inf) - 1
        orb_mass_ratio = np.divide(calc_mass_prf_orb,act_mass_prf_orb) - 1
        all_dens_ratio = np.divide(calc_dens_prf_all,act_dens_prf_all) - 1
        inf_dens_ratio = np.divide(calc_dens_prf_inf,act_dens_prf_inf) - 1
        orb_dens_ratio = np.divide(calc_dens_prf_orb,act_dens_prf_orb) - 1

    
    # Find the upper and lower bound for scatter for calculated profiles
    # Want shape to be (1,80)
    upper_calc_mass_prf_orb = np.percentile(calc_mass_prf_orb, q=84.1, axis=0)
    lower_calc_mass_prf_orb = np.percentile(calc_mass_prf_orb, q=15.9, axis=0)
    upper_calc_mass_prf_inf = np.percentile(calc_mass_prf_inf, q=84.1, axis=0)
    lower_calc_mass_prf_inf = np.percentile(calc_mass_prf_inf, q=15.9, axis=0)
    upper_calc_mass_prf_all = np.percentile(calc_mass_prf_all, q=84.1, axis=0)
    lower_calc_mass_prf_all = np.percentile(calc_mass_prf_all, q=15.9, axis=0)
    upper_calc_dens_prf_orb = np.percentile(calc_dens_prf_orb, q=84.1, axis=0)
    lower_calc_dens_prf_orb = np.percentile(calc_dens_prf_orb, q=15.9, axis=0)
    upper_calc_dens_prf_inf = np.percentile(calc_dens_prf_inf, q=84.1, axis=0)
    lower_calc_dens_prf_inf = np.percentile(calc_dens_prf_inf, q=15.9, axis=0)
    upper_calc_dens_prf_all = np.percentile(calc_dens_prf_all, q=84.1, axis=0)
    lower_calc_dens_prf_all = np.percentile(calc_dens_prf_all, q=15.9, axis=0)
    
    # Same for actual profiles    
    upper_orb_dens_ratio = np.percentile(orb_dens_ratio, q=84.1, axis=0)
    lower_orb_dens_ratio = np.percentile(orb_dens_ratio, q=15.9, axis=0)
    upper_inf_dens_ratio = np.percentile(inf_dens_ratio, q=84.1, axis=0)
    lower_inf_dens_ratio = np.percentile(inf_dens_ratio, q=15.9, axis=0)
    upper_all_dens_ratio = np.percentile(all_dens_ratio, q=84.1, axis=0)
    lower_all_dens_ratio = np.percentile(all_dens_ratio, q=15.9, axis=0)

    # Take the median value of the ratios
    med_all_ratio = np.median(all_dens_ratio, axis=0)
    med_inf_ratio = np.median(inf_dens_ratio, axis=0)
    med_orb_ratio = np.median(orb_dens_ratio, axis=0)

    middle_bins = (prf_bins[1:] + prf_bins[:-1]) / 2

    fig, ax = plt.subplots(1,3, figsize=(15,30))
    titlefntsize=26
    axisfntsize=20
    tickfntsize=16
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
    
    # ax[0].fill_between(middle_bins, lower_calc_mass_prf_all, upper_calc_mass_prf_all, color='r', alpha=fill_alpha)
    # ax[0].fill_between(middle_bins, lower_calc_mass_prf_inf, upper_calc_mass_prf_inf, color='g', alpha=fill_alpha)
    # ax[0].fill_between(middle_bins, lower_calc_mass_prf_orb, upper_calc_mass_prf_orb, color='b', alpha=fill_alpha)
    
    ax[0].set_title("ML Predicted vs Actual Mass Profile",fontsize=titlefntsize)
    ax[0].set_xlabel("Radius $r/R_{200m}$", fontsize=axisfntsize)
    ax[0].set_ylabel("Mass $M_\odot$", fontsize=axisfntsize)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_box_aspect(1)
    ax[0].tick_params(axis='both',which='both',labelsize=tickfntsize)
    ax[0].tick_params(axis='both',which='both',labelsize=tickfntsize)
    ax[0].legend(fontsize=legendfntsize)
    
    ax[1].plot(middle_bins, med_calc_dens_prf_all, 'r-', label = "ML density profile all ptls")
    ax[1].plot(middle_bins, med_calc_dens_prf_orb, 'b-', label = "ML density profile orb ptls")
    ax[1].plot(middle_bins, med_calc_dens_prf_inf, 'g-', label = "ML density profile inf ptls")
    ax[1].plot(middle_bins, med_act_dens_prf_all, 'r--', label = "SPARTA density profile all ptls")
    ax[1].plot(middle_bins, med_act_dens_prf_orb, 'b--', label = "SPARTA density profile orb ptls")
    ax[1].plot(middle_bins, med_act_dens_prf_inf, 'g--', label = "SPARTA density profile inf ptls")
    
    # ax[1].fill_between(middle_bins, lower_calc_dens_prf_all, upper_calc_dens_prf_all, color='r', alpha=fill_alpha)
    # ax[1].fill_between(middle_bins, lower_calc_dens_prf_inf, upper_calc_dens_prf_inf, color='g', alpha=fill_alpha)
    # ax[1].fill_between(middle_bins, lower_calc_dens_prf_orb, upper_calc_dens_prf_orb, color='b', alpha=fill_alpha)
    
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
    
    ax[2].fill_between(middle_bins, lower_all_dens_ratio, upper_all_dens_ratio, color='r', alpha=fill_alpha)
    ax[2].fill_between(middle_bins, lower_inf_dens_ratio, upper_inf_dens_ratio, color='g', alpha=fill_alpha)
    ax[2].fill_between(middle_bins, lower_orb_dens_ratio, upper_orb_dens_ratio, color='b', alpha=fill_alpha)    
    
    ax[2].set_title("(ML Predicted / Actual Density Profile) - 1",fontsize=titlefntsize)
    ax[2].set_xlabel("Radius $r/R_{200m}$", fontsize=axisfntsize)
    ax[2].set_ylabel("(ML Dens Prf / Act Dens Prf) - 1", fontsize=axisfntsize)
    #ax[2].set_ylim(0,8)
    top_orb_tick = np.round(np.nanmax(upper_orb_dens_ratio),2)
    bot_orb_tick = np.round(np.nanmin(lower_orb_dens_ratio),2)
    top_inf_tick = np.round(np.nanmax(upper_inf_dens_ratio),2)
    bot_inf_tick = np.round(np.nanmin(lower_inf_dens_ratio),2)

    ax[2].set_xscale("log")
    ax[2].set_yscale("symlog")
    ax[2].set_box_aspect(1)
    ax[2].tick_params(axis='both',which='both',labelsize=tickfntsize)
    ax[2].set_yticks([bot_inf_tick, bot_orb_tick, 0, top_inf_tick, top_orb_tick])
    ax[2].legend(fontsize=legendfntsize)    
    
    if save_graph:
        fig.set_size_inches(50, 25)
        fig.savefig(save_location + "plots/" + title + "dens_prfl_rat.png", bbox_inches='tight')
    if show_graph:
        plt.show()
    plt.close()

    t2 = time.time()
    print("Finished Density Profile Plot in: ", np.round((t2-t1),2), "seconds", np.round(((t2-t1)/60),2), "minutes")
    return diff_n_inf_ptls, diff_n_orb_ptls, diff_n_all_ptls, middle_bins
    
def brute_force(curr_particles_pos, r200, halo_x, halo_y, halo_z):
    within_box = curr_particles_pos[np.where((curr_particles_pos[:,0] < r200 + halo_x) & (curr_particles_pos[:,0] > r200 - halo_x) & (curr_particles_pos[:,1] < r200 + halo_y) & (curr_particles_pos[:,1] > r200 - halo_y) & (curr_particles_pos[:,2] < r200 + halo_z) & (curr_particles_pos[:,2] > r200 - halo_z))]
    brute_radii = calculate_distance(halo_x, halo_y, halo_z, within_box[:,0], within_box[:,1], within_box[:,2], within_box.shape[0])
    return within_box[np.where(brute_radii <= r200)]

#TODO add brute force comparison graph

#TODO add radial vel vs position graph

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

def histogram(x,y,bins,range,min_ptl,set_ptl):
    hist = np.histogram2d(x, y, bins=bins, range=range)
    hist[0][hist[0] < min_ptl] = set_ptl
    return hist
  
def split_orb_inf(data, labels):
    infall = data[np.where(labels == 0)]
    orbit = data[np.where(labels == 1)]
    return infall, orbit
 
def phase_plot(ax, x, y, min_ptl, max_ptl, range, num_bins, cmap, x_label="", y_label="", norm = "log", xrange=None, yrange=None, hide_xticks=False, hide_yticks=False,text="", axisfontsize=18, title=""):
    ax.hist2d(x, y, bins=num_bins, range=range, density=False, weights=None, cmin=min_ptl, cmap=cmap, norm=norm, vmin=min_ptl, vmax=max_ptl)
    
    if xrange != None:
        ax.set_xlim(xrange)
    if yrange != None:
        ax.set_ylim(yrange)
    if title != "":
        ax.set_title(title)
    if x_label != "":
        ax.set_xlabel(x_label,fontsize=axisfontsize)
    if y_label != "":
        ax.set_ylabel(y_label,fontsize=axisfontsize)
    if hide_xticks:
        ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False) 
    if hide_yticks:
        ax.tick_params(axis='y', which='both',left=False,labelleft=False) 
    if text != "":
        ax.text(.01,.03, text, ha="left", va="bottom", transform=ax.transAxes, fontsize="x-large", bbox={"facecolor":'white',"alpha":.9,})
        
def imshow_plot(ax, img, extent, x_label="", y_label="", text="", title="", return_img=False, hide_xticks=False, hide_yticks=False, axisfontsize=16, kwargs={}):
    img=ax.imshow(img, interpolation="none", extent = extent, **kwargs)
    if text != "":
        ax.text(.01,.03, text, ha="left", va="bottom", transform=ax.transAxes, fontsize="large", bbox={"facecolor":'white',"alpha":0.9,})
    if title != "":
        ax.set_title(title)
    if x_label != "":
        ax.set_xlabel(x_label,fontsize=axisfontsize)
    if y_label != "":
        ax.set_ylabel(y_label,fontsize=axisfontsize)
    if hide_xticks:
        ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False) 
    if hide_yticks:
        ax.tick_params(axis='y', which='both',left=False,labelleft=False) 
    if return_img:
        return img

def update_miss_class(img, miss_class, act, miss_class_min, act_min):
    # Where there are no misclassified particles but there are actual particles set to 0
    img = np.where((miss_class < 1) & (act >= act_min), miss_class_min, img)
    # Where there are miss classified particles but they won't show up on the image, set them to the min
    img = np.where((miss_class >= 1) & (img < miss_class_min) & (act >= act_min), miss_class_min, img)
    return img.T

def create_hist_max_ptl(min_ptl, set_ptl, inf_r, orb_r, inf_rv, orb_rv, inf_tv, orb_tv, num_bins, r_range, rv_range, tv_range, bin_r_rv = None, bin_r_tv = None, bin_rv_tv = None):
    if bin_r_rv == None:
        bins = num_bins
        orb_r_rv = histogram(orb_r, orb_rv, bins=bins, range=[r_range,rv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        orb_r_tv = histogram(orb_r, orb_tv, bins=bins, range=[r_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        orb_rv_tv = histogram(orb_rv, orb_tv, bins=bins, range=[rv_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_r_rv = histogram(inf_r, inf_rv, bins=bins, range=[r_range,rv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_r_tv = histogram(inf_r, inf_tv, bins=bins, range=[r_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_rv_tv = histogram(inf_rv, inf_tv, bins=bins, range=[rv_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl)
    else:
        orb_r_rv = histogram(orb_r, orb_rv, bins=bin_r_rv, range=[r_range,rv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        orb_r_tv = histogram(orb_r, orb_tv, bins=bin_r_tv, range=[r_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        orb_rv_tv = histogram(orb_rv, orb_tv, bins=bin_rv_tv, range=[rv_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_r_rv = histogram(inf_r, inf_rv, bins=bin_r_rv, range=[r_range,rv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_r_tv = histogram(inf_r, inf_tv, bins=bin_r_tv, range=[r_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_rv_tv = histogram(inf_rv, inf_tv, bins=bin_rv_tv, range=[rv_range,tv_range],min_ptl=min_ptl,set_ptl=set_ptl)

    max_ptl = np.max(np.array([np.max(orb_r_rv[0]),np.max(orb_r_tv[0]),np.max(orb_rv_tv[0]),np.max(inf_r_rv[0]),np.max(inf_r_tv[0]),np.max(inf_rv_tv[0]),]))
    
    return max_ptl, orb_r_rv, orb_r_tv, orb_rv_tv, inf_r_rv, inf_r_tv, inf_rv_tv

def percent_error(pred, act):
    return (((pred - act))/act) * 100

def calc_misclassified(correct_labels, ml_labels, r, rv, tv, r_range, rv_range, tv_range, num_bins, model_info,dataset_name): 
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

    # print("tot num ptl:",tot_num_ptl)
    # print("num incorrect inf", inc_inf.shape[0], ",", np.round(((inc_inf.shape[0]/num_inf)*100),2), "% of infalling ptls")
    # print("num incorrect orb", inc_orb.shape[0], ",", np.round(((inc_orb.shape[0]/num_orb)*100),2), "% of orbiting ptls")
    # print("num incorrect tot", tot_num_inc, ",", np.round(((tot_num_inc/tot_num_ptl) * 100),2), "% of all ptls")
    
    inc_orb_r = r[inc_orb]
    inc_inf_r = r[inc_inf]
    inc_orb_rv = rv[inc_orb]
    inc_inf_rv = rv[inc_inf]
    inc_orb_tv = tv[inc_orb]
    inc_inf_tv = tv[inc_inf]

    act_inf_r, act_orb_r = split_orb_inf(r, correct_labels)
    act_inf_rv, act_orb_rv = split_orb_inf(rv, correct_labels)
    act_inf_tv, act_orb_tv = split_orb_inf(tv, correct_labels)

    max_all_ptl, act_orb_r_rv, act_orb_r_tv, act_orb_rv_tv, act_inf_r_rv, act_inf_r_tv, act_inf_rv_tv = create_hist_max_ptl(act_min_ptl, 0, act_inf_r, act_orb_r, act_inf_rv, act_orb_rv, act_inf_tv, act_orb_tv, num_bins, r_range, rv_range, tv_range)    
    max_ptl, inc_orb_r_rv, inc_orb_r_tv, inc_orb_rv_tv, inc_inf_r_rv, inc_inf_r_tv, inc_inf_rv_tv = create_hist_max_ptl(min_ptl, min_ptl, inc_inf_r, inc_orb_r, inc_inf_rv, inc_orb_rv, inc_inf_tv, inc_orb_tv, num_bins, r_range, rv_range, tv_range, bin_r_rv=act_orb_r_rv[1:], bin_r_tv=act_orb_r_tv[1:],bin_rv_tv=act_orb_rv_tv[1:])

    all_inc_r_rv = (inc_orb_r_rv[0] + inc_inf_r_rv[0])
    all_inc_r_tv = (inc_orb_r_tv[0] + inc_inf_r_tv[0])
    all_inc_rv_tv = (inc_orb_rv_tv[0] + inc_inf_rv_tv[0])
    all_act_r_rv = (act_orb_r_rv[0] + act_inf_r_rv[0])
    all_act_r_tv = (act_orb_r_tv[0] + act_inf_r_tv[0])
    all_act_rv_tv = (act_orb_rv_tv[0] + act_inf_rv_tv[0])

    scaled_orb_r_rv = (np.divide(inc_orb_r_rv[0],act_orb_r_rv[0],out=np.zeros_like(inc_orb_r_rv[0]), where=act_orb_r_rv[0]!=0)).T
    scaled_orb_r_tv = (np.divide(inc_orb_r_tv[0],act_orb_r_tv[0],out=np.zeros_like(inc_orb_r_tv[0]), where=act_orb_r_tv[0]!=0)).T
    scaled_orb_rv_tv = (np.divide(inc_orb_rv_tv[0],act_orb_rv_tv[0],out=np.zeros_like(inc_orb_rv_tv[0]), where=act_orb_rv_tv[0]!=0)).T
    scaled_inf_r_rv = (np.divide(inc_inf_r_rv[0],act_inf_r_rv[0],out=np.zeros_like(inc_inf_r_rv[0]), where=act_inf_r_rv[0]!=0)).T
    scaled_inf_r_tv = (np.divide(inc_inf_r_tv[0],act_inf_r_tv[0],out=np.zeros_like(inc_inf_r_tv[0]), where=act_inf_r_tv[0]!=0)).T
    scaled_inf_rv_tv = (np.divide(inc_inf_rv_tv[0],act_inf_rv_tv[0],out=np.zeros_like(inc_inf_rv_tv[0]), where=act_inf_rv_tv[0]!=0)).T
    scaled_all_r_rv = (np.divide(all_inc_r_rv,all_act_r_rv,out=np.zeros_like(all_inc_r_rv), where=all_act_r_rv!=0)).T
    scaled_all_r_tv = (np.divide(all_inc_r_tv,all_act_r_tv,out=np.zeros_like(all_inc_r_tv), where=all_act_r_tv!=0)).T
    scaled_all_rv_tv = (np.divide(all_inc_rv_tv,all_act_rv_tv,out=np.zeros_like(all_inc_rv_tv), where=all_act_rv_tv!=0)).T

    # For any spots that have no missclassified particles but there are particles there set it to the minimum amount so it still shows up in the plot.
    scaled_orb_r_rv = update_miss_class(scaled_orb_r_rv.T, inc_orb_r_rv[0], act_orb_r_rv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_orb_r_tv = update_miss_class(scaled_orb_r_tv.T, inc_orb_r_tv[0], act_orb_r_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_orb_rv_tv = update_miss_class(scaled_orb_rv_tv.T, inc_orb_rv_tv[0], act_orb_rv_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_inf_r_rv = update_miss_class(scaled_inf_r_rv.T, inc_inf_r_rv[0], act_inf_r_rv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_inf_r_tv = update_miss_class(scaled_inf_r_tv.T, inc_inf_r_tv[0], act_inf_r_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_inf_rv_tv = update_miss_class(scaled_inf_rv_tv.T, inc_inf_rv_tv[0], act_inf_rv_tv[0], miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_all_r_rv = update_miss_class(scaled_all_r_rv.T, all_inc_r_rv, all_act_r_rv, miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_all_r_tv = update_miss_class(scaled_all_r_tv.T, all_inc_r_tv, all_act_r_tv, miss_class_min=min_ptl, act_min=act_min_ptl)
    scaled_all_rv_tv = update_miss_class(scaled_all_rv_tv.T, all_inc_rv_tv, all_act_rv_tv, miss_class_min=min_ptl, act_min=act_min_ptl)
    
    max_diff = np.max(np.array([np.max(scaled_orb_r_rv),np.max(scaled_orb_r_tv),np.max(scaled_orb_rv_tv),
                                np.max(scaled_inf_r_rv),np.max(scaled_inf_r_tv),np.max(scaled_inf_rv_tv),
                                np.max(scaled_all_r_rv),np.max(scaled_all_r_tv),np.max(scaled_all_rv_tv)]))
    
    return misclass_dict, min_ptl, max_diff, max_all_ptl, all_inc_r_rv, all_inc_r_tv, all_inc_rv_tv, scaled_inf_r_rv, scaled_inf_r_tv, scaled_inf_rv_tv, scaled_orb_r_rv, scaled_orb_r_tv, scaled_orb_rv_tv, scaled_all_r_rv, scaled_all_r_tv, scaled_all_rv_tv
    
def plot_misclassified(p_corr_labels, p_ml_labels, p_r, p_rv, p_tv, c_r, c_rv, c_tv, title, num_bins,save_location,model_info,dataset_name):
    if "Results" not in model_info:
        model_info["Results"] = {}
    
    t1 = time.time()
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
    
    print("Primary Snap Misclassification")
    p_misclass_dict, p_min_ptl, p_max_diff, p_max_all_ptl, p_all_inc_r_rv, p_all_inc_r_tv, p_all_inc_rv_tv, p_scaled_inf_r_rv, p_scaled_inf_r_tv, p_scaled_inf_rv_tv, p_scaled_orb_r_rv, p_scaled_orb_r_tv, p_scaled_orb_rv_tv, p_scaled_all_r_rv, p_scaled_all_r_tv, p_scaled_all_rv_tv = calc_misclassified(p_corr_labels, p_ml_labels, p_r, p_rv, p_tv, r_range, rv_range, tv_range, num_bins=num_bins, model_info=model_info,dataset_name=dataset_name)
    print("Secondary Snap Misclassification")
    c_misclass_dict, c_min_ptl, c_max_diff, c_max_all_ptl, c_all_inc_r_rv, c_all_inc_r_tv, c_all_inc_rv_tv, c_scaled_inf_r_rv, c_scaled_inf_r_tv, c_scaled_inf_rv_tv, c_scaled_orb_r_rv, c_scaled_orb_r_tv, c_scaled_orb_rv_tv, c_scaled_all_r_rv, c_scaled_all_r_tv, c_scaled_all_rv_tv = calc_misclassified(c_corr_labels, c_ml_labels, c_r, c_rv, c_tv, r_range, rv_range, tv_range, num_bins=num_bins, model_info=model_info,dataset_name=dataset_name)
    
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

    widths = [4,4,4,.5]
    heights = [4,4,4,4,4]
    
    scal_miss_class_fig = plt.figure(constrained_layout=True, figsize=(15,15))
    #scal_miss_class_fig.suptitle("Misclassified Particles/Num Targets " + title)
    gs = scal_miss_class_fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
    plt.rcParams.update({'font.size': 12})
    
    phase_plot(scal_miss_class_fig.add_subplot(gs[0,0]), c_r, c_rv, min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,rv_range],num_bins=num_bins,cmap=cividis_cmap,y_label="$v_r/v_{200m}$", hide_xticks=True, text="Actual\nDistribution", title="Secondary Snap")
    phase_plot(scal_miss_class_fig.add_subplot(gs[0,1]), p_r, p_rv, min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,rv_range],num_bins=num_bins,cmap=cividis_cmap, hide_xticks=True, hide_yticks=False,y_label="$v_r/v_{200m}$",text="Actual\nDistribution")
    phase_plot(scal_miss_class_fig.add_subplot(gs[0,2]), p_r, p_tv, min_ptl=10, max_ptl=p_max_all_ptl, range=[r_range,tv_range],num_bins=num_bins,cmap=cividis_cmap,y_label="$v_t/v_{200m}$",hide_xticks=True)
    phase_plot(scal_miss_class_fig.add_subplot(gs[0,3]), p_rv, p_tv, min_ptl=10, max_ptl=p_max_all_ptl, range=[rv_range,tv_range],num_bins=num_bins,cmap=cividis_cmap, hide_xticks=True, hide_yticks=True)
    
    imshow_plot(scal_miss_class_fig.add_subplot(gs[1,0]), c_all_inc_r_rv.T, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",hide_xticks=True,text="All Misclassified",kwargs=all_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[1,1]), p_all_inc_r_rv.T, extent=[0,max_r,min_rv,max_rv],hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="All Misclassified",kwargs=all_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[1,2]), p_all_inc_r_tv.T, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=all_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[1,3]), p_all_inc_rv_tv.T, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=all_miss_class_args)
    phase_plt_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=1, vmax=p_max_all_ptl),cmap=cividis_cmap), cax=plt.subplot(gs[0:2,-1]))
    phase_plt_color_bar.set_label("Number of Particles")

    imshow_plot(scal_miss_class_fig.add_subplot(gs[2,0]), c_scaled_inf_r_rv, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",hide_xticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[2,1]), p_scaled_inf_r_rv, extent=[0,max_r,min_rv,max_rv],hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[2,2]), p_scaled_inf_r_tv, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[2,3]), p_scaled_inf_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=scale_miss_class_args)
    
    imshow_plot(scal_miss_class_fig.add_subplot(gs[3,0]), c_scaled_orb_r_rv, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",hide_xticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[3,1]), p_scaled_orb_r_rv, extent=[0,max_r,min_rv,max_rv],hide_xticks=True,hide_yticks=False,y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[3,2]), p_scaled_orb_r_tv, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[3,3]), p_scaled_orb_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=scale_miss_class_args)
    
    imshow_plot(scal_miss_class_fig.add_subplot(gs[4,0]), c_scaled_all_r_rv, extent=[0,max_r,min_rv,max_rv],x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",text="All Misclassified\nScaled",kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[4,1]), p_scaled_all_r_rv, extent=[0,max_r,min_rv,max_rv],x_label="$r/R_{200m}$",hide_yticks=False,y_label="$v_r/v_{200m}$",text="All Misclassified\nScaled",kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[4,2]), p_scaled_all_r_tv, extent=[0,max_r,min_tv,max_tv],x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$",kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[4,3]), p_scaled_all_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],x_label="$v_r/v_{200m}$",hide_yticks=True,kwargs=scale_miss_class_args)
    
    scal_misclas_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=p_min_ptl, vmax=p_max_diff),cmap=magma_cmap), cax=plt.subplot(gs[2:,-1]))
    scal_misclas_color_bar.set_label("Num Incorrect Particles (inf/orb) / Total Particles (inf/orb)")
    
    create_directory(save_location + "/plots/")
    scal_miss_class_fig.savefig(save_location + "/plots/" + title + "scaled_miss_class.png")

    t2 = time.time()
    print("Finished Misclassified Particle Plot in: ", np.round((t2-t1),2), "seconds", np.round(((t2-t1)/60),2), "minutes")

def plot_r_rv_tv_graph(orb_inf, r, rv, tv, correct_orb_inf, title, num_bins, save_location):
    t1 = time.time()
    print("Starting r vs rv vs tv Plot")
    create_directory(save_location + "plots/")
    mpl.rcParams.update({'font.size': 8})
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
    
    phase_plot(inf_fig.add_subplot(gs[0,0]), ml_inf_r, ml_inf_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(inf_fig.add_subplot(gs[0,1]), ml_inf_r, ml_inf_tv, min_ptl, max_ptl, range=[r_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="ML Predictions")
    phase_plot(inf_fig.add_subplot(gs[0,2]), ml_inf_rv, ml_inf_tv, min_ptl, max_ptl, range=[rv_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$")
    phase_plot(inf_fig.add_subplot(gs[1,0]), act_inf_r, act_inf_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(inf_fig.add_subplot(gs[1,1]), act_inf_r, act_inf_tv, min_ptl, max_ptl, range=[r_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="Actual Distribution")
    phase_plot(inf_fig.add_subplot(gs[1,2]), act_inf_rv, act_inf_tv, min_ptl, max_ptl, range=[rv_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$")
    
    inf_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]))
    
    inf_fig.savefig(save_location + "/plots/" + title + "ptls_inf.png")
    
#########################################################################################################################################################
    
    orb_fig = plt.figure()
    orb_fig.suptitle("Orbiting Particles: " + title)
    gs = orb_fig.add_gridspec(2,4,width_ratios = widths, height_ratios = heights)
    
    phase_plot(orb_fig.add_subplot(gs[0,0]), ml_orb_r, ml_orb_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(orb_fig.add_subplot(gs[0,1]), ml_orb_r, ml_orb_tv, min_ptl, max_ptl, range=[r_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="ML Predictions")
    phase_plot(orb_fig.add_subplot(gs[0,2]), ml_orb_rv, ml_orb_tv, min_ptl, max_ptl, range=[rv_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$")
    phase_plot(orb_fig.add_subplot(gs[1,0]), act_orb_r, act_orb_rv, min_ptl, max_ptl, range=[r_range,rv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(orb_fig.add_subplot(gs[1,1]), act_orb_r, act_orb_tv, min_ptl, max_ptl, range=[r_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="Actual Distribution")
    phase_plot(orb_fig.add_subplot(gs[1,2]), act_orb_rv, act_orb_tv, min_ptl, max_ptl, range=[rv_range,tv_range], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$")
    
    orb_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]), pad = 0.1)
    
    orb_fig.savefig(save_location + "/plots/" + title + "ptls_orb.png")    
    
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
    only_r_rv_fig.savefig(save_location + "/plots/" + title + "only_r_rv.png")
    
#########################################################################################################################################################

    err_fig = plt.figure()
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
    
    err_fig.savefig(save_location + "/plots/" + title + "percent_error.png") 

    t2 = time.time()
    print("Finished r vs rv vs tv Plot in: ", np.round((t2-t1),2), "seconds", np.round(((t2-t1)/60),2), "minutes")
    
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
    print(corr_mtrx)
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
    
def graph_acc_by_bin(pred_orb_inf, corr_orb_inf, radius, num_bins, title, plot, save, save_location):
    bin_width = (np.max(radius) - 0) / num_bins
    inf_radius = radius[np.where(corr_orb_inf == 0)]
    orb_radius = radius[np.where(corr_orb_inf == 1)]

    all_accuracy = []
    inf_accuracy = []
    orb_accuracy = []
    bins = []
    start_bin = 0
    for i in range(num_bins):
        bins.append(start_bin)
        finish_bin = start_bin + bin_width
        idx_in_bin = np.where((radius >= start_bin) & (radius < finish_bin))[0]
        if idx_in_bin.shape[0] == 0:
            start_bin = finish_bin
            all_accuracy.append(np.NaN)
            continue
        bin_preds = pred_orb_inf[idx_in_bin]
        bin_corr = corr_orb_inf[idx_in_bin]
        classification = classification_report(bin_corr, bin_preds, output_dict=True, zero_division=0)
        all_accuracy.append(classification["accuracy"])
        
        start_bin = finish_bin
    bins.append(start_bin)    
        
    start_bin = 0
    for j in range(num_bins):
        finish_bin = start_bin + bin_width
        idx_in_bin = np.where((inf_radius >= start_bin) & (inf_radius < finish_bin))[0]
        if idx_in_bin.shape[0] == 0:
            start_bin = finish_bin
            inf_accuracy.append(np.NaN)
            continue
        bin_preds = pred_orb_inf[idx_in_bin]
        bin_corr = corr_orb_inf[idx_in_bin]
        classification = classification_report(bin_corr, bin_preds, output_dict=True, zero_division=0)
        inf_accuracy.append(classification["accuracy"])
        

        start_bin = finish_bin

    start_bin = 0
    for k in range(num_bins):
        finish_bin = start_bin + bin_width
        idx_in_bin = np.where((orb_radius >= start_bin) & (orb_radius < finish_bin))[0]
        if idx_in_bin.shape[0] == 0:
            start_bin = finish_bin
            orb_accuracy.append(np.NaN)
            continue
        bin_preds = pred_orb_inf[idx_in_bin]
        bin_corr = corr_orb_inf[idx_in_bin]
        classification = classification_report(bin_corr, bin_preds, output_dict=True, zero_division=0)
        orb_accuracy.append(classification["accuracy"])
        
        start_bin = finish_bin
    
    
    fig, ax = plt.subplots(2,2, layout="constrained")
    fig.suptitle("Accuracy by Radius for: " + title)
    ax[0,0].stairs(all_accuracy, bins, color = "black", alpha = 0.6, label = "all ptl")    
    ax[0,0].stairs(inf_accuracy, bins, color = "blue", alpha = 0.4, label = "inf ptl")
    ax[0,0].stairs(orb_accuracy, bins, color = "red", alpha = 0.4, label = "orb ptl")
    ax[0,0].set_title("All Density Prfs")
    ax[0,0].set_xlabel("radius $r/R_{200m}$")
    ax[0,0].set_ylabel("Accuracy")
    ax[0,0].set_ylim(-0.1,1.1)
    ax[0,0].legend()
    
    ax[0,1].stairs(all_accuracy, bins, color = "black", label = "all ptl")    
    ax[0,1].set_title("All Mass Profile")
    ax[0,1].set_xlabel("radius $r/R_{200m}$")
    ax[0,1].set_ylabel("Accuracy")
    ax[0,1].set_ylim(-0.1,1.1)
    ax[0,1].legend()
   
    ax[1,0].stairs(inf_accuracy, bins, color = "blue", label = "inf ptl")
    ax[1,0].set_title("Infalling Profile")
    ax[1,0].set_xlabel("radius $r/R_{200m}$")
    ax[1,0].set_ylabel("Accuracy")
    ax[1,0].set_ylim(-0.1,1.1)
    ax[1,0].legend()

    ax[1,1].stairs(orb_accuracy, bins, color = "red", label = "orb ptl")
    ax[1,1].set_title("Orbiting Profile")
    ax[1,1].set_xlabel("radius $r/R_{200m}$")
    ax[1,1].set_ylabel("Accuracy")
    ax[1,1].set_ylim(-0.1,1.1)
    ax[1,1].legend()

    if plot:
        plt.show()
        plt.close()
    if save:
        create_directory(save_location + "error_by_rad_graphs/")
        fig.savefig(save_location + "error_by_rad_graphs/error_by_rad_" + title + ".png")
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

def get_ptl_halo_pos(p_snap, num_plt_snaps, i, sparta_z, low_tjy_loc, high_tjy_loc, low_idxs, high_idxs, halos_pos, halos_vel, halos_r200m):
    curr_snap = (p_snap-num_plt_snaps) + i + 1
    snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(curr_snap) + "/snapshot_" + snap_format.format(curr_snap)
    if os.path.isdir(path_to_snaps + "snapdir_" + snap_format.format(curr_snap)):
        curr_red_shift = readheader(snapshot_path, 'redshift')
        curr_scale_factor = 1/(1+curr_red_shift)
        curr_sparta_snap = np.abs(sparta_z - curr_red_shift).argmin()

        ptls_pos = readsnap(snapshot_path, 'pos', 'dm', suppress=1) * 10**3 * curr_scale_factor
        ptls_vel = readsnap(snapshot_path, 'vel', 'dm', suppress=1)
        
        low_use_ptl_pos = ptls_pos[low_tjy_loc,:]
        low_use_ptl_vel = ptls_vel[low_tjy_loc,:]
        high_use_ptl_pos = ptls_pos[high_tjy_loc,:]
        high_use_ptl_vel = ptls_vel[high_tjy_loc,:]
        
        low_use_halo_pos = halos_pos[low_idxs,curr_sparta_snap]
        low_use_halo_vel = halos_vel[low_idxs,curr_sparta_snap]
        low_use_halo_r200m =  halos_r200m[low_idxs,curr_sparta_snap]
        
        high_use_halo_pos = halos_pos[high_idxs,curr_sparta_snap]
        high_use_halo_vel = halos_vel[high_idxs,curr_sparta_snap]
        high_use_halo_r200m = halos_r200m[high_idxs,curr_sparta_snap]

        return low_use_ptl_pos,low_use_ptl_vel,high_use_ptl_pos,high_use_ptl_vel,low_use_halo_pos,low_use_halo_vel,low_use_halo_r200m,high_use_halo_pos,high_use_halo_vel,high_use_halo_r200m

def update_anim(curr_frame, ax, q, halo_pos, halo_vel, ptl_pos, ptl_vel, radius, snap_hist, num_halo_search, halo_clrs, alphas, p_snap, p_box_size, num_plt_snaps):
    with timed("Frame " + str(curr_frame) + " completed"):
        for i in range(num_halo_search):
            if curr_frame == 1:
                ax.scatter(np.array([]),np.array([]),color=halo_clrs[i], label=("Halo " + str(i)))
            
            ax.quiver(halo_pos[curr_frame,i,0],halo_pos[curr_frame,i,1],halo_pos[curr_frame,i,2],halo_vel[curr_frame,i,0],halo_vel[curr_frame,i,1],halo_vel[curr_frame,i,2], alpha=alphas[curr_frame], color=halo_clrs[i])
            # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            # x = halo_pos[i,j,0] + radius[i,j] * np.cos(u)*np.sin(v)
            # y = halo_pos[i,j,1] + radius[i,j] * np.sin(u)*np.sin(v)
            # z = halo_pos[i,j,2] + radius[i,j] * np.cos(v)

            # ax.plot_wireframe(x, y, z, color=halo_clrs[j])
            # ax.set_box_aspect([1,1,1])
        
        if curr_frame == 1:
            ax.scatter(np.array([]),np.array([]), color="blue", label = "Ptl")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if (p_snap-num_plt_snaps+curr_frame+1) in snap_hist:
            ax.quiver(ptl_pos[curr_frame,0],ptl_pos[curr_frame,1],ptl_pos[curr_frame,2],ptl_vel[curr_frame,0],ptl_vel[curr_frame,1],ptl_vel[curr_frame,2], color="red", alpha=1, label="Orbiting Event Snap:"+str(p_snap-num_plt_snaps+curr_frame+1))
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.quiver(ptl_pos[curr_frame,0],ptl_pos[curr_frame,1],ptl_pos[curr_frame,2],ptl_vel[curr_frame,0],ptl_vel[curr_frame,1],ptl_vel[curr_frame,2], color="blue", alpha=alphas[curr_frame])

        if curr_frame == 0:    
            min_x = np.amin([np.amin(halo_pos[:,:,0]),np.amin(ptl_pos[:,0])])
            max_x = np.amax([np.amax(halo_pos[:,:,0]),np.amin(ptl_pos[:,0])])
            min_y = np.amin([np.amin(halo_pos[:,:,1]),np.amin(ptl_pos[:,1])])
            max_y = np.amax([np.amax(halo_pos[:,:,1]),np.amin(ptl_pos[:,1])])
            min_z = np.amin([np.amin(halo_pos[:,:,2]),np.amin(ptl_pos[:,2])])
            max_z = np.amax([np.amax(halo_pos[:,:,2]),np.amin(ptl_pos[:,2])])
            
            # x_adj = 0.5*(max_x-min_x)
            # y_adj = 0.5*(max_y-min_y)
            # z_adj = 0.5*(max_z-min_z)
            x_adj=0
            y_adj=0
            z_adj=0

            ax.set_xlim(np.max([min_x-x_adj,0]), np.min([(max_x+x_adj),p_box_size]))
            ax.set_ylim(np.max([min_y-y_adj,0]), np.min([(max_y+y_adj),p_box_size]))
            ax.set_zlim(np.max([min_z-z_adj,0]), np.min([(max_z+z_adj),p_box_size]))

        ax.text2D(.01,.95, s=str("Snapshot: " + str(p_snap-num_plt_snaps+curr_frame+1)), ha="left", va="top", transform=ax.transAxes, fontsize="x-large", bbox={"facecolor":'white',"alpha":.9,})
        
    return q,

def anim_ptl_path(red_shift,cosmol,num_halo_search,num_plt_snaps,ptl_props_path,save_path):
    with timed("sparta info load time"):
        with h5py.File(path_to_hdf5_file,"r") as file:      
            p_snap, p_red_shift = find_closest_z(red_shift)
            print("Snapshot number found:", p_snap, "Closest redshift found:", p_red_shift)
            tot_num_halos = file['halos']['id'][:].shape[0]
            use_halos = np.zeros(tot_num_halos)
            use_halos[0]=1
            # num_halos = file
            with timed("SPARTA load:"):
                sparta_output = sparta.load(filename=path_to_hdf5_file, load_halo_data=False, halo_mask = use_halos, log_level= 0)
            all_red_shifts = sparta_output["simulation"]["snap_z"][:]
            p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
            print("corresponding SPARTA snap num:", p_sparta_snap)
            print("check sparta redshift:",all_red_shifts[p_sparta_snap])
            
            # Set constants
            p_snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

            p_scale_factor = 1/(1+p_red_shift)
            p_hubble_constant = cosmol.Hz(p_red_shift) * 0.001 # convert to units km/s/kpc
            sim_box_size = sparta_output["simulation"]["box_size"] #units Mpc/h comoving
            p_box_size = sim_box_size * 10**3 * p_scale_factor #convert to Kpc/h physical

            p_snap_dict = {
                "snap":p_snap,
                "red_shift":p_red_shift,
                "scale_factor": p_scale_factor,
                "hubble_const": p_hubble_constant,
                "box_size": p_box_size,
            }
            halos_pos = file['halos']['position'][:,:,:] * 10**3 * p_scale_factor
            halos_vel = file['halos']['velocity'][:,:,:]
            halos_r200m = file['halos']['R200m'][:]
            tcr_id = file['tcr_ptl']['res_oct']['tracer_id'][:]

            with timed("ptl info load"):
                p_ptls_pid, p_ptls_vel, p_ptls_pos = load_or_pickle_ptl_data(curr_sparta_file, str(p_snap), p_snapshot_path, p_scale_factor)
            
            snap_hist = file['tcr_ptl']['res_oct']['snap_hist'][:]
            sparta_z = sparta_output["simulation"]["snap_z"][:]

    with timed("ptl prop load time"):
        with h5py.File(ptl_props_path, 'a') as all_particle_properties:
            scaled_rad_vel = all_particle_properties["Radial_vel_"][:,0]
            scal_rad = all_particle_properties["Scaled_radii_"][:,0]
            hipids = all_particle_properties["HIPIDS"][:]
            labels = all_particle_properties["Orbit_Infall"][:]

    # conditions
    low_mask = np.logical_and.reduce(((scal_rad>=0.9), (scal_rad<1.05), (labels==1)))
    high_mask = np.logical_and.reduce(((scal_rad>=1.05), (scal_rad<1.2), (labels==1)))
    # find where in original array the array with the conditions applied is max
    print(np.where(scaled_rad_vel==np.max(np.abs(scaled_rad_vel[low_mask])))[0].shape)
    print("radial vel low rad ptl:",scaled_rad_vel[np.where(scaled_rad_vel==np.max(np.abs(scaled_rad_vel[low_mask])))[0][0]])
    print("radial vel high rad ptl:",scaled_rad_vel[np.where(scaled_rad_vel==np.max(np.abs(scaled_rad_vel[high_mask])))[0][0]])
    low_id = depair(hipids[np.where(scaled_rad_vel==np.max(np.abs(scaled_rad_vel[low_mask])))[0][0]])
    high_id = depair(hipids[np.where(scaled_rad_vel==np.max(np.abs(scaled_rad_vel[high_mask])))[0][0]])

    low_tjy_loc = np.where(p_ptls_pid==low_id[0])[0]
    high_tjy_loc = np.where(p_ptls_pid==high_id[0])[0]

    low_snap_hist = snap_hist[np.where(tcr_id==low_id[0])[0]].ravel()
    high_snap_hist = snap_hist[np.where(tcr_id==high_id[0])[0]].ravel()

    print(low_snap_hist,high_snap_hist)

    tree = cKDTree(data = halos_pos[:,p_sparta_snap,:], leafsize = 3, balanced_tree = False, boxsize = p_box_size)
    low_dist, low_idxs = tree.query(p_ptls_pos[low_tjy_loc,:], k=num_halo_search, workers=num_processes)
    high_dist, high_idxs = tree.query(p_ptls_pos[high_tjy_loc,:], k=num_halo_search, workers=num_processes)
    low_idxs = low_idxs[0]
    high_idxs = high_idxs[0]

    all_low_use_ptl_pos = np.zeros((num_plt_snaps,3))
    all_low_use_ptl_vel = np.zeros((num_plt_snaps,3))
    all_high_use_ptl_pos = np.zeros((num_plt_snaps,3))
    all_high_use_ptl_vel = np.zeros((num_plt_snaps,3))
    all_low_use_halo_pos = np.zeros((num_plt_snaps,num_halo_search,3))
    all_low_use_halo_vel = np.zeros((num_plt_snaps,num_halo_search,3))
    all_low_use_halo_r200m = np.zeros((num_plt_snaps,num_halo_search))
    all_high_use_halo_pos = np.zeros((num_plt_snaps,num_halo_search,3))
    all_high_use_halo_vel = np.zeros((num_plt_snaps,num_halo_search,3))
    all_high_use_halo_r200m = np.zeros((num_plt_snaps,num_halo_search))

    with mp.Pool(processes=num_processes) as p:
        # halo position, halo r200m, if comparison snap, want mass?, want indices?
        low_use_ptl_pos,low_use_ptl_vel,high_use_ptl_pos,high_use_ptl_vel,low_use_halo_pos,low_use_halo_vel,low_use_halo_r200m,high_use_halo_pos,high_use_halo_vel,high_use_halo_r200m = zip(*p.starmap(get_ptl_halo_pos, 
                                                zip(repeat(p_snap), repeat(num_plt_snaps), np.arange(0,num_plt_snaps), repeat(sparta_z), repeat(low_tjy_loc),
                                                    repeat(high_tjy_loc), repeat(low_idxs), repeat(high_idxs), repeat(halos_pos), repeat(halos_vel), repeat(halos_r200m)),
                                                    chunksize=curr_chunk_size))
    p.close()
    p.join() 
    
    all_low_use_ptl_pos = np.stack(low_use_ptl_pos, axis=0).squeeze()
    all_low_use_ptl_vel = np.stack(low_use_ptl_vel, axis=0).squeeze()
    all_high_use_ptl_pos = np.stack(high_use_ptl_pos, axis=0).squeeze()
    all_high_use_ptl_vel = np.stack(high_use_ptl_vel, axis=0).squeeze()
    all_low_use_halo_pos = np.stack(low_use_halo_pos, axis=0)
    all_low_use_halo_vel = np.stack(low_use_halo_vel, axis=0)
    all_low_use_halo_r200m = np.stack(low_use_halo_r200m, axis=0)
    all_high_use_halo_pos = np.stack(high_use_halo_pos, axis=0)
    all_high_use_halo_vel = np.stack(high_use_halo_vel, axis=0)
    all_high_use_halo_r200m = np.stack(high_use_halo_r200m, axis=0)
    print(all_low_use_ptl_pos.shape)
    print(all_low_use_halo_pos.shape)

    halo_clrs = plt.cm.viridis(np.linspace(0, 1, num_halo_search))
    alphas = np.logspace(np.log10(0.1),np.log10(1),num_plt_snaps)

    low_fig = plt.figure()
    low_ax = low_fig.add_subplot(projection='3d')
    q = low_ax.quiver([], [], [], [], [], [], color='r')

    high_fig = plt.figure()
    high_ax = high_fig.add_subplot(projection='3d')
    q = high_ax.quiver([], [], [], [], [], [], color='r')

    fps = 3
                                                                
    ani = FuncAnimation(low_fig, update_anim, frames=num_plt_snaps, fargs=(low_ax, q, all_low_use_halo_pos, all_low_use_halo_vel, all_low_use_ptl_pos, all_low_use_ptl_vel, all_low_use_halo_r200m, low_snap_hist, num_halo_search, halo_clrs,alphas,p_snap,p_box_size,num_plt_snaps), interval=200, blit=True)
    ani.save(save_path+"low_ptl_track.mp4", writer='ffmpeg', fps=fps)

    ani = FuncAnimation(high_fig, update_anim, frames=num_plt_snaps, fargs=(high_ax, q, all_high_use_halo_pos, all_high_use_halo_vel, all_high_use_ptl_pos, all_high_use_ptl_vel, all_high_use_halo_r200m, high_snap_hist, num_halo_search, halo_clrs,alphas,p_snap,p_box_size,num_plt_snaps), interval=200, blit=True)
    ani.save(save_path+"high_ptl_track.mp4", writer='ffmpeg', fps=fps)

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
    
    