import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
mpl.use('agg')
from utils.calculation_functions import calculate_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report
from colossus.lss.peaks import peakHeight
import matplotlib.colors as colors
import multiprocessing as mp
from itertools import repeat
from sparta_tools import sparta # type: ignore
import os
from contextlib import contextmanager
import sys
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import json

from utils.data_and_loading_functions import check_pickle_exist_gadget, create_directory, find_closest_z, load_or_pickle_ptl_data, timed, parse_ranges,create_nu_string
from utils.calculation_functions import calc_v200m, calculate_density, create_mass_prf


plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

num_processes = mp.cpu_count()
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
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
lin_rticks = json.loads(config.get("XGBOOST","lin_rticks"))
plt_nu_splits = config["XGBOOST"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)
plt_nu_string = create_nu_string(plt_nu_splits)

sys.path.insert(1, path_to_pygadgetreader)  
from pygadgetreader import readsnap, readheader # type: ignore

def comb_prf(prf, num_halo, dtype):
    if num_halo > 1:
        prf = np.stack(prf, axis=0)
    else:
        prf = np.asarray(prf)
        prf = np.reshape(prf, (1,prf.size))
    
    prf = prf.astype(dtype)

    return prf

def adj_dens_prf(prf, nu_fltr = None, use_med = True):
    if nu_fltr is not None:
        prf = prf[nu_fltr]
        # for each bin checking how many halos have particles there
        # if there are less than half the total number of halos then just treat that bin as having 0
        for i in range(calc_mass_prf_orb.shape[1]):
            if np.where(calc_mass_prf_orb[:,i] > 0)[0].shape[0] < min_disp_halos:
                calc_mass_prf_orb[:,i] = act_mass_prf_orb[:,i] = med_calc_mass_prf_orb[i] = med_act_mass_prf_orb[i] = avg_calc_mass_prf_orb[i] = avg_act_mass_prf_orb[i] = np.nan
            if np.where(calc_mass_prf_inf[:,i] > 0)[0].shape[0] < min_disp_halos:
                calc_mass_prf_inf[:,i] = act_mass_prf_inf[:,i] = med_calc_mass_prf_inf[i] = med_act_mass_prf_inf[i] = avg_calc_mass_prf_inf[i] = avg_act_mass_prf_inf[i] = np.nan
            if np.where(calc_mass_prf_all[:,i] > 0)[0].shape[0] < min_disp_halos:
                calc_mass_prf_all[:,i] = act_mass_prf_all[:,i] = med_calc_mass_prf_all[i] = med_act_mass_prf_all[i] = avg_calc_mass_prf_all[i] = avg_act_mass_prf_all[i] = np.nan
            if np.where(calc_dens_prf_orb[:,i] > 0)[0].shape[0] < min_disp_halos:
                calc_dens_prf_orb[:,i] = act_dens_prf_orb[:,i] = med_calc_dens_prf_orb[i] = med_act_dens_prf_orb[i] = avg_calc_dens_prf_orb[i] = avg_act_dens_prf_orb[i] = np.nan
            if np.where(calc_dens_prf_inf[:,i] > 0)[0].shape[0] < min_disp_halos:
                calc_dens_prf_inf[:,i] = act_dens_prf_inf[:,i] = med_calc_dens_prf_inf[i] = med_act_dens_prf_inf[i] = avg_calc_dens_prf_inf[i] = avg_act_dens_prf_inf[i] = np.nan
            if np.where(calc_dens_prf_all[:,i] > 0)[0].shape[0] < min_disp_halos:
                calc_dens_prf_all[:,i] = act_dens_prf_all[:,i] = med_calc_dens_prf_all[i] = med_act_dens_prf_all[i] = avg_calc_dens_prf_all[i] = avg_act_dens_prf_all[i] = np.nan
        
        # Get the ratio of the calculated profile with the actual profile
        with np.errstate(divide='ignore', invalid='ignore'):
            med_all_dens_ratio = np.divide(calc_dens_prf_all,act_dens_prf_all) - 1
            med_inf_dens_ratio = np.divide(calc_dens_prf_inf,act_dens_prf_inf) - 1
            med_orb_dens_ratio = np.divide(calc_dens_prf_orb,act_dens_prf_orb) - 1
            avg_all_dens_ratio = np.divide(avg_calc_dens_prf_all,act_dens_prf_all) - 1
            avg_inf_dens_ratio = np.divide(avg_calc_dens_prf_inf,act_dens_prf_inf) - 1
            avg_orb_dens_ratio = np.divide(avg_calc_dens_prf_orb,act_dens_prf_orb) - 1

        med_inf_dens_ratio[np.isinf(med_inf_dens_ratio)] = np.nan
        avg_inf_dens_ratio[np.isinf(avg_inf_dens_ratio)] = np.nan
        
        # Same for actual profiles    
        med_upper_orb_dens_ratio = np.nanpercentile(med_orb_dens_ratio, q=84.1, axis=0)
        med_lower_orb_dens_ratio = np.nanpercentile(med_orb_dens_ratio, q=15.9, axis=0)
        med_upper_inf_dens_ratio = np.nanpercentile(med_inf_dens_ratio, q=84.1, axis=0)
        med_lower_inf_dens_ratio = np.nanpercentile(med_inf_dens_ratio, q=15.9, axis=0)
        med_upper_all_dens_ratio = np.nanpercentile(med_all_dens_ratio, q=84.1, axis=0)
        med_lower_all_dens_ratio = np.nanpercentile(med_all_dens_ratio, q=15.9, axis=0)
        
        avg_upper_orb_dens_ratio = np.nanpercentile(avg_orb_dens_ratio, q=84.1, axis=0)
        avg_lower_orb_dens_ratio = np.nanpercentile(avg_orb_dens_ratio, q=15.9, axis=0)
        avg_upper_inf_dens_ratio = np.nanpercentile(avg_inf_dens_ratio, q=84.1, axis=0)
        avg_lower_inf_dens_ratio = np.nanpercentile(avg_inf_dens_ratio, q=15.9, axis=0)
        avg_upper_all_dens_ratio = np.nanpercentile(avg_all_dens_ratio, q=84.1, axis=0)
        avg_lower_all_dens_ratio = np.nanpercentile(avg_all_dens_ratio, q=15.9, axis=0)


        # Take the median value of the ratios
        med_all_ratio = np.nanmedian(med_all_dens_ratio, axis=0)
        med_inf_ratio = np.nanmedian(med_inf_dens_ratio, axis=0)
        med_orb_ratio = np.nanmedian(med_orb_dens_ratio, axis=0)

        avg_all_ratio = np.nanmedian(avg_all_dens_ratio, axis=0)
        avg_inf_ratio = np.nanmedian(avg_inf_dens_ratio, axis=0)
        avg_orb_ratio = np.nanmedian(avg_orb_dens_ratio, axis=0)

# all_z only needs to be passed if split_by_nu is passed
def compare_density_prf(splits, radii, halo_first, halo_n, act_mass_prf_all, act_mass_prf_orb, mass, orbit_assn, prf_bins, title, save_location, use_mp = False, use_med = True, split_by_nu = False, all_z = []):
    # Shape of profiles should be (num halo,num bins)
    # EX: 10 halos, 80 bins (10,80)

    with timed("Density Profile Plot"):        
        act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb
        tot_num_halos = halo_first.shape[0]
        if tot_num_halos > 5:
            min_disp_halos = int(np.ceil(0.3 * tot_num_halos))
        else:
            min_disp_halos = 0
        
        med_act_mass_prf_all = med_act_mass_prf_orb = med_act_mass_prf_inf = med_act_dens_prf_all = med_act_dens_prf_orb = med_act_dens_prf_inf = 0
        
        med_calc_mass_prf_all = med_calc_mass_prf_orb = med_calc_mass_prf_inf = med_calc_dens_prf_all = med_calc_dens_prf_orb = med_calc_dens_prf_inf = 0
        
        avg_act_mass_prf_all = avg_act_mass_prf_orb = avg_act_mass_prf_inf = avg_act_dens_prf_all = avg_act_dens_prf_orb = avg_act_dens_prf_inf = 0
        
        avg_calc_mass_prf_all = avg_calc_mass_prf_orb = avg_calc_mass_prf_inf = avg_calc_dens_prf_all = avg_calc_dens_prf_orb = avg_calc_dens_prf_inf = 0
        
        calc_mass_prf_orb_lst = []
        calc_mass_prf_inf_lst = []
        calc_mass_prf_all_lst = []
        calc_dens_prf_orb_lst = []
        calc_dens_prf_inf_lst = []
        calc_dens_prf_all_lst = []   
        calc_nu_lst = []  

        # Get density profiles by dividing the mass profiles by the volume of each bin
        act_dens_prf_all = calculate_density(act_mass_prf_all, prf_bins[1:])
        act_dens_prf_orb = calculate_density(act_mass_prf_orb, prf_bins[1:])
        act_dens_prf_inf = calculate_density(act_mass_prf_inf, prf_bins[1:])

        # Loop through each simulation's halos
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
                    calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_m200m = zip(*p.starmap(create_mass_prf, 
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
                calc_m200m = []
                
                for j in range(curr_num_halos):
                    halo_mass_prf_all, halo_mass_prf_orb, halo_mass_prf_inf, m200m = create_mass_prf(radii[halo_first[splits[i]+j]:halo_first[splits[i]+j]+halo_n[splits[i]+j]], orbit_assn[halo_first[splits[i]+j]:halo_first[splits[i]+j]+halo_n[splits[i]+j]], prf_bins,mass[i])
                    calc_mass_prf_orb.append(np.array(halo_mass_prf_orb))
                    calc_mass_prf_inf.append(np.array(halo_mass_prf_inf))
                    calc_mass_prf_all.append(np.array(halo_mass_prf_all))
                    calc_m200m.append(np.array(m200m))
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
            calc_nu_lst.append(peakHeight(np.array(calc_m200m),all_z[i]))

        calc_mass_prf_orb = np.vstack(calc_mass_prf_orb_lst)
        calc_mass_prf_inf = np.vstack(calc_mass_prf_inf_lst)
        calc_mass_prf_all = np.vstack(calc_mass_prf_all_lst)
        calc_dens_prf_orb = np.vstack(calc_dens_prf_orb_lst)
        calc_dens_prf_inf = np.vstack(calc_dens_prf_inf_lst)
        calc_dens_prf_all = np.vstack(calc_dens_prf_all_lst)
        calc_nus = np.vstack(calc_nu_lst) 
    
        if split_by_nu:
            for nu_split in plt_nu_splits:
                print(nu_split)
                nu_fltr = np.where((calc_nus > nu_split[0])&(calc_nus<nu_split[1]))[0]
                adj_dens_prf(calc_mass_prf_orb,nu_fltr)

        middle_bins = (prf_bins[1:] + prf_bins[:-1]) / 2

        widths = [1]
        heights = [1,0.5]
            
        fig = plt.figure(constrained_layout=True,figsize=(8,10))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
        
        titlefntsize=22
        axisfntsize=20
        tickfntsize=16
        legendfntsize=16
        fill_alpha = 0.2
        
        # Get rid of the jump from 0 to the first occupied bin by setting them to nan
        med_calc_mass_prf_all[med_calc_mass_prf_all == 0] = med_calc_mass_prf_orb[med_calc_mass_prf_orb == 0] = med_calc_mass_prf_inf[med_calc_mass_prf_inf == 0] = med_calc_dens_prf_all[med_calc_dens_prf_all == 0] = med_calc_dens_prf_orb[med_calc_dens_prf_orb == 0] = med_calc_dens_prf_inf[med_calc_dens_prf_inf == 0] = np.nan
        med_act_mass_prf_all[med_act_mass_prf_all == 0] = med_act_mass_prf_orb[med_act_mass_prf_orb == 0] = med_act_mass_prf_inf[med_act_mass_prf_inf == 0] = med_act_dens_prf_all[med_act_dens_prf_all == 0] = med_act_dens_prf_orb[med_act_dens_prf_orb == 0] = med_act_dens_prf_inf[med_act_dens_prf_inf == 0] = np.nan
        
        avg_calc_mass_prf_all[med_calc_mass_prf_all == 0] = avg_calc_mass_prf_orb[med_calc_mass_prf_orb == 0] = avg_calc_mass_prf_inf[med_calc_mass_prf_inf == 0] = avg_calc_dens_prf_all[med_calc_dens_prf_all == 0] = avg_calc_dens_prf_orb[med_calc_dens_prf_orb == 0] = avg_calc_dens_prf_inf[med_calc_dens_prf_inf == 0] = np.nan
        avg_act_mass_prf_all[med_act_mass_prf_all == 0] = avg_act_mass_prf_orb[med_act_mass_prf_orb == 0] = avg_act_mass_prf_inf[med_act_mass_prf_inf == 0] = avg_act_dens_prf_all[med_act_dens_prf_all == 0] = avg_act_dens_prf_orb[med_act_dens_prf_orb == 0] = avg_act_dens_prf_inf[med_act_dens_prf_inf == 0] = np.nan
        
        ax_0 = fig.add_subplot(gs[0])
        ax_1 = fig.add_subplot(gs[1],sharex=ax_0)
        
        invis_calc, = ax_0.plot([0], [0], color='black', linestyle='-')
        invis_act, = ax_0.plot([0], [0], color='black', linestyle='--')
        
        all_lb, = ax_0.plot(middle_bins, med_calc_dens_prf_all, 'r-', label = "All")
        orb_lb, = ax_0.plot(middle_bins, med_calc_dens_prf_orb, 'b-', label = "Orbiting")
        inf_lb, = ax_0.plot(middle_bins, med_calc_dens_prf_inf, 'g-', label = "Infalling")
        ax_0.plot(middle_bins, med_act_dens_prf_all, 'r--')
        ax_0.plot(middle_bins, med_act_dens_prf_orb, 'b--')
        ax_0.plot(middle_bins, med_act_dens_prf_inf, 'g--')

        ax_0.set_ylabel(r"$\rho (M_\odot \mathrm{kpc}^{-3})$", fontsize=axisfntsize)
        ax_0.set_xscale("log")
        ax_0.set_yscale("log")
        ax_0.set_xlim(0.05,np.max(lin_rticks))
        ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        fig.legend([(invis_calc, invis_act),all_lb,orb_lb,inf_lb], ['Predicted, Actual','All','Orbiting','Infalling'], numpoints=1,handlelength=3,handler_map={tuple: HandlerTuple(ndivide=None)},frameon=False,fontsize=legendfntsize)

        ax_1.plot(middle_bins, med_all_ratio, 'r')
        ax_1.plot(middle_bins, med_orb_ratio, 'b')
        ax_1.plot(middle_bins, med_inf_ratio, 'g')
        
        if tot_num_halos > 5:
            ax_1.fill_between(middle_bins, med_lower_all_dens_ratio, med_upper_all_dens_ratio, color='r', alpha=fill_alpha)
            ax_1.fill_between(middle_bins, med_lower_inf_dens_ratio, med_upper_inf_dens_ratio, color='g', alpha=fill_alpha)
            ax_1.fill_between(middle_bins, med_lower_orb_dens_ratio, med_upper_orb_dens_ratio, color='b', alpha=fill_alpha)    
            
        ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        ax_1.set_ylabel(r"$\frac{\rho_{pred}}{\rho_{act}} - 1$", fontsize=axisfntsize)
        
        ax_1.set_xlim(0.05,np.max(lin_rticks))
        ax_1.set_ylim(bottom=-0.3,top=0.3)
        ax_1.set_xscale("log")
        tick_locs = lin_rticks
        if 0 in lin_rticks:
            tick_locs.remove(0)
        strng_ticks = list(map(str, tick_locs))
        
        ax_1.set_xticks(tick_locs,strng_ticks)  
        ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)

        fig.savefig(save_location + title + "med_dens_prfl_rat.png",bbox_inches='tight')
        
        ######################################################################################################################################################
        fig = plt.figure(constrained_layout=True,figsize=(8,10))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
        
        ax_0 = fig.add_subplot(gs[0])
        ax_1 = fig.add_subplot(gs[1])
        
        invis_calc, = ax_0.plot([0], [0], color='black', linestyle='-')
        invis_act, = ax_0.plot([0], [0], color='black', linestyle='--')
        
        all_lb, = ax_0.plot(middle_bins, avg_calc_dens_prf_all, 'r-', label = "All")
        orb_lb, = ax_0.plot(middle_bins, avg_calc_dens_prf_orb, 'b-', label = "Orbiting")
        inf_lb, = ax_0.plot(middle_bins, avg_calc_dens_prf_inf, 'g-', label = "Infalling")
        ax_0.plot(middle_bins, avg_act_dens_prf_all, 'r--')
        ax_0.plot(middle_bins, avg_act_dens_prf_orb, 'b--')
        ax_0.plot(middle_bins, avg_act_dens_prf_inf, 'g--')

        ax_0.set_ylabel(r"$\rho (M_\odot \mathrm{kpc}^{-3})$", fontsize=axisfntsize)
        ax_0.set_xscale("log")
        ax_0.set_yscale("log")
        ax_0.set_xlim(0.05)
        ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        fig.legend([(invis_calc, invis_act),all_lb,orb_lb,inf_lb], ['Predicted, Actual','All','Orbiting','Infalling'], numpoints=1,handlelength=3,handler_map={tuple: HandlerTuple(ndivide=None)},loc='outside left upper',bbox_to_anchor=(1, 1),frameon=False,fontsize=legendfntsize)

        ax_1.plot(middle_bins, avg_all_ratio, 'r')
        ax_1.plot(middle_bins, avg_orb_ratio, 'b')
        ax_1.plot(middle_bins, avg_inf_ratio, 'g')
        
        if tot_num_halos > 5:
            ax_1.fill_between(middle_bins, avg_lower_all_dens_ratio, avg_upper_all_dens_ratio, color='r', alpha=fill_alpha)
            ax_1.fill_between(middle_bins, avg_lower_inf_dens_ratio, avg_upper_inf_dens_ratio, color='g', alpha=fill_alpha)
            ax_1.fill_between(middle_bins, avg_lower_orb_dens_ratio, avg_upper_orb_dens_ratio, color='b', alpha=fill_alpha)    
            
        ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        ax_1.set_ylabel(r"$\frac{\rho_{pred}}{\rho_{act}} - 1$", fontsize=axisfntsize)

        ax_1.set_xlim(0.05)
        ax_1.set_ylim(top=0.3)
        ax_1.set_xscale("log")
        tick_locs = lin_rticks
        if 0 in lin_rticks:
            tick_locs.remove(0)
        strng_ticks = list(map(str, tick_locs))

        ax_1.set_xticks(tick_locs,strng_ticks)  
        ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        fig.savefig(save_location + title + "avg_dens_prfl_rat.png",bbox_inches='tight')

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
    ax.legend(frameon=False)
    
    return ax.plot(hubble_vel[:,0], hubble_vel[:,1], color = "purple", alpha = 0.5, linestyle = "dashed", label = r"Hubble Flow")
  
def split_orb_inf(data, labels):
    infall = data[np.where(labels == 0)[0]]
    orbit = data[np.where(labels == 1)[0]]
    return infall, orbit

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
    ax[0].legend(frameon=False)
    
    ax[1].scatter(pred_inf_ptls[:,0], pred_inf_ptls[:,1], c='g', label = "Infalling Particles")
    ax[1].scatter(pred_orb_ptls[:,0], pred_orb_ptls[:,1], c='b', label = "Orbiting Particles")
    ax[1].set_title("Predicted Distribution of Orbiting/Infalling Particles")
    ax[1].set_xlabel("X position (kpc)")
    ax[1].set_ylabel("Y position (kpc)")
    ax[1].legend(frameon=False)
    fig.savefig(save_path + "plot_of_halo_both_dist.png")

    fig, ax = plt.subplots(1)
    ax.scatter(corr_class[:,0], corr_class[:,1], c='g', label = "Correctly Labeled")
    ax.scatter(inc_class[:,0], inc_class[:,1], c='r', label = "Incorrectly Labeled")
    ax.set_title("Predicted Distribution of Orbiting/Infalling Particles")
    ax.set_xlabel("X position (kpc)")
    ax.set_ylabel("Y position (kpc)")
    ax.legend(frameon=False)
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
    ax1.legend(fontsize=axis_fontsize,frameon=False)

    ax2 = fig.add_subplot(132,projection='3d')
    ax2.scatter(ptl_pos[pred_inf,0],ptl_pos[pred_inf,1],ptl_pos[pred_inf,2],c='orange', alpha=0.1)
    ax2.scatter(ptl_pos[pred_orb,0],ptl_pos[pred_orb,1],ptl_pos[pred_orb,2],c='b', alpha=0.1)
    ax2.set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_ylabel("Y position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_zlabel("Z position (kpc/h)",fontsize=axis_fontsize)
    ax2.set_title("Model Predicted Labels", fontsize=title_fontsize)
    ax2.scatter([],[],[],c="orange",label="Infalling Particles")
    ax2.scatter([],[],[],c="b",label="Orbiting Particles")
    ax2.legend(fontsize=axis_fontsize,frameon=False)

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
    ax3.legend(fontsize=axis_fontsize,frameon=False)

    fig.subplots_adjust(wspace=0.05)
    
    fig.savefig("/home/zvladimi/MLOIS/Random_figures/3d_one_halo_all.png")

    fig, ax = plt.subplots(1, 3,figsize=(30,10))
    
    alpha = 0.25

    ax[0].scatter(ptl_pos[real_inf_slice,0],ptl_pos[real_inf_slice,1],c='orange', alpha = alpha, label="Inalling ptls")
    ax[0].scatter(ptl_pos[real_orb_slice,0],ptl_pos[real_orb_slice,1],c='b', alpha = alpha, label="Orbiting ptls")
    ax[0].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[0].set_ylabel("Y position (kpc/h)",fontsize=axis_fontsize)
    ax[0].set_title("Particles Labeled by SPARTA",fontsize=title_fontsize)
    ax[0].legend(fontsize=axis_fontsize,frameon=False)
    
    ax[1].scatter(ptl_pos[pred_inf_slice,0],ptl_pos[pred_inf_slice,1],c='orange', alpha = alpha, label="Predicted Inalling ptls")
    ax[1].scatter(ptl_pos[pred_orb_slice,0],ptl_pos[pred_orb_slice,1],c='b', alpha = alpha, label="Predicted Orbiting ptls")
    ax[1].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[1].set_title("Particles Labeled by ML Model",fontsize=title_fontsize)
    ax[1].tick_params(axis='y', which='both',left=False,labelleft=False)
    ax[1].legend(fontsize=axis_fontsize,frameon=False)
    
    ax[2].scatter(ptl_pos[inc_orb_slice,0],ptl_pos[inc_orb_slice,1],c='r', marker='x', label="Pred: Inalling \n Actual: Orbiting")
    ax[2].scatter(ptl_pos[inc_inf_slice,0],ptl_pos[inc_inf_slice,1],c='r', marker='+', label="Pred: Orbiting \n Actual: Infalling")
    ax[2].set_xlabel("X position (kpc/h)",fontsize=axis_fontsize)
    ax[2].set_title("Incorrectly Labeled Particles",fontsize=title_fontsize)
    ax[2].tick_params(axis='y', which='both',left=False,labelleft=False)
    ax[2].legend(fontsize=axis_fontsize,frameon=False)
    
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
    ax[0].legend(frameon=False)
    
    ax[1].bar(index, log_orb_cnt, bar_width, label='Orbiting',log=True)
    ax[1].bar(index + bar_width, log_inf_cnt, bar_width, label='Infalling',log=True)
    ax[1].set_xlabel('Radius Bins')
    ax[1].set_title('Number of Orbiting and Infalling Particles by Radius Bin')
    ax[1].set_xticks(index + bar_width / 2)
    ax[1].set_xticklabels([f'{log_bins[i]:.2f}-{log_bins[i + 1]:.2f}' for i in range(num_bins)],rotation=90)
    ax[1].legend(frameon=False)
    
    fig.savefig(save_path + "orb_inf_dist.png",bbox_inches="tight")

