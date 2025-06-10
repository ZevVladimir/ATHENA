import numpy as np
import os
import h5py
import warnings
import multiprocessing as mp
from itertools import repeat
import matplotlib.pyplot as plt
from colossus.lss.peaks import peakHeight
from colossus.halo.mass_so import M_to_R
from matplotlib.legend_handler import HandlerTuple

from .util_fxns import load_SPARTA_data, load_pickle, load_config, get_comp_snap_info, timed, load_sparta_mass_prf, get_past_z
from .calc_fxns import calc_rho, calc_mass_acc_rate, calc_tdyn
from .ML_fxns import split_sparta_hdf5_name
from .util_fxns import parse_ranges, set_cosmology

##################################################################################################################
# LOAD CONFIG PARAMETERS
config_dict = load_config(os.getcwd() + "/config.ini")
rand_seed = config_dict["MISC"]["random_seed"]
curr_sparta_file = config_dict["SPARTA_DATA"]["curr_sparta_file"]
debug_indiv_dens_prf = config_dict["MISC"]["debug_indiv_dens_prf"]
save_intermediate_data = config_dict["MISC"]["save_intermediate_data"]

snap_path = config_dict["SNAP_DATA"]["snap_path"]

SPARTA_output_path = config_dict["SPARTA_DATA"]["sparta_output_path"]
pickled_path = config_dict["PATHS"]["pickled_path"]
ML_dset_path = config_dict["PATHS"]["ml_dset_path"]
debug_plt_path = config_dict["PATHS"]["debug_plt_path"]

plt_nu_splits = config_dict["EVAL_MODEL"]["plt_nu_splits"]
plt_nu_splits = parse_ranges(plt_nu_splits)

plt_macc_splits = config_dict["EVAL_MODEL"]["plt_macc_splits"]
plt_macc_splits = parse_ranges(plt_macc_splits)
min_halo_nu_bin = config_dict["EVAL_MODEL"]["min_halo_nu_bin"]

# Get the mass of the next bin of a profile
def update_mass_prf(calc_prf, radii, idx, start_bin, end_bin, mass):
    radii_within_range = np.where((radii >= start_bin) & (radii < end_bin))[0]
    
    # If there are particles in this bin and its not the first bin
    # Then add the mass of prior bin to the mass of this bin
    if radii_within_range.size != 0 and idx != 0:
        calc_prf[idx] = calc_prf[idx - 1] + radii_within_range.size * mass
    # If there are particles in this bin and its  the first bin
    # Then simply the mass of this bin
    elif radii_within_range.size != 0 and idx == 0:
        calc_prf[idx] = radii_within_range.size * mass
    # If there are  no particles in this bin and its not the first bin
    # Then use the mass of prior bin 
    elif radii_within_range.size == 0 and idx != 0:
        calc_prf[idx] = calc_prf[idx-1]
    # If there are  no particles in this bin and its the first bin
    # Then use the mass of this bin 
    else:
        calc_prf[idx] = calc_prf[idx-1]
    
    return calc_prf

# Create a mass profile from particle information
def create_mass_prf(radii, orbit_assn, prf_bins, mass):  
    # Create bins for the density profile calculation
    num_prf_bins = prf_bins.shape[0] - 1

    calc_mass_prf_orb = np.zeros(num_prf_bins)
    calc_mass_prf_inf = np.zeros(num_prf_bins)
    calc_mass_prf_all = np.zeros(num_prf_bins)

    # Can adjust this to cut out halos that don't have enough particles within R200m
    # Anything at 200 or less (shouldn't) doesn't do anything as these halos should already be filtered out when generating the datasets
    min_ptl = 200
    ptl_in_r200m = np.where(radii <= 1)[0].size
    if ptl_in_r200m < min_ptl:
        calc_mass_prf_orb[:]=np.nan
        calc_mass_prf_inf[:]=np.nan
        calc_mass_prf_all[:]=np.nan
        m200m = np.nan
    else:
        # determine which radii correspond to orbiting and which to infalling
        orbit_radii = radii[np.where(orbit_assn == 1)[0]]
        infall_radii = radii[np.where(orbit_assn == 0)[0]]

        # loop through each bin's radii range and get the mass of each type of particle
        for i in range(num_prf_bins):
            start_bin = prf_bins[i]
            end_bin = prf_bins[i+1]          

            calc_mass_prf_all  = update_mass_prf(calc_mass_prf_all, radii, i, start_bin, end_bin, mass)    
            calc_mass_prf_orb = update_mass_prf(calc_mass_prf_orb, orbit_radii, i, start_bin, end_bin, mass)      
            calc_mass_prf_inf = update_mass_prf(calc_mass_prf_inf, infall_radii, i, start_bin, end_bin, mass)      
    
        m200m = ptl_in_r200m * mass
    
    return calc_mass_prf_all,calc_mass_prf_orb, calc_mass_prf_inf, m200m

# Used to combine the profiles that are output from multiprocessing
def comb_prf(prf, num_halo, dtype):
    if num_halo > 1:
        prf = np.stack(prf, axis=0)
    else:
        prf = np.asarray(prf)
        prf = np.reshape(prf, (1,prf.size))
    
    prf = prf.astype(dtype)

    return prf

# Apply two filters:
# 1. For each bin checking how many halos have particles in that bin and if there are less than the desired number of halos then treat that bin as having no halos (Reduces noise in density profile plots)
# 2. (If desired) will filter by the supplied nu (peak height) range
def filter_prf(calc_prf, act_prf, min_disp_halos, nu_fltr = None):
    for i in range(calc_prf.shape[1]):
        if np.where(calc_prf[:,i]>0)[0].shape[0] < min_disp_halos:
            calc_prf[:,i] = np.nan
            act_prf[:,i] = np.nan
            
    if nu_fltr is not None:
        calc_prf = calc_prf[nu_fltr,:]
        act_prf = act_prf[nu_fltr,:]
        
    return calc_prf, act_prf        

# If less than some percentage of halos are non-zero do not plot
# prf should be in shape (n_halo, n_bin)
def clean_prf(prf_1, prf_2, frac=0.5):
    for i in range(prf_1.shape[1]):
        nonzero_frac_1 = np.count_nonzero(prf_1[:, i]) / prf_1.shape[0]
        nonzero_frac_2 = np.count_nonzero(prf_2[:, i]) / prf_2.shape[0]

        if (nonzero_frac_1 < frac) or (nonzero_frac_2 < frac):    
            prf_1[:, i] = np.nan 
            prf_2[:, i] = np.nan
    return prf_1, prf_2

def compute_prfs_info(calc_prf, act_prf, prf_func=None):
    if prf_func is not None:
        clean_prf(calc_prf, act_prf)
        func_calc = prf_func(calc_prf, axis=0)
        func_act = prf_func(act_prf, axis=0)
        mid_ratio = (func_calc / func_act) - 1
    else:
        func_calc = calc_prf
        func_act = act_prf
        mid_ratio = (func_calc / func_act) - 1
    all_ratios = (calc_prf / act_prf) - 1
    return func_calc, func_act, mid_ratio, all_ratios

# Profiles should be a list [calc_prf,act_prf]
# You can either use the median plots with use_med=True or the average with use_med=False
def compare_prfs(all_prfs, orb_prfs, inf_prfs, bins, lin_rticks, save_location, title, prf_func=np.nanmedian):       
    with timed("Compare Profiles"):     
        # Parameters to tune sizes of plots and fonts
        widths = [1]
        heights = [1,0.5]
        axisfntsize=12
        tickfntsize=10
        legendfntsize=10
            
        fig = plt.figure(constrained_layout=True,figsize=(10,6))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
        
        ax_0 = fig.add_subplot(gs[0])
        ax_1 = fig.add_subplot(gs[1],sharex=ax_0)
        
        invis_calc, = ax_0.plot([0], [0], color='black', linestyle='-')
        invis_act, = ax_0.plot([0], [0], color='black', linestyle='--')

        calc_all_prfs, act_all_prfs, mid_ratio_all_prf, all_ratio_all_prf = compute_prfs_info(all_prfs[0],all_prfs[1],prf_func)
        calc_orb_prfs, act_orb_prfs, mid_ratio_orb_prf, all_ratio_orb_prf = compute_prfs_info(orb_prfs[0],orb_prfs[1],prf_func)
        calc_inf_prfs, act_inf_prfs, mid_ratio_inf_prf, all_ratio_inf_prf = compute_prfs_info(inf_prfs[0],inf_prfs[1],prf_func)
                    
        # Plot the calculated profiles
        all_lb, = ax_0.plot(bins, calc_all_prfs, 'r-', label = "All")
        orb_lb, = ax_0.plot(bins, calc_orb_prfs, 'b-', label = "Orbiting")
        inf_lb, = ax_0.plot(bins, calc_inf_prfs, 'g-', label = "Infalling")
        
        # Plot the SPARTA (actual) profiles 
        ax_0.plot(bins, act_all_prfs, 'r--')
        ax_0.plot(bins, act_orb_prfs, 'b--')
        ax_0.plot(bins, act_inf_prfs, 'g--')
        
        fig.legend([(invis_calc, invis_act),all_lb,orb_lb,inf_lb], ['Predicted, Actual','All','Orbiting','Infalling'], numpoints=1,handlelength=3,handler_map={tuple: HandlerTuple(ndivide=None)},frameon=False,fontsize=legendfntsize)

        ax_1.plot(bins, mid_ratio_all_prf, 'r')
        ax_1.plot(bins, mid_ratio_orb_prf, 'b')
        ax_1.plot(bins, mid_ratio_inf_prf, 'g')
        
        ax_0.set_ylabel(r"$\rho / \rho_m$", fontsize=axisfntsize)
        ax_0.set_xscale("log")
        ax_0.set_yscale("log")
        ax_0.set_xlim(0.05,np.max(lin_rticks))
        ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        
        fig.legend([(invis_calc, invis_act),all_lb,orb_lb,inf_lb], ['Predicted, Actual','All','Orbiting','Infalling'], numpoints=1,handlelength=3,handler_map={tuple: HandlerTuple(ndivide=None)},frameon=False,fontsize=legendfntsize)

        ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        ax_1.set_ylabel(r"$\frac{\rho_{pred}}{\rho_{act}} - 1$", fontsize=axisfntsize)
        
        ax_1.set_xlim(0.05,np.max(lin_rticks))
        ax_1.set_ylim(bottom=-0.3,top=0.3)
        ax_1.set_xscale("log")
        tick_locs = lin_rticks.copy()
        
        if 0 in lin_rticks:
            tick_locs.remove(0)
        if 0.1 not in tick_locs:
            tick_locs.append(0.1)
            tick_locs = sorted(tick_locs)
        strng_ticks = list(map(str, tick_locs))

        ax_1.set_xticks(tick_locs,strng_ticks)  
        ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        fig.suptitle(title)
        fig.savefig(save_location + title + "prfl_rat.pdf",bbox_inches='tight')

def plot_split_prf(ax,bins,calc_prf,act_prf,curr_color,plt_lines,plt_lbls,var_split,split_name):
    lb, = ax.plot(bins, calc_prf, linestyle='-', color = curr_color, label = rf"{var_split[0]}$< {split_name} <$ {var_split[1]}")    
        
    plt_lines.append(lb)
    plt_lbls.append(rf"{var_split[0]}$< {split_name} <$ {var_split[1]}")
    
    # Plot the SPARTA (actual) profiles 
    ax.plot(bins, act_prf, linestyle='--', color = curr_color)

    return plt_lines, plt_lbls\
        
def plot_split_prf_rat(ax, bins, mid_ratio_prf, all_ratio_prf, curr_color, fill_alpha):
    ax.plot(bins, mid_ratio_prf, color = curr_color)
    
    ax.fill_between(bins, np.nanpercentile(all_ratio_prf, q=15.9, axis=0),np.nanpercentile(all_ratio_prf, q=84.1, axis=0), color=curr_color, alpha=fill_alpha)
   
def create_invis_prf_line(ax, curr_cmap, n_lines, prf_name_0, prf_name_1):
    colors = [curr_cmap(i) for i in np.linspace(0.3, 1, n_lines)]
    
    invis_calc, = ax.plot([0], [0], color=curr_cmap(0.75), linestyle='-')
    invis_act, = ax.plot([0], [0], color=curr_cmap(0.75), linestyle='--')
    
    plt_lines = [invis_calc, invis_act]
    plt_lbls = [prf_name_0,prf_name_1]
    
    return colors, plt_lines, plt_lbls

def plot_split_prf_and_rat(ax0,ax1,bins,calc_prf,act_prf,prf_func,plt_lines,plt_lbls,var_split,split_name,curr_color,fill_alpha):
    calc_prfs, act_prfs, mid_ratio_prf, all_ratio_prf = compute_prfs_info(calc_prf,act_prf,prf_func)
    
    plot_split_prf(ax0, bins, calc_prfs, act_prfs, curr_color, plt_lines, plt_lbls, var_split, split_name)
    
    plot_split_prf_rat(ax1, bins, mid_ratio_prf, all_ratio_prf, curr_color, fill_alpha)

# Profiles should be a list of lists where each list consists of [calc_prf,act_prf] for each split
# You can either use the median plots with use_med=True or the average with use_med=False
# The prf_name_0 and prf_name_1 correspond to what you want each profile to be named in the plot corresponding to where they are located in the _prfs variable
def compare_split_prfs(plt_splits, n_lines, all_prfs, orb_prfs, inf_prfs, bins, lin_rticks, save_location, title, prf_func=np.nanmedian, split_name="\\nu", prf_name_0 = "ML Model", prf_name_1 = "SPARTA"): 
    with timed("Compare Split Profiles"):
        # Parameters to tune sizes of plots and fonts
        widths = [1,1,1]
        heights = [1,0.5]
        axisfntsize=12
        textfntsize = 10
        tickfntsize=10
        legendfntsize=8
        fill_alpha = 0.2
            
        fig = plt.figure(constrained_layout=True,figsize=(10,5))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
        
        all_ax_0 = fig.add_subplot(gs[0,0])
        all_ax_1 = fig.add_subplot(gs[1,0],sharex=all_ax_0)
        orb_ax_0 = fig.add_subplot(gs[0,1])
        orb_ax_1 = fig.add_subplot(gs[1,1],sharex=orb_ax_0)
        inf_ax_0 = fig.add_subplot(gs[0,2])
        inf_ax_1 = fig.add_subplot(gs[1,2],sharex=inf_ax_0)
        
        all_cmap = plt.cm.Reds
        orb_cmap = plt.cm.Blues
        inf_cmap = plt.cm.Greens
        
        all_colors, all_plt_lines, all_plt_lbls = create_invis_prf_line(all_ax_0,all_cmap,n_lines,prf_name_0,prf_name_1)
        orb_colors, orb_plt_lines, orb_plt_lbls = create_invis_prf_line(all_ax_0,orb_cmap,n_lines,prf_name_0,prf_name_1)
        inf_colors, inf_plt_lines, inf_plt_lbls = create_invis_prf_line(all_ax_0,inf_cmap,n_lines,prf_name_0,prf_name_1)

        for i,var_split in enumerate(plt_splits):
            plot_split_prf_and_rat(all_ax_0, all_ax_1, bins, all_prfs[i][0], all_prfs[i][1], prf_func, all_plt_lines, all_plt_lbls, var_split, split_name, all_colors[i], fill_alpha)
            plot_split_prf_and_rat(orb_ax_0, orb_ax_1, bins, orb_prfs[i][0], orb_prfs[i][1], prf_func, orb_plt_lines, orb_plt_lbls, var_split, split_name, orb_colors[i], fill_alpha)
            plot_split_prf_and_rat(inf_ax_0, inf_ax_1, bins, inf_prfs[i][0], inf_prfs[i][1], prf_func, inf_plt_lines, inf_plt_lbls, var_split, split_name, inf_colors[i], fill_alpha)
            
        all_ax_0.set_ylabel(r"$\rho / \rho_m$", fontsize=axisfntsize)
        all_ax_0.set_xscale("log")
        all_ax_0.set_yscale("log")
        all_ax_0.set_xlim(0.05,np.max(lin_rticks))
        all_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        all_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        all_ax_0.legend(all_plt_lines,all_plt_lbls,fontsize=legendfntsize, loc = "upper right")
        all_ax_0.text(0.05,0.05, "All Particles", ha="left", va="bottom", transform=all_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})
        
        orb_ax_0.set_xscale("log")
        orb_ax_0.set_yscale("log")
        orb_ax_0.set_xlim(0.05,np.max(lin_rticks))
        orb_ax_0.set_ylim(all_ax_0.get_ylim())
        orb_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        orb_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        orb_ax_0.tick_params(axis='y', which='both', labelleft=False)
        orb_ax_0.legend(orb_plt_lines,orb_plt_lbls,fontsize=legendfntsize, loc = "upper right")
        orb_ax_0.text(0.05,0.05, "Orbiting Particles", ha="left", va="bottom", transform=orb_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})
        
        inf_ax_0.set_xscale("log")
        inf_ax_0.set_yscale("log")
        inf_ax_0.set_xlim(0.05,np.max(lin_rticks))
        inf_ax_0.set_ylim(all_ax_0.get_ylim())
        inf_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        inf_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        inf_ax_0.tick_params(axis='y', which='both', labelleft=False) 
        inf_ax_0.legend(inf_plt_lines,inf_plt_lbls,fontsize=legendfntsize, loc = "upper right")
        inf_ax_0.text(0.05,0.05, "Infalling Particles", ha="left", va="bottom", transform=inf_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})

        all_y_min, all_y_max = all_ax_0.get_ylim()
        orb_y_min, orb_y_max = orb_ax_0.get_ylim()
        inf_y_min, inf_y_max = inf_ax_0.get_ylim()

        global_y_min = min(all_y_min, orb_y_min, inf_y_min)
        global_y_max = max(all_y_max, orb_y_max, inf_y_max)

        # Set the same y-axis limits for all axes
        all_ax_0.set_ylim(0.1, global_y_max)
        orb_ax_0.set_ylim(0.1, global_y_max)
        inf_ax_0.set_ylim(0.1, global_y_max)
        
        all_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        all_ax_1.set_ylabel(r"$\frac{\rho_{pred}}{\rho_{act}} - 1$", fontsize=axisfntsize)
        orb_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        inf_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        
        all_ax_1.set_xlim(0.05,np.max(lin_rticks))
        all_ax_1.set_ylim(bottom=-0.3,top=0.3)
        all_ax_1.set_xscale("log")
        
        orb_ax_1.set_xlim(0.05,np.max(lin_rticks))
        orb_ax_1.set_ylim(bottom=-0.3,top=0.3)
        orb_ax_1.set_xscale("log")
        
        inf_ax_1.set_xlim(0.05,np.max(lin_rticks))
        inf_ax_1.set_ylim(bottom=-0.3,top=0.3)
        inf_ax_1.set_xscale("log")
        
        tick_locs = lin_rticks.copy()
        if 0 in tick_locs:
            tick_locs.remove(0)
        if 0.1 not in tick_locs:
            tick_locs.append(0.1)
            tick_locs = sorted(tick_locs)
        strng_ticks = list(map(str, tick_locs))

        all_ax_1.set_xticks(tick_locs,strng_ticks)  
        all_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        
        orb_ax_1.set_xticks(tick_locs,strng_ticks)
        orb_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize, labelleft=False)
        
        inf_ax_1.set_xticks(tick_locs,strng_ticks)  
        inf_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize, labelleft=False)
        
        fig.savefig(save_location + title + "prfl_rat.pdf",bbox_inches='tight',dpi=300)

# Creates a stacked mass profile for an entire dataset by generating mass profiles for each halo and combining them
def create_stack_mass_prf(splits, radii, halo_first, halo_n, mass, orbit_assn, prf_bins, use_mp = True, all_z = []):
    calc_mass_prf_orb_lst = []
    calc_mass_prf_inf_lst = []
    calc_mass_prf_all_lst = []
    calc_nu_lst = []
    calc_r200m_lst = []
    
    for i in range(splits.size):
        if i < splits.size - 1:
            curr_num_halos = halo_n[splits[i]:splits[i+1]].shape[0]
        else:
            curr_num_halos = halo_n[splits[i]:].shape[0]
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
            calc_m200m = []
            
            for j in range(curr_num_halos):
                halo_mass_prf_all, halo_mass_prf_orb, halo_mass_prf_inf, m200m = create_mass_prf(radii[halo_first[splits[i]+j]:halo_first[splits[i]+j]+halo_n[splits[i]+j]], orbit_assn[halo_first[splits[i]+j]:halo_first[splits[i]+j]+halo_n[splits[i]+j]], prf_bins,mass[i])
                calc_mass_prf_orb.append(np.array(halo_mass_prf_orb))
                calc_mass_prf_inf.append(np.array(halo_mass_prf_inf))
                calc_mass_prf_all.append(np.array(halo_mass_prf_all))
                m200m = np.array(m200m)
                calc_m200m.append(m200m)
                
        # For each profile combine all halos for each bin
        # calc_mass_prf_xxx has shape (num_halo, num_bins)

        calc_mass_prf_orb_lst.append(comb_prf(calc_mass_prf_orb, curr_num_halos, np.float32))
        calc_mass_prf_inf_lst.append(comb_prf(calc_mass_prf_inf, curr_num_halos, np.float32))
        calc_mass_prf_all_lst.append(comb_prf(calc_mass_prf_all, curr_num_halos, np.float32))

        calc_nu_lst.append(peakHeight(np.array(calc_m200m),all_z[i]))
        calc_r200m_lst.append(M_to_R(np.array(calc_m200m),all_z[i],"200m"))

    calc_mass_prf_orb = np.vstack(calc_mass_prf_orb_lst)
    calc_mass_prf_inf = np.vstack(calc_mass_prf_inf_lst)
    calc_mass_prf_all = np.vstack(calc_mass_prf_all_lst)
    calc_nus = np.concatenate(calc_nu_lst)
    calc_r200m = np.concatenate(calc_r200m_lst)    
    
    return calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m.flatten()

def compare_split_prfs_ke(plt_splits, n_lines, opt_orb_prfs, opt_inf_prfs, fast_orb_prfs, fast_inf_prfs, bins, lin_rticks, save_location, title="comb_ke_fits_", prf_func=np.nanmedian, split_name="\\nu", prf_name_0 = "Optimized Cut", prf_name_1 = "SPARTA", prf_name_2 = "Fast Cut", prf_name_3 = "SPARTA"): 
    with timed("Compare Split KE Cut Profiles"):
        # Parameters to tune sizes of plots and fonts
        widths = [1,1,1,1]
        heights = [1,0.5]

        axisfntsize=22
        textfntsize = 22
        tickfntsize=16
        legendfntsize=16
        fill_alpha = 0.2
            
        fig = plt.figure(constrained_layout=True,figsize=(24,8))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
        
        fast_orb_ax_0 = fig.add_subplot(gs[0,0])
        fast_orb_ax_1 = fig.add_subplot(gs[1,0],sharex=fast_orb_ax_0)
        fast_inf_ax_0 = fig.add_subplot(gs[0,1])
        fast_inf_ax_1 = fig.add_subplot(gs[1,1],sharex=fast_inf_ax_0)
        
        opt_orb_ax_0 = fig.add_subplot(gs[0,2],sharex=fast_orb_ax_0)
        opt_orb_ax_1 = fig.add_subplot(gs[1,2],sharex=fast_orb_ax_0)
        opt_inf_ax_0 = fig.add_subplot(gs[0,3],sharex=fast_inf_ax_0)
        opt_inf_ax_1 = fig.add_subplot(gs[1,3],sharex=fast_inf_ax_0)
         
        opt_orb_cmap = plt.cm.Blues
        opt_inf_cmap = plt.cm.Greens
        fast_orb_cmap = plt.cm.Blues
        fast_inf_cmap = plt.cm.Greens
        
        opt_orb_colors, opt_orb_plt_lines, opt_orb_plt_lbls = create_invis_prf_line(opt_orb_ax_0,opt_orb_cmap,n_lines,prf_name_0,prf_name_1)
        opt_inf_colors, opt_inf_plt_lines, opt_inf_plt_lbls = create_invis_prf_line(opt_inf_ax_0,opt_inf_cmap,n_lines,prf_name_0,prf_name_1)
        fast_orb_colors, fast_orb_plt_lines, fast_orb_plt_lbls = create_invis_prf_line(fast_orb_ax_0,fast_orb_cmap,n_lines,prf_name_2,prf_name_3)
        fast_inf_colors, fast_inf_plt_lines, fast_inf_plt_lbls = create_invis_prf_line(fast_inf_ax_0,fast_inf_cmap,n_lines,prf_name_2,prf_name_3)
        
        for i,var_split in enumerate(plt_splits):
            plot_split_prf_and_rat(opt_orb_ax_0, opt_orb_ax_1, bins, opt_orb_prfs[i][0],opt_orb_prfs[i][1], prf_func, opt_orb_plt_lines, opt_orb_plt_lbls, var_split, split_name, opt_orb_colors[i], fill_alpha)
            plot_split_prf_and_rat(opt_inf_ax_0, opt_inf_ax_1, bins, opt_inf_prfs[i][0],opt_inf_prfs[i][1], prf_func, opt_inf_plt_lines, opt_inf_plt_lbls, var_split, split_name, opt_inf_colors[i], fill_alpha)
            plot_split_prf_and_rat(fast_orb_ax_0, fast_orb_ax_1, bins, fast_orb_prfs[i][0],fast_orb_prfs[i][1], prf_func, fast_orb_plt_lines, fast_orb_plt_lbls, var_split, split_name, fast_orb_colors[i], fill_alpha)
            plot_split_prf_and_rat(fast_inf_ax_0, fast_inf_ax_1, bins, fast_inf_prfs[i][0],fast_inf_prfs[i][1], prf_func, fast_inf_plt_lines, fast_inf_plt_lbls, var_split, split_name, fast_inf_colors[i], fill_alpha)
                

        fast_orb_ax_0.set_xscale("log")
        fast_orb_ax_0.set_yscale("log")
        fast_orb_ax_0.set_xlim(0.05,np.max(lin_rticks))
        fast_orb_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        fast_orb_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        fast_orb_ax_0.tick_params(axis='y', which='both')
        fast_orb_ax_0.legend(fast_orb_plt_lines,fast_orb_plt_lbls,fontsize=legendfntsize, loc = "lower left")
        fast_orb_ax_0.text(0.95,0.95, "Orbiting Particles", ha="right", va="top", transform=fast_orb_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})
        
        fast_inf_ax_0.set_xscale("log")
        fast_inf_ax_0.set_yscale("log")
        fast_inf_ax_0.set_xlim(0.05,np.max(lin_rticks))
        fast_inf_ax_0.set_ylim(fast_orb_ax_0.get_ylim())
        fast_inf_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        fast_inf_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        fast_inf_ax_0.tick_params(axis='y', which='both', labelleft=False) 
        fast_inf_ax_0.legend(fast_inf_plt_lines,fast_inf_plt_lbls,fontsize=legendfntsize, loc = "lower left")
        fast_inf_ax_0.text(0.95,0.95, "Infalling Particles", ha="right", va="top", transform=fast_inf_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})

        opt_orb_ax_0.set_xscale("log")
        opt_orb_ax_0.set_yscale("log")
        opt_orb_ax_0.set_xlim(0.05,np.max(lin_rticks))
        opt_orb_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        opt_orb_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        opt_orb_ax_0.tick_params(axis='y', which='both', labelleft=False)
        opt_orb_ax_0.legend(opt_orb_plt_lines,opt_orb_plt_lbls,fontsize=legendfntsize, loc = "lower left")
        opt_orb_ax_0.text(0.95,0.95, "Orbiting Particles", ha="right", va="top", transform=opt_orb_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})
        
        opt_inf_ax_0.set_xscale("log")
        opt_inf_ax_0.set_yscale("log")
        opt_inf_ax_0.set_xlim(0.05,np.max(lin_rticks))
        opt_inf_ax_0.set_ylim(opt_orb_ax_0.get_ylim())
        opt_inf_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        opt_inf_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
        opt_inf_ax_0.tick_params(axis='y', which='both', labelleft=False) 
        opt_inf_ax_0.legend(opt_inf_plt_lines,opt_inf_plt_lbls,fontsize=legendfntsize, loc = "lower left")
        opt_inf_ax_0.text(0.95,0.95, "Infalling Particles", ha="right", va="top", transform=opt_inf_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})

        opt_orb_y_min, opt_orb_y_max = opt_orb_ax_0.get_ylim()
        opt_inf_y_min, opt_inf_y_max = opt_inf_ax_0.get_ylim()
        
        fast_orb_y_min, fast_orb_y_max = fast_orb_ax_0.get_ylim()
        fast_inf_y_min, fast_inf_y_max = fast_inf_ax_0.get_ylim()

        global_y_min = min(opt_orb_y_min, opt_inf_y_min, fast_orb_y_min, fast_inf_y_min)
        global_y_max = max(opt_orb_y_max, opt_inf_y_max, fast_orb_y_max, fast_inf_y_max)

        # Set the same y-axis limits for all axes
        opt_orb_ax_0.set_ylim(0.1, global_y_max)
        opt_inf_ax_0.set_ylim(0.1, global_y_max)
        
        fast_orb_ax_0.set_ylim(0.1, global_y_max)
        fast_inf_ax_0.set_ylim(0.1, global_y_max)
        
        fast_orb_ax_0.set_ylabel(r"$\rho/\rho_m$", fontsize=axisfntsize)
        fast_orb_ax_1.set_ylabel(r"$\frac{\rho_{pred}}{\rho_{act}} - 1$", fontsize=axisfntsize)
        opt_orb_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        opt_inf_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        fast_orb_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        fast_inf_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
        
        opt_orb_ax_1.set_xlim(0.05,np.max(lin_rticks))
        opt_orb_ax_1.set_ylim(bottom=-0.5,top=0.5)
        opt_orb_ax_1.set_xscale("log")
        
        opt_inf_ax_1.set_xlim(0.05,np.max(lin_rticks))
        opt_inf_ax_1.set_ylim(bottom=-0.5,top=0.5)
        opt_inf_ax_1.set_xscale("log")
        
        fast_orb_ax_1.set_xlim(0.05,np.max(lin_rticks))
        fast_orb_ax_1.set_ylim(bottom=-0.5,top=0.5)
        fast_orb_ax_1.set_xscale("log")
        
        fast_inf_ax_1.set_xlim(0.05,np.max(lin_rticks))
        fast_inf_ax_1.set_ylim(bottom=-0.5,top=0.5)
        fast_inf_ax_1.set_xscale("log")
        
        tick_locs = lin_rticks.copy()
        if 0 in tick_locs:
            tick_locs.remove(0)
        if 0.1 not in tick_locs:
            tick_locs.append(0.1)
            tick_locs = sorted(tick_locs)
        strng_ticks = list(map(str, tick_locs))
        
        opt_orb_ax_1.set_xticks(tick_locs,strng_ticks)
        opt_orb_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize, labelleft=False)
        
        opt_inf_ax_1.set_xticks(tick_locs,strng_ticks)  
        opt_inf_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize, labelleft=False)
        
        fast_orb_ax_1.set_xticks(tick_locs,strng_ticks)
        fast_orb_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
        
        fast_inf_ax_1.set_xticks(tick_locs,strng_ticks)  
        fast_inf_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize, labelleft=False)
        
        fig.savefig(save_location + title + "prfl_rat.pdf",bbox_inches='tight',dpi=400)    

# Creates the density profiles seen throughout the paper.
# 3 panels for all, orbiting, and infalling profiles with options to be split by nu or by mass accretion rate
def paper_dens_prf_plt(X,y,preds,halo_df,use_sims,sim_cosmol,split_scale_dict,plot_save_loc,split_by_nu=False,split_by_macc=False):
    halo_first = halo_df["Halo_first"].values
    halo_n = halo_df["Halo_n"].values
    all_idxs = halo_df["Halo_indices"].values
    
    lin_rticks = split_scale_dict["lin_rticks"]

    cosmol = set_cosmology(sim_cosmol)

    all_z = []
    all_rhom = []
    # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
    sim_splits = np.where(halo_first == 0)[0]

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
        dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")

        curr_z = dset_params["all_snap_info"]["prime_snap_info"]["red_shift"][()]
        curr_rho_m = dset_params["all_snap_info"]["prime_snap_info"]["rho_m"][()]
        all_z.append(curr_z)
        all_rhom.append(curr_rho_m)
        h = dset_params["all_snap_info"]["prime_snap_info"]["h"][()]
    
    tot_num_halos = halo_n.shape[0]
    min_disp_halos = int(np.ceil(0.3 * tot_num_halos))
    
    # Get SPARTA's mass profiles
    act_mass_prf_all, act_mass_prf_orb,all_masses,bins = load_sparta_mass_prf(sim_splits,all_idxs,use_sims)
    act_mass_prf_inf = act_mass_prf_all - act_mass_prf_orb
    
    # Create mass profiles from the model's predictions
    prime_radii = X["p_Scaled_radii"].values.compute()
    calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m = create_stack_mass_prf(sim_splits,radii=prime_radii, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=preds.values, prf_bins=bins, use_mp=True, all_z=all_z)
    my_mass_prf_all, my_mass_prf_orb, my_mass_prf_inf, my_nus, my_r200m = create_stack_mass_prf(sim_splits,radii=prime_radii, halo_first=halo_first, halo_n=halo_n, mass=all_masses, orbit_assn=y.compute().values.flatten(), prf_bins=bins, use_mp=True, all_z=all_z)
    
    # Halos that get returned with a nan R200m mean that they didn't meet the required number of ptls within R200m and so we need to filter them from our calculated profiles and SPARTA profiles 
    small_halo_fltr = np.isnan(calc_r200m)
    act_mass_prf_all[small_halo_fltr,:] = np.nan
    act_mass_prf_orb[small_halo_fltr,:] = np.nan
    act_mass_prf_inf[small_halo_fltr,:] = np.nan
    
    all_prfs = [calc_mass_prf_all, act_mass_prf_all]
    orb_prfs = [calc_mass_prf_orb, act_mass_prf_orb]
    inf_prfs = [calc_mass_prf_inf, act_mass_prf_inf]

    # Calculate the density by divide the mass of each bin by the volume of that bin's radius
    calc_dens_prf_all = calc_rho(calc_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    calc_dens_prf_orb = calc_rho(calc_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    calc_dens_prf_inf = calc_rho(calc_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    
    act_dens_prf_all = calc_rho(act_mass_prf_all*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    act_dens_prf_orb = calc_rho(act_mass_prf_orb*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    act_dens_prf_inf = calc_rho(act_mass_prf_inf*h,bins[1:],calc_r200m*h,sim_splits,all_rhom)
    
    if debug_indiv_dens_prf > 0:
        my_dens_prf_orb = calc_rho(my_mass_prf_orb*h,bins[1:],my_r200m*h,sim_splits,all_rhom)
        my_dens_prf_all = calc_rho(my_mass_prf_all*h,bins[1:],my_r200m*h,sim_splits,all_rhom)
        my_dens_prf_inf = calc_rho(my_mass_prf_inf*h,bins[1:],my_r200m*h,sim_splits,all_rhom)
    
        ratio = np.where(act_dens_prf_all != 0, calc_dens_prf_all / act_dens_prf_all, np.nan)

        # Compute the difference for each halo (using range: max - min)
        diff = np.nanmax(ratio, axis=1) - np.nanmin(ratio, axis=1)

        # If you want the top k halos with the largest differences, use:
        k = 5  # Example value
        big_halo_loc = np.argsort(diff)[-k:]
    
        for i in range(k):
            all_prfs = [my_mass_prf_all[big_halo_loc[i]], act_mass_prf_all[big_halo_loc[i]]]
            orb_prfs = [my_mass_prf_orb[big_halo_loc[i]], act_mass_prf_orb[big_halo_loc[i]]]
            inf_prfs = [my_mass_prf_inf[big_halo_loc[i]], act_mass_prf_inf[big_halo_loc[i]]]
            compare_prfs(all_prfs,orb_prfs,inf_prfs,bins[1:],lin_rticks,debug_plt_path,sim + "_" + str(i)+"_mass",prf_func=None)

        for i in range(k):
            all_prfs = [my_dens_prf_all[big_halo_loc[i]], act_dens_prf_all[big_halo_loc[i]]]
            orb_prfs = [my_dens_prf_orb[big_halo_loc[i]], act_dens_prf_orb[big_halo_loc[i]]]
            inf_prfs = [my_dens_prf_inf[big_halo_loc[i]], act_dens_prf_inf[big_halo_loc[i]]]
            compare_prfs(all_prfs,orb_prfs,inf_prfs,bins[1:],lin_rticks,debug_plt_path,sim + "_" + str(i)+"_dens",prf_func=None)
            
    curr_halos_r200m_list = []
    past_halos_r200m_list = []
    
    for i,sim in enumerate(use_sims):
        if i < len(use_sims) - 1:
            curr_idxs = all_idxs[sim_splits[i]:sim_splits[i+1]]
        else:
            curr_idxs = all_idxs[sim_splits[i]:]
        # We reload dset_params as we need this information on a sim by sim basis and it doesn't take very long to load
        dset_params = load_pickle(ML_dset_path + sim + "/dset_params.pickle")
        curr_z = dset_params["all_snap_info"]["prime_snap_info"]["red_shift"][()]
        p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"][()]
        snap_dir_format = dset_params["snap_dir_format"]
        snap_format = dset_params["snap_format"]
        
        sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"
                
        # Load the halo's positions and radii
        param_paths = [["halos","R200m"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, save_data=save_intermediate_data)
        p_halos_r200m = sparta_params[sparta_param_names[0]][:,p_sparta_snap]

        p_halos_r200m = p_halos_r200m[curr_idxs]

        # If we want the density profiles to only consist of halos of a specific peak height (nu) bin 
        if split_by_nu:
            nu_all_prf_lst = []
            nu_orb_prf_lst = []
            nu_inf_prf_lst = []
        
            cpy_plt_nu_splits = plt_nu_splits.copy()
            for i,nu_split in enumerate(cpy_plt_nu_splits):
                # Take the second element of the where to filter by the halos (?)
                fltr = np.where((calc_nus > nu_split[0]) & (calc_nus < nu_split[1]))[0]

                if fltr.shape[0] > min_halo_nu_bin:
                    nu_all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
                    nu_orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
                    nu_inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
                else:
                    plt_nu_splits.remove(nu_split)
            compare_split_prfs(plt_nu_splits,len(cpy_plt_nu_splits),nu_all_prf_lst,nu_orb_prf_lst,nu_inf_prf_lst,bins[1:],lin_rticks,plot_save_loc,title="nu_dens_med_",prf_func=np.nanmedian, prf_name_0="ML Model", prf_name_1="SPARTA")
        if split_by_macc and dset_params["t_dyn_steps"]:
            all_tdyn_steps = dset_params["t_dyn_steps"]
            if all_tdyn_steps[0] == 1:
                # we can just use the secondary snap here because if it was already calculated for 1 dynamical time forago
                past_z = dset_params["all_snap_info"]["comp_"+str(all_tdyn_steps[0]) + "_tdstp_snap_info"]["red_shift"][()] 
                c_sparta_snap = dset_params["all_snap_info"]["comp_"+str(all_tdyn_steps[0]) + "_tdstp_snap_info"]["sparta_snap"][()]
            else:
                # If the prior secondary snap is not 1 dynamical time ago get that information
                
                with h5py.File(curr_sparta_HDF5_path,"r") as f:
                    dic_sim = {}
                    grp_sim = f['simulation']
                    for f in grp_sim.attrs:
                        dic_sim[f] = grp_sim.attrs[f]
                    
                all_sparta_z = dic_sim['snap_z']
                little_h = dic_sim["h"]
                
                past_z = get_past_z(cosmol, curr_z, tdyn_step=1)
                c_snap_dict = get_comp_snap_info(cosmol = cosmol, past_z=past_z, all_sparta_z=all_sparta_z,snap_dir_format=snap_dir_format,snap_format=snap_format,snap_path=snap_path)
                c_sparta_snap = c_snap_dict["sparta_snap"]
            c_halos_r200m = sparta_params[sparta_param_names[0]][:,c_sparta_snap]
            c_halos_r200m = c_halos_r200m[curr_idxs]
            
            curr_halos_r200m_list.append(p_halos_r200m)
            past_halos_r200m_list.append(c_halos_r200m)
            
            curr_halos_r200m = np.concatenate(curr_halos_r200m_list)
            past_halos_r200m = np.concatenate(past_halos_r200m_list)
            macc_all_prf_lst = []
            macc_orb_prf_lst = []
            macc_inf_prf_lst = []
            
            calc_maccs = calc_mass_acc_rate(curr_halos_r200m,past_halos_r200m,curr_z,past_z)
            cpy_plt_macc_splits = plt_macc_splits.copy()
            for i,macc_split in enumerate(cpy_plt_macc_splits):
                # Take the second element of the where to filter by the halos (?)
                fltr = np.where((calc_maccs > macc_split[0]) & (calc_maccs < macc_split[1]))[0]
                if fltr.shape[0] > 25:
                    macc_all_prf_lst.append(filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos,fltr))
                    macc_orb_prf_lst.append(filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos,fltr))
                    macc_inf_prf_lst.append(filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos,fltr))
                else:
                    plt_macc_splits.remove(macc_split)

            
            compare_split_prfs(plt_macc_splits,len(cpy_plt_macc_splits),macc_all_prf_lst,macc_orb_prf_lst,macc_inf_prf_lst,bins[1:],lin_rticks,plot_save_loc,title= "macc_dens_", split_name="\Gamma", prf_name_0="ML Model", prf_name_1="SPARTA")
        if not split_by_nu and not split_by_macc:
            all_prf_lst = filter_prf(calc_dens_prf_all,act_dens_prf_all,min_disp_halos)
            orb_prf_lst = filter_prf(calc_dens_prf_orb,act_dens_prf_orb,min_disp_halos)
            inf_prf_lst = filter_prf(calc_dens_prf_inf,act_dens_prf_inf,min_disp_halos)
            
            # Ignore warnigns about taking mean/median of empty slices and division by 0 that are expected with how the profiles are handled
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                compare_prfs(all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_save_loc,title="dens_med_",prf_func=np.nanmedian)
                compare_prfs(all_prf_lst,orb_prf_lst,inf_prf_lst,bins[1:],lin_rticks,plot_save_loc,title="dens_avg_",prf_func=np.nanmean)