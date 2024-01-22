import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg')
import matplotlib.gridspec as gridspec
from calculation_functions import calculate_distance
#import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report
from colossus.halo import mass_so
from data_and_loading_functions import check_pickle_exist_gadget, create_directory
from calculation_functions import calc_v200m
# import general_plotting as gp
from textwrap import wrap
from scipy.ndimage import rotate
import matplotlib.colors as colors

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
    if radii_within_range.size != 0 and idx != 0:
        calc_prf[idx] = calc_prf[idx - 1] + radii_within_range.size * mass
        diff_n_ptl[idx] = ((act_prf[idx] - act_prf[idx-1])/mass) - radii_within_range.size
    elif radii_within_range.size != 0 and idx == 0:
        calc_prf[idx] = radii_within_range.size * mass
        diff_n_ptl[idx] = act_prf[idx]/mass - radii_within_range.size
    elif radii_within_range.size == 0 and idx != 0:
        calc_prf[idx] = calc_prf[idx-1]
        diff_n_ptl[idx] = (act_prf[idx] - act_prf[idx-1])/mass   
    else:
        calc_prf[idx] = calc_prf[idx-1]
        diff_n_ptl[idx] = (act_prf[idx])/mass
    
    return calc_prf, diff_n_ptl

def compare_density_prf(radii, actual_prf_all, actual_prf_1halo, mass, orbit_assn, prf_bins, title, save_location, show_graph = False, save_graph = False):
    create_directory(save_location + "dens_prfl_ratio/")
    
    actual_prf_inf = actual_prf_all - actual_prf_1halo
    # Create bins for the density profile calculation
    num_prf_bins = actual_prf_all.shape[0]

    calc_prf_orb = np.zeros(num_prf_bins)
    calc_prf_inf = np.zeros(num_prf_bins)
    calc_prf_all = np.zeros(num_prf_bins)
    diff_n_inf_ptls = np.zeros(num_prf_bins)
    diff_n_orb_ptls = np.zeros(num_prf_bins)
    diff_n_all_ptls = np.zeros(num_prf_bins)
    
    orbit_radii = radii[np.where(orbit_assn == 1)[0]]
    infall_radii = radii[np.where(orbit_assn == 0)[0]]

    for i in range(num_prf_bins):
        start_bin = prf_bins[i]
        end_bin = prf_bins[i+1]  
        
        calc_prf_orb, diff_n_orb_ptls = update_density_prf(calc_prf_orb, diff_n_orb_ptls, orbit_radii, i, start_bin, end_bin, mass, actual_prf_1halo)      
        calc_prf_inf, diff_n_inf_ptls = update_density_prf(calc_prf_inf, diff_n_inf_ptls, infall_radii, i, start_bin, end_bin, mass, actual_prf_inf)      
        calc_prf_all, diff_n_all_ptls = update_density_prf(calc_prf_all, diff_n_all_ptls, radii, i, start_bin, end_bin, mass, actual_prf_all)      
       
    middle_bins = (prf_bins[1:] + prf_bins[:-1]) / 2

    fig, ax = plt.subplots(1,2)

    with np.errstate(divide='ignore', invalid='ignore'):
        all_ratio = np.divide(calc_prf_all,actual_prf_all)
        inf_ratio = np.divide(calc_prf_inf,(actual_prf_all - actual_prf_1halo))
        orb_ratio = np.divide(calc_prf_orb,actual_prf_1halo)

    ax[0].plot(middle_bins, all_ratio, 'r', label = "My prf / SPARTA prf all")
    ax[0].plot(middle_bins, orb_ratio, 'b', label = "My prf / SPARTA profile orb")
    ax[0].plot(middle_bins, inf_ratio, 'g', label = "My prf / SPARTA profile inf")
    
    ax[0].set_title(wrap("My Predicted  / Actual Density Profile for halo idx: " + title))
    ax[0].set_xlabel("radius $r/R_{200m}$")
    ax[0].set_ylabel("My Dens Prf / Act Dens Prf")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].legend()

    ax[1].plot(middle_bins, calc_prf_all, 'r-', label = "My prf all")
    ax[1].plot(middle_bins, calc_prf_orb, 'b-', label = "My prf orb")
    ax[1].plot(middle_bins, calc_prf_inf, 'g-', label = "My prf inf")
    ax[1].plot(middle_bins, actual_prf_all, 'r--', label = "SPARTA prf all")
    ax[1].plot(middle_bins, actual_prf_1halo, 'b--', label = "SPARTA prf orb")
    ax[1].plot(middle_bins, (actual_prf_all - actual_prf_1halo), 'g--', label = "SPARTA prf inf")
    ax[1].set_title(wrap("ML Predicted vs Actual Density Profile for halo idx: " + title))
    ax[1].set_xlabel("radius $r/R_{200m}$")
    ax[1].set_ylabel("Mass $M_/odot$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].legend()
    
    if save_graph:
        fig.set_size_inches(21, 13)
        create_directory(save_location + "dens_prfl_ratio/")
        fig.savefig(save_location + "dens_prfl_ratio/" + title + ".png", bbox_inches='tight')
    if show_graph:
        plt.show()
    plt.close()
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
 
def phase_plot(ax, x, y, min_ptl, max_ptl, range, num_bins, cmap, x_label="", y_label="", hide_xticks=False, hide_yticks=False, title=""):
    ax.hist2d(x, y, bins=num_bins, range=range, density=False, weights=None, cmin=min_ptl, cmap=cmap, norm="log", vmin=min_ptl, vmax=max_ptl)
    if title != "":
        ax.text(.01,.03,title, ha="left", va="bottom", transform=ax.transAxes, fontsize="large")
    if x_label != "":
        ax.set_xlabel(x_label)
    if y_label != "":
        ax.set_ylabel(y_label)
    if hide_xticks:
        ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False) 
    if hide_yticks:
        ax.tick_params(axis='y', which='both',left=False,labelleft=False) 
        
def imshow_plot(ax, img, extent, x_label="", y_label="", title="", return_img=False, hide_xticks=False, hide_yticks=False, kwargs={}):
    img=ax.imshow(img, interpolation="none", extent = extent, **kwargs)
    if title != "":
        ax.text(.01,.03,title, ha="left", va="bottom", transform=ax.transAxes, fontsize="large")
    if x_label != "":
        ax.set_xlabel(x_label)
    if y_label != "":
        ax.set_ylabel(y_label)
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


def create_hist_max_ptl(min_ptl, set_ptl, inf_r, orb_r, inf_rv, orb_rv, inf_tv, orb_tv, num_bins, max_r, max_rv, min_rv, max_tv, min_tv, bin_r_rv = None, bin_r_tv = None, bin_rv_tv = None):
    if bin_r_rv == None:
        bins = num_bins
        orb_r_rv = histogram(orb_r, orb_rv, bins=bins, range=[[0,max_r],[min_rv,max_rv]],min_ptl=min_ptl,set_ptl=set_ptl)
        orb_r_tv = histogram(orb_r, orb_tv, bins=bins, range=[[0,max_r],[min_tv,max_tv]],min_ptl=min_ptl,set_ptl=set_ptl)
        orb_rv_tv = histogram(orb_rv, orb_tv, bins=bins, range=[[min_rv,max_rv],[min_tv,max_tv]],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_r_rv = histogram(inf_r, inf_rv, bins=bins, range=[[0,max_r],[min_rv,max_rv]],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_r_tv = histogram(inf_r, inf_tv, bins=bins, range=[[0,max_r],[min_tv,max_tv]],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_rv_tv = histogram(inf_rv, inf_tv, bins=bins, range=[[min_rv,max_rv],[min_tv,max_tv]],min_ptl=min_ptl,set_ptl=set_ptl)
    else:
        orb_r_rv = histogram(orb_r, orb_rv, bins=bin_r_rv, range=[[0,max_r],[min_rv,max_rv]],min_ptl=min_ptl,set_ptl=set_ptl)
        orb_r_tv = histogram(orb_r, orb_tv, bins=bin_r_tv, range=[[0,max_r],[min_tv,max_tv]],min_ptl=min_ptl,set_ptl=set_ptl)
        orb_rv_tv = histogram(orb_rv, orb_tv, bins=bin_rv_tv, range=[[min_rv,max_rv],[min_tv,max_tv]],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_r_rv = histogram(inf_r, inf_rv, bins=bin_r_rv, range=[[0,max_r],[min_rv,max_rv]],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_r_tv = histogram(inf_r, inf_tv, bins=bin_r_tv, range=[[0,max_r],[min_tv,max_tv]],min_ptl=min_ptl,set_ptl=set_ptl)
        inf_rv_tv = histogram(inf_rv, inf_tv, bins=bin_rv_tv, range=[[min_rv,max_rv],[min_tv,max_tv]],min_ptl=min_ptl,set_ptl=set_ptl)

    max_ptl = np.max(np.array([np.max(orb_r_rv[0]),np.max(orb_r_tv[0]),np.max(orb_rv_tv[0]),np.max(inf_r_rv[0]),np.max(inf_r_tv[0]),np.max(inf_rv_tv[0]),]))
    
    return max_ptl, orb_r_rv, orb_r_tv, orb_rv_tv, inf_r_rv, inf_r_tv, inf_rv_tv

def percent_error(pred, act):
    return (((pred - act))/act) * 100

def plot_incorrectly_classified(correct_labels, ml_labels, r, rv, tv, num_bins, title, save_location, model_save_location, act_orb_r_rv, act_orb_r_tv, act_orb_rv_tv, act_inf_r_rv, act_inf_r_tv, act_inf_rv_tv): 
    min_ptl = 1e-4
    act_min_ptl = 1
    max_r = np.max(r)
    max_rv = np.max(rv)
    min_rv = np.min(rv)
    max_tv = np.max(tv)
    min_tv = np.min(tv)
       
    inc_inf = np.where((ml_labels == 1) & (correct_labels == 0))[0]
    num_orb = np.where(correct_labels == 1)[0].shape[0]
    inc_orb = np.where((ml_labels == 0) & (correct_labels == 1))[0]
    num_inf = np.where(correct_labels == 0)[0].shape[0]
    tot_num_inc = inc_orb.shape[0] + inc_inf.shape[0]
    tot_num_ptl = num_orb + num_inf
    print("num incorrect inf", inc_inf.shape, ",", np.round(((inc_inf.shape[0]/num_inf)*100),2), "% of infalling ptls")
    print("num incorrect orb", inc_orb.shape, ",", np.round(((inc_orb.shape[0]/num_orb) * 100),2), "% of orbiting ptls")
    print("num incorrect tot", tot_num_ptl, ",", np.round(((tot_num_inc/tot_num_ptl) * 100),2), "% of all ptls")

    file = open(model_save_location + "model_info.txt", 'a')
    file.write("Percent of Orbiting Particles Mislabeled: " + str(np.round(((inc_orb.shape[0]/num_orb)*100),2)) + "%\n")
    file.write("Percent of Infalling Particles Mislabeled: " + str(np.round(((inc_inf.shape[0]/num_inf)*100),2)) + "%\n")
    file.write("Percent of Total Particles Misclassified: " + str(np.round(((tot_num_inc/tot_num_ptl)*100),2)) + "%\n")
    file.close()
    
    inc_orb_r = r[inc_orb]
    inc_inf_r = r[inc_inf]
    inc_orb_rv = rv[inc_orb]
    inc_inf_rv = rv[inc_inf]
    inc_orb_tv = tv[inc_orb]
    inc_inf_tv = tv[inc_inf]

    act_inf_r, act_orb_r = split_orb_inf(r, correct_labels)
    act_inf_rv, act_orb_rv = split_orb_inf(rv, correct_labels)
    act_inf_tv, act_orb_tv = split_orb_inf(tv, correct_labels)

    max_all_ptl, act_orb_r_rv, act_orb_r_tv, act_orb_rv_tv, act_inf_r_rv, act_inf_r_tv, act_inf_rv_tv = create_hist_max_ptl(act_min_ptl, 0, act_inf_r, act_orb_r, act_inf_rv, act_orb_rv, act_inf_tv, act_orb_tv, num_bins, max_r, max_rv, min_rv, max_tv, min_tv)    
    max_ptl, inc_orb_r_rv, inc_orb_r_tv, inc_orb_rv_tv, inc_inf_r_rv, inc_inf_r_tv, inc_inf_rv_tv = create_hist_max_ptl(min_ptl, min_ptl, inc_inf_r, inc_orb_r, inc_inf_rv, inc_orb_rv, inc_inf_tv, inc_orb_tv, num_bins, max_r, max_rv, min_rv, max_tv, min_tv, bin_r_rv=act_orb_r_rv[1:], bin_r_tv=act_orb_r_tv[1:],bin_rv_tv=act_orb_rv_tv[1:])
    
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

    # all_r_rv_hist = histogram(x=r,y=rv,bins=num_bins,range=[[0,max_r],[min_rv,max_rv]],min_ptl=0)
    # all_r_tv_hist = histogram(x=r,y=tv,bins=num_bins,range=[[0,max_r],[min_tv,max_tv]],min_ptl=0)
    # all_rv_tv_hist = histogram(x=rv,y=tv,bins=num_bins,range=[[min_rv,max_rv],[min_tv,max_tv]],min_ptl=0)
    

    max_diff = np.max(np.array([np.max(scaled_orb_r_rv),np.max(scaled_orb_r_tv),np.max(scaled_orb_rv_tv),
                                np.max(scaled_inf_r_rv),np.max(scaled_inf_r_tv),np.max(scaled_inf_rv_tv),
                                np.max(scaled_all_r_rv),np.max(scaled_all_r_tv),np.max(scaled_all_rv_tv)]))
    
    cividis_cmap = plt.get_cmap("cividis_r")
    cividis_cmap.set_under(color='white')   
    magma_cmap = plt.get_cmap("magma_r")
    magma_cmap.set_under(color='white') 
    cmap = plt.get_cmap("magma")
    test_cmap = plt.get_cmap("viridis")
    
    scale_miss_class_args = {
        "vmin":min_ptl,
        "vmax":max_diff,
        "norm":"log",
        "origin":"lower",
        "aspect":"auto",
        "cmap":magma_cmap,
    }

    all_miss_class_args = {
        "vmin":1,
        "vmax":max_all_ptl,
        "norm":"log",
        "origin":"lower",
        "aspect":"auto",
        "cmap":cividis_cmap,
    }
    
    # print(act_inf_r_rv[0].T[13,:15])
    # print(inc_inf_r_rv[0].T[13,:15])
    # print(scaled_inf_r_rv[13,:15])

    # fig, ax = plt.subplots(3)
    # ax[0].imshow(act_inf_r_rv[0].T, interpolation="none", extent =[0,max_r,min_rv,max_rv], **all_miss_class_args)
    # ax[1].imshow(inc_inf_r_rv[0].T, interpolation="none", extent =[0,max_r,min_rv,max_rv], **all_miss_class_args)
    # ax[2].imshow(scaled_inf_r_rv, interpolation="none", extent =[0,max_r,min_rv,max_rv], **scale_miss_class_args)
    # fig.savefig(save_location+"/2dhist/"+title+"test.png")

    widths = [4,4,4,.5]
    heights = [4,4,4,4,4]
    
    scal_miss_class_fig = plt.figure(constrained_layout=True, figsize=(10,15))
    scal_miss_class_fig.suptitle("Misclassified Particles/Num Targets " + title)
    gs = scal_miss_class_fig.add_gridspec(5,4,width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
    plt.rcParams.update({'font.size': 12})

    phase_plot(scal_miss_class_fig.add_subplot(gs[0,0]), r, rv, min_ptl=1, max_ptl=max_all_ptl, range=[[0,max_r],[min_rv,max_rv]],num_bins=num_bins,cmap=cividis_cmap,y_label="$v_r/v_{200m}$", hide_xticks=True, title="Actual\nDistribution")
    phase_plot(scal_miss_class_fig.add_subplot(gs[0,1]), r, tv, min_ptl=1, max_ptl=max_all_ptl, range=[[0,max_r],[min_tv,max_tv]],num_bins=num_bins,cmap=cividis_cmap,y_label="$v_t/v_{200m}$",hide_xticks=True)
    phase_plot(scal_miss_class_fig.add_subplot(gs[0,2]), rv, tv, min_ptl=1, max_ptl=max_all_ptl, range=[[min_rv,max_rv],[min_tv,max_tv]],num_bins=num_bins,cmap=cividis_cmap, hide_xticks=True, hide_yticks=True)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[1,0]), all_inc_r_rv.T, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",hide_xticks=True,title="All Misclassified",kwargs=all_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[1,1]), all_inc_r_tv.T, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=all_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[1,2]), all_inc_rv_tv.T, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=all_miss_class_args)
    phase_plt_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=1, vmax=max_all_ptl),cmap=cividis_cmap), cax=plt.subplot(gs[0:2,-1]))

    imshow_plot(scal_miss_class_fig.add_subplot(gs[2,0]), scaled_inf_r_rv, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",title="Label: Orbit\nReal: Infall",hide_xticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[2,1]), scaled_inf_r_tv, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[2,2]), scaled_inf_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[3,0]), scaled_orb_r_rv, extent=[0,max_r,min_rv,max_rv],y_label="$v_r/v_{200m}$",title="Label: Infall\nReal: Orbit",hide_xticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[3,1]), scaled_orb_r_tv, extent=[0,max_r,min_tv,max_tv],y_label="$v_t/v_{200m}$",hide_xticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[3,2]), scaled_orb_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],hide_xticks=True,hide_yticks=True,kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[4,0]), scaled_all_r_rv, extent=[0,max_r,min_rv,max_rv],x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",title="All Misclassified Scaled",kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[4,1]), scaled_all_r_tv, extent=[0,max_r,min_tv,max_tv],x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$",kwargs=scale_miss_class_args)
    imshow_plot(scal_miss_class_fig.add_subplot(gs[4,2]), scaled_all_rv_tv, extent=[min_rv,max_rv,min_tv,max_tv],x_label="$v_r/v_{200m}$",hide_yticks=True,kwargs=scale_miss_class_args)
    
    scal_misclas_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_diff),cmap=magma_cmap), cax=plt.subplot(gs[2:,-1]))
    
    create_directory(save_location + "/2dhist/")
    scal_miss_class_fig.savefig(save_location + "/2dhist/" + title + "_scaled_miss_class.png")
    
#########################################################################################################################################################
    widths = [4,4,4,.5]
    heights = [4,4]
    miss_class_fig = plt.figure()
    miss_class_fig.suptitle("Misclassified Particles " + title)
    gs = miss_class_fig.add_gridspec(2,4,width_ratios = widths, height_ratios = heights)

    min_ptl = 30
    
    phase_plot(miss_class_fig.add_subplot(gs[0,0]), inc_inf_r, inc_inf_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(miss_class_fig.add_subplot(gs[0,1]), inc_inf_r, inc_inf_tv, min_ptl, max_ptl, range=[[0,max_r],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="Label: Infall Real: Orbit")
    phase_plot(miss_class_fig.add_subplot(gs[0,2]), inc_inf_rv, inc_inf_tv, min_ptl, max_ptl, range=[[min_rv,max_rv],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(miss_class_fig.add_subplot(gs[1,0]), inc_orb_r, inc_orb_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(miss_class_fig.add_subplot(gs[1,1]), inc_orb_r, inc_orb_tv, min_ptl, max_ptl, range=[[0,max_r],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="Label: Orbit Real: Infall")
    phase_plot(miss_class_fig.add_subplot(gs[1,2]), inc_orb_rv, inc_orb_tv, min_ptl, max_ptl, range=[[min_rv,max_rv],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    
    color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:,-1]))
    
    miss_class_fig.savefig(save_location + "/2dhist/" + title + "_miss_class.png")

def plot_r_rv_tv_graph(orb_inf, r, rv, tv, correct_orb_inf, title, num_bins, show, save, save_location, model_save_location):
    create_directory(save_location + "2dhist/")
    print(save_location + "2dhist/")
    mpl.rcParams.update({'font.size': 8})
    plt.rcParams['figure.constrained_layout.use'] = True

    min_ptl = 1e-3

    max_r = np.max(r)
    max_rv = np.max(rv)
    min_rv = np.min(rv)
    max_tv = np.max(tv)
    min_tv = np.min(tv)
    
    ml_inf_r, ml_orb_r = split_orb_inf(r, orb_inf)
    ml_inf_rv, ml_orb_rv = split_orb_inf(rv, orb_inf)
    ml_inf_tv, ml_orb_tv = split_orb_inf(tv, orb_inf)
    
    act_inf_r, act_orb_r = split_orb_inf(r, correct_orb_inf)
    act_inf_rv, act_orb_rv = split_orb_inf(rv, correct_orb_inf)
    act_inf_tv, act_orb_tv = split_orb_inf(tv, correct_orb_inf)

    ml_max_ptl, ml_orb_r_rv, ml_orb_r_tv, ml_orb_rv_tv, ml_inf_r_rv, ml_inf_r_tv, ml_inf_rv_tv = create_hist_max_ptl(min_ptl,min_ptl, ml_inf_r, ml_orb_r, ml_inf_rv, ml_orb_rv, ml_inf_tv, ml_orb_tv, num_bins, max_r, max_rv, min_rv, max_tv, min_tv)
    act_max_ptl, act_orb_r_rv, act_orb_r_tv, act_orb_rv_tv, act_inf_r_rv, act_inf_r_tv, act_inf_rv_tv = create_hist_max_ptl(min_ptl,min_ptl, act_inf_r, act_orb_r, act_inf_rv, act_orb_rv, act_inf_tv, act_orb_tv, num_bins, max_r, max_rv, min_rv, max_tv, min_tv, bin_r_rv=ml_orb_r_rv[1:], bin_r_tv=ml_orb_r_tv[1:],bin_rv_tv=ml_orb_rv_tv[1:])    
    
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
    
    phase_plot(inf_fig.add_subplot(gs[0,0]), ml_inf_r, ml_inf_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(inf_fig.add_subplot(gs[0,1]), ml_inf_r, ml_inf_tv, min_ptl, max_ptl, range=[[0,max_r],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="ML Predictions")
    phase_plot(inf_fig.add_subplot(gs[0,2]), ml_inf_rv, ml_inf_tv, min_ptl, max_ptl, range=[[min_rv,max_rv],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$")
    phase_plot(inf_fig.add_subplot(gs[1,0]), act_inf_r, act_inf_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(inf_fig.add_subplot(gs[1,1]), act_inf_r, act_inf_tv, min_ptl, max_ptl, range=[[0,max_r],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="Actual Distribution")
    phase_plot(inf_fig.add_subplot(gs[1,2]), act_inf_rv, act_inf_tv, min_ptl, max_ptl, range=[[min_rv,max_rv],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$")
    
    inf_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]))
    
    inf_fig.savefig(save_location + "/2dhist/" + title + "_ptls_inf.png")
    
#########################################################################################################################################################
    
    orb_fig = plt.figure()
    orb_fig.suptitle("Orbiting Particles: " + title)
    gs = orb_fig.add_gridspec(2,4,width_ratios = widths, height_ratios = heights)
    
    phase_plot(orb_fig.add_subplot(gs[0,0]), ml_orb_r, ml_orb_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(orb_fig.add_subplot(gs[0,1]), ml_orb_r, ml_orb_tv, min_ptl, max_ptl, range=[[0,max_r],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="ML Predictions")
    phase_plot(orb_fig.add_subplot(gs[0,2]), ml_orb_rv, ml_orb_tv, min_ptl, max_ptl, range=[[min_rv,max_rv],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$")
    phase_plot(orb_fig.add_subplot(gs[1,0]), act_orb_r, act_orb_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$")
    phase_plot(orb_fig.add_subplot(gs[1,1]), act_orb_r, act_orb_tv, min_ptl, max_ptl, range=[[0,max_r],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_t/v_{200m}$", title="Actual Distribution")
    phase_plot(orb_fig.add_subplot(gs[1,2]), act_orb_rv, act_orb_tv, min_ptl, max_ptl, range=[[min_rv,max_rv],[min_tv,max_tv]], num_bins=num_bins, cmap=cmap, x_label="$v_r/v_{200m}$", y_label="$v_t/v_{200m}$")
    
    orb_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]), pad = 0.1)
    
    orb_fig.savefig(save_location + "/2dhist/" + title + "_ptls_orb.png")    
    
#########################################################################################################################################################
    
    only_r_rv_widths = [4,4,.5]
    only_r_rv_heights = [4,4]
    only_r_rv_fig = plt.figure()
    only_r_rv_fig.suptitle("Radial Velocity Versus Radius: " + title)
    gs = only_r_rv_fig.add_gridspec(2,3,width_ratios = only_r_rv_widths, height_ratios = only_r_rv_heights)
    
    phase_plot(only_r_rv_fig.add_subplot(gs[0,0]), ml_orb_r, ml_orb_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="ML Predicted Orbiting Particles")
    phase_plot(only_r_rv_fig.add_subplot(gs[0,1]), ml_inf_r, ml_inf_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="ML Predicted Infalling Particles")
    phase_plot(only_r_rv_fig.add_subplot(gs[1,0]), act_orb_r, act_orb_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="Actual Orbiting Particles")
    phase_plot(only_r_rv_fig.add_subplot(gs[1,1]), act_inf_r, act_inf_rv, min_ptl, max_ptl, range=[[0,max_r],[min_rv,max_rv]], num_bins=num_bins, cmap=cmap, x_label="$r/R_{200m}$", y_label="$v_r/v_{200m}$", title="Actual Infalling Particles")

    
    only_r_rv_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]))
    only_r_rv_fig.savefig(save_location + "/2dhist/" + title + "_only_r_rv.png")
    
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
    
    err_fig.savefig(save_location + "/2dhist/" + title + "_percent_error.png") 
    
    plot_incorrectly_classified(correct_labels=correct_orb_inf, ml_labels=orb_inf, r=r, rv=rv, tv=tv, num_bins=num_bins, title=title, save_location=save_location, model_save_location=model_save_location, act_orb_r_rv=act_orb_r_rv, act_orb_r_tv=act_orb_r_tv, act_orb_rv_tv=act_orb_rv_tv, act_inf_r_rv=act_inf_r_rv, act_inf_r_tv=act_inf_r_tv, act_inf_rv_tv=act_inf_rv_tv)

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

def graph_correlation_matrix(data, save_location, title, show, save):
    return
    # create_directory(save_location + "/corr_matrix/")
    # mpl.rcParams.update({'font.size': 12})

    # heatmap = sns.heatmap(data.corr(), annot = True, cbar = True)
    # heatmap.set_title("Feature Correlation Heatmap")
    # heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=45)

    # if show:
    #     plt.show()
    # if save:
    #     fig = heatmap.get_figure()
    #     fig.set_size_inches(21, 13)
    #     fig.savefig(save_location + "/corr_matrix/" + title + ".png")
    # plt.close()
    
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
