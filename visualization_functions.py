import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from calculation_functions import calculate_distance
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report
from data_and_loading_functions import check_pickle_exist_gadget, create_directory
import general_plotting as gp
from textwrap import wrap

def compare_density_prf(radii, actual_prf_all, actual_prf_1halo, mass, orbit_assn, title, save_location, show_graph = False, save_graph = False):
    create_directory(save_location + "dens_prfl_ratio/")
    # Create bins for the density profile calculation
    num_prf_bins = actual_prf_all.shape[0]
    start_prf_bins = 0.01 # set by parameters in SPARTA
    end_prf_bins = 3.0 # set by parameters in SPARTA
    prf_bins = np.logspace(np.log10(start_prf_bins), np.log10(end_prf_bins), num_prf_bins)
    
    calculated_prf_orb = np.zeros(num_prf_bins)
    calculated_prf_inf = np.zeros(num_prf_bins)
    calculated_prf_all = np.zeros(num_prf_bins)
    start_bin = 0

    orbit_radii = radii[np.where(orbit_assn == 1)[0]]
    infall_radii = radii[np.where(orbit_assn == 0)[0]]

    for i in range(num_prf_bins):
        end_bin = prf_bins[i]  
        
        orb_radii_within_range = np.where((orbit_radii >= start_bin) & (orbit_radii < end_bin))[0]
        if orb_radii_within_range.size != 0 and i != 0:
            calculated_prf_orb[i] = calculated_prf_orb[i - 1] + orb_radii_within_range.size * mass
        elif i == 0:
            calculated_prf_orb[i] = orb_radii_within_range.size * mass
        else:
            calculated_prf_orb[i] = calculated_prf_orb[i - 1]
            
        inf_radii_within_range = np.where((infall_radii >= start_bin) & (infall_radii < end_bin))[0]
        if inf_radii_within_range.size != 0 and i != 0:
            calculated_prf_inf[i] = calculated_prf_inf[i - 1] + inf_radii_within_range.size * mass
        elif i == 0:
            calculated_prf_inf[i] = inf_radii_within_range.size * mass
        else:
            calculated_prf_inf[i] = calculated_prf_inf[i - 1]
            
        radii_within_range = np.where((radii >= start_bin) & (radii < end_bin))[0]
        if radii_within_range.size != 0 and i != 0:
            calculated_prf_all[i] = calculated_prf_all[i - 1] + radii_within_range.size * mass
        elif i == 0:
            calculated_prf_all[i] = radii_within_range.size * mass
        else:
            calculated_prf_all[i] = calculated_prf_all[i - 1]

        start_bin = end_bin

    prf_bins = np.insert(prf_bins,0,0)
    middle_bins = (prf_bins[1:] + prf_bins[:-1]) / 2

    fig, ax = plt.subplots(1,2, layout = "tight")

    ax[0].plot(middle_bins, calculated_prf_all/actual_prf_all, 'r', label = "ML / SPARTA prf all")
    #ax.plot(middle_bins, actual_prf_all, 'c--', label = "SPARTA profile all")

    ax[0].plot(middle_bins, calculated_prf_orb/actual_prf_1halo, 'b', label = "ML / SPARTA profile orb")
    ax[0].plot(middle_bins, calculated_prf_inf/(actual_prf_all - actual_prf_1halo), 'g', label = "ML / SPARTA profile inf")
    
    #ax.plot(middle_bins, actual_prf_1halo, 'm--', label = "SPARTA profile orbit")
    #ax.plot(middle_bins, actual_prf_all - actual_prf_1halo, 'y--', label = "SPARTA profile inf")
    
    ax[0].set_title(wrap("ML Predicted  / Actual Density Profile for nu: " + title))
    ax[0].set_xlabel("radius $r/R_{200m}$")
    ax[0].set_ylabel("ML Dens Prf / Act Dens Prf")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].legend()

    ax[1].plot(middle_bins, calculated_prf_all, 'r-', label = "ML prf all")
    ax[1].plot(middle_bins, calculated_prf_orb, 'b-', label = "ML prf orb")
    ax[1].plot(middle_bins, calculated_prf_inf, 'g-', label = "ML prf inf")
    ax[1].plot(middle_bins, actual_prf_all, 'r--', label = "SPARTA prf all")
    ax[1].plot(middle_bins, actual_prf_1halo, 'b--', label = "SPARTA prf orb")
    ax[1].plot(middle_bins, (actual_prf_all - actual_prf_1halo), 'g--', label = "SPARTA prf inf")
    ax[1].set_title(wrap("ML Predicted vs Actual Density Profile for nu: " + title))
    ax[1].set_xlabel("radius $r/R_{200m}$")
    ax[1].set_ylabel("Mass $M_/odot$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].legend()
    
    if save_graph:
        fig.set_size_inches(21, 13)
        fig.savefig(save_location + "dens_prfl_ratio/" + title + ".png", bbox_inches='tight')
    if show_graph:
        plt.show()
    plt.close()


def brute_force(curr_particles_pos, r200, halo_x, halo_y, halo_z):
    within_box = curr_particles_pos[np.where((curr_particles_pos[:,0] < r200 + halo_x) & (curr_particles_pos[:,0] > r200 - halo_x) & (curr_particles_pos[:,1] < r200 + halo_y) & (curr_particles_pos[:,1] > r200 - halo_y) & (curr_particles_pos[:,2] < r200 + halo_z) & (curr_particles_pos[:,2] > r200 - halo_z))]
    brute_radii = calculate_distance(halo_x, halo_y, halo_z, within_box[:,0], within_box[:,1], within_box[:,2], within_box.shape[0])
    return within_box[np.where(brute_radii <= r200)]

#TODO add brute force comparison graph

#TODO add radial vel vs position graph

def rad_vel_vs_radius_plot(rad_vel, hubble_vel, start_nu, end_nu, color, ax = None):
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

def create_hist_max_ptl(min_ptl, radius, radial_vel, tang_vel, labels, num_bins, max_radius, max_rad_vel, min_rad_vel, max_tang_vel, min_tang_vel):
    inf_radius = radius[np.where(labels == 0)]
    orb_radius = radius[np.where(labels == 1)]
    inf_rad_vel = radial_vel[np.where(labels == 0)]
    orb_rad_vel = radial_vel[np.where(labels == 1)]
    inf_tang_vel = tang_vel[np.where(labels == 0)]
    orb_tang_vel = tang_vel[np.where(labels == 1)]

    np_hist1, x_edge1, y_edge1 = np.histogram2d(orb_radius, orb_rad_vel, bins=num_bins, range=[[0,max_radius],[min_rad_vel,max_rad_vel]])
    np_hist2, x_edge2, y_edge2 = np.histogram2d(orb_radius, orb_tang_vel, bins=num_bins, range=[[0,max_radius],[min_tang_vel,max_tang_vel]])
    np_hist3, x_edge3, y_edge3 = np.histogram2d(orb_tang_vel, orb_rad_vel, bins=num_bins, range=[[min_tang_vel,max_tang_vel],[min_rad_vel,max_rad_vel]])
    np_hist4, x_edge4, y_edge4 = np.histogram2d(inf_radius, inf_rad_vel, bins=num_bins, range=[[0,max_radius],[min_rad_vel,max_rad_vel]])
    np_hist5, x_edge5, y_edge5 = np.histogram2d(inf_radius, inf_tang_vel, bins=num_bins, range=[[0,max_radius],[min_tang_vel,max_tang_vel]])
    np_hist6, x_edge6, y_edge6 = np.histogram2d(inf_tang_vel, inf_rad_vel, bins=num_bins, range=[[min_tang_vel,max_tang_vel],[min_rad_vel,max_rad_vel]])
    np_hist7, x_edge7, y_edge7 = np.histogram2d(radius, radial_vel, bins=num_bins, range=[[0,max_radius],[min_rad_vel,max_rad_vel]])
    np_hist8, x_edge8, y_edge8 = np.histogram2d(radius, tang_vel, bins=num_bins, range=[[0,max_radius],[min_tang_vel,max_tang_vel]])
    np_hist9, x_edge9, y_edge9 = np.histogram2d(tang_vel, radial_vel, bins=num_bins, range=[[min_tang_vel,max_tang_vel],[min_rad_vel,max_rad_vel]])

    np_hist1[np_hist1 < min_ptl] = min_ptl
    np_hist2[np_hist2 < min_ptl] = min_ptl
    np_hist3[np_hist3 < min_ptl] = min_ptl
    np_hist4[np_hist4 < min_ptl] = min_ptl
    np_hist5[np_hist5 < min_ptl] = min_ptl
    np_hist6[np_hist6 < min_ptl] = min_ptl
    np_hist7[np_hist7 < min_ptl] = min_ptl
    np_hist8[np_hist8 < min_ptl] = min_ptl
    np_hist9[np_hist9 < min_ptl] = min_ptl

    #orbital divided by infall
    ratio_rad_rvel = np_hist1/(np_hist1 + np_hist4)
    ratio_rad_rvel[(np.isnan(ratio_rad_rvel))] = 0
    ratio_rad_rvel = np.round(ratio_rad_rvel,2)

    ratio_rad_tvel = np_hist2/(np_hist2 + np_hist5)
    ratio_rad_tvel[(np.isnan(ratio_rad_tvel))] = 0
    ratio_rad_tvel = np.round(ratio_rad_tvel,2)

    ratio_rvel_tvel = np_hist3/(np_hist3 + np_hist6)
    ratio_rvel_tvel[(np.isnan(ratio_rvel_tvel))] = 0
    ratio_rvel_tvel = np.round(ratio_rvel_tvel,2)

    max_ptl = np.max(np_hist1)
    if np.max(np_hist2) > max_ptl:
        max_ptl = np.max(np_hist2)
    if np.max(np_hist3) > max_ptl:
        max_ptl = np.max(np_hist3)
    if np.max(np_hist4) > max_ptl:
        max_ptl = np.max(np_hist4)
    if np.max(np_hist5) > max_ptl:
        max_ptl = np.max(np_hist5)    
    if np.max(np_hist6) > max_ptl:
        max_ptl = np.max(np_hist6)
    
    return max_ptl, inf_radius, orb_radius, inf_rad_vel, orb_rad_vel, inf_tang_vel, orb_tang_vel, np_hist1, np_hist2, np_hist3, np_hist4, np_hist5, np_hist6, np_hist7, np_hist8, np_hist9

def percent_error(pred, act):
    return ((pred - act))/act

def bigger(new, old):
    if new > old:
        return new
    else:
        return old
def smaller(new,old):
    if new < old:
        return new
    else:
        return old

def plot_radius_rad_vel_tang_vel_graphs(orb_inf, radius, radial_vel, tang_vel, correct_orb_inf, title, num_bins, start_nu, end_nu, show, save, save_location):
    create_directory(save_location + "/2dhist/")
    mpl.rcParams.update({'font.size': 8})
    min_ptl = 30

    max_radius = np.max(radius)
    max_rad_vel = np.max(radial_vel)
    min_rad_vel = np.min(radial_vel)
    max_tang_vel = np.max(tang_vel)
    min_tang_vel = np.min(tang_vel)

    ml_max_ptl, ml_inf_radius, ml_orb_radius, ml_inf_rad_vel, ml_orb_rad_vel, ml_inf_tang_vel, ml_orb_tang_vel, ml_hist1, ml_hist2, ml_hist3, ml_hist4, ml_hist5, ml_hist6, ml_hist7, ml_hist8, ml_hist9 = create_hist_max_ptl(min_ptl, radius, radial_vel, tang_vel, orb_inf, num_bins, max_radius, max_rad_vel, min_rad_vel, max_tang_vel, min_tang_vel)
    act_max_ptl, act_inf_radius, act_orb_radius, act_inf_rad_vel, act_orb_rad_vel, act_inf_tang_vel, act_orb_tang_vel, act_hist1, act_hist2, act_hist3, act_hist4, act_hist5, act_hist6, act_hist7, act_hist8, act_hist9 = create_hist_max_ptl(min_ptl, radius, radial_vel, tang_vel, correct_orb_inf, num_bins, max_radius, max_rad_vel, min_rad_vel, max_tang_vel, min_tang_vel)    
    
    per_err_1 = percent_error(ml_hist1, act_hist1)
    per_err_2 = percent_error(ml_hist2, act_hist2)
    per_err_3 = percent_error(ml_hist3, act_hist3)
    per_err_4 = percent_error(ml_hist4, act_hist4)
    per_err_5 = percent_error(ml_hist5, act_hist5)
    per_err_6 = percent_error(ml_hist6, act_hist6)
    per_err_7 = percent_error(ml_hist7, act_hist7)
    per_err_8 = percent_error(ml_hist8, act_hist8)
    per_err_9 = percent_error(ml_hist9, act_hist9)

    max_err = np.max(per_err_1)
    max_err = bigger(np.max(per_err_2), max_err)
    max_err = bigger(np.max(per_err_3), max_err)
    max_err = bigger(np.max(per_err_4), max_err)
    max_err = bigger(np.max(per_err_5), max_err)
    max_err = bigger(np.max(per_err_6), max_err)

    min_err = np.min(per_err_1)
    min_err = smaller(np.max(per_err_2), max_err)
    min_err = smaller(np.max(per_err_3), max_err)
    min_err = smaller(np.max(per_err_4), max_err)
    min_err = smaller(np.max(per_err_5), max_err)
    min_err = smaller(np.max(per_err_6), max_err)

    if ml_max_ptl > act_max_ptl:
        max_ptl = ml_max_ptl
    else:
        max_ptl = act_max_ptl
  
    cmap = plt.get_cmap("inferno")
    per_err_cmap = plt.get_cmap("inferno")

    widths = [4,4,4,.5]
    heights = [4,4,4]
    
    inf_fig = plt.figure(constrained_layout=True)
    inf_fig.suptitle("Infalling Particles nu:" + str(start_nu) + " to " + str(end_nu))
    gs = inf_fig.add_gridspec(3,4,width_ratios = widths, height_ratios = heights)
    
    ml_inf_r_rv = inf_fig.add_subplot(gs[0,0])
    ml_inf_r_rv.hist2d(ml_inf_radius, ml_inf_rad_vel, num_bins, [[0,max_radius],[min_rad_vel,max_rad_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    ml_inf_r_rv.set_xlabel("$r/R_{200m}$")
    ml_inf_r_rv.set_ylabel("$v_r/v_{200m}$")
    
    ml_inf_r_tv = inf_fig.add_subplot(gs[0,1])
    ml_inf_r_tv.hist2d(ml_inf_radius, ml_inf_tang_vel, num_bins, [[0,max_radius],[min_tang_vel,max_tang_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    ml_inf_r_tv.set_xlabel("$r/R_{200m}$")
    ml_inf_r_tv.set_ylabel("$v_t/v_{200m}$")
    ml_inf_r_tv.set_title("ML Predictions")
    
    ml_inf_rv_tv = inf_fig.add_subplot(gs[0,2])
    ml_inf_rv_tv.hist2d(ml_inf_rad_vel, ml_inf_tang_vel, num_bins, [[min_rad_vel,max_rad_vel],[min_tang_vel,max_tang_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    ml_inf_rv_tv.set_xlabel("$v_r/v_{200m}$")
    ml_inf_rv_tv.set_ylabel("$v_t/v_{200m}$")
    
    act_inf_r_rv = inf_fig.add_subplot(gs[1,0])
    act_inf_r_rv.hist2d(act_inf_radius, act_inf_rad_vel, num_bins, [[0,max_radius],[min_rad_vel,max_rad_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    act_inf_r_rv.set_xlabel("$r/R_{200m}$")
    act_inf_r_rv.set_ylabel("$v_r/v_{200m}$")
    
    act_inf_r_tv = inf_fig.add_subplot(gs[1,1])
    act_inf_r_tv.hist2d(act_inf_radius, act_inf_tang_vel, num_bins, [[0,max_radius],[min_tang_vel,max_tang_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    act_inf_r_tv.set_xlabel("$r/R_{200m}$")
    act_inf_r_tv.set_ylabel("$v_t/v_{200m}$")
    act_inf_r_tv.set_title("Actual Labels")
    
    act_inf_rv_tv = inf_fig.add_subplot(gs[1,2])
    act_inf_rv_tv.hist2d(act_inf_rad_vel, act_inf_tang_vel, num_bins, [[min_rad_vel,max_rad_vel],[min_tang_vel,max_tang_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    act_inf_rv_tv.set_xlabel("$v_r/v_{200m}$")
    act_inf_rv_tv.set_ylabel("$v_t/v_{200m}$")
    
    inf_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]))
    
    perr_inf_r_rv = inf_fig.add_subplot(gs[2,0])
    inf_imshow_img = perr_inf_r_rv.imshow(per_err_1, cmap = per_err_cmap, vmin = min_err, vmax = max_err, origin = 'lower', aspect = 'auto', extent = [0,max_radius,min_rad_vel,max_rad_vel])
    perr_inf_r_rv.set_xlabel("$r/R_{200m}$")
    perr_inf_r_rv.set_ylabel("$v_r/v_{200m}$")
    
    perr_inf_r_tv = inf_fig.add_subplot(gs[2,1])
    perr_inf_r_tv.imshow(per_err_2, cmap = per_err_cmap, vmin = min_err, vmax = max_err, origin = 'lower', aspect = 'auto', extent = [0,max_radius,min_tang_vel,max_tang_vel])
    perr_inf_r_tv.set_xlabel("$r/R_{200m}$")
    perr_inf_r_tv.set_ylabel("$v_t/v_{200m}$")
    perr_inf_r_tv.set_title("Percent Error")
    
    perr_inf_rv_tv = inf_fig.add_subplot(gs[2,2])
    perr_inf_rv_tv.imshow(per_err_3, cmap = per_err_cmap, vmin = min_err, vmax = max_err, origin = 'lower', aspect = 'auto', extent = [min_rad_vel,max_rad_vel,min_tang_vel,max_tang_vel])
    perr_inf_rv_tv.set_xlabel("$v_r/v_{200m}$")
    perr_inf_rv_tv.set_ylabel("$v_t/v_{200m}$")
    
    inf_perr_color_bar = plt.colorbar(inf_imshow_img, cax=plt.subplot(gs[2,-1]))
    
    inf_fig.savefig(save_location + "/2dhist/nu:" + str(start_nu) + " to " + str(end_nu) + "_ptls_inf.png")
###################################################
    orb_fig = plt.figure(constrained_layout=True)
    orb_fig.suptitle("Orbiting Particles nu:" + str(start_nu) + " to " + str(end_nu))
    gs = orb_fig.add_gridspec(3,4,width_ratios = widths, height_ratios = heights)
    
    ml_orb_r_rv = orb_fig.add_subplot(gs[0,0])
    ml_orb_r_rv.hist2d(ml_orb_radius, ml_orb_rad_vel, num_bins, [[0,max_radius],[min_rad_vel,max_rad_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    ml_orb_r_rv.set_xlabel("$r/R_{200m}$")
    ml_orb_r_rv.set_ylabel("$v_r/v_{200m}$")
    
    ml_orb_r_tv = orb_fig.add_subplot(gs[0,1])
    ml_orb_r_tv.hist2d(ml_orb_radius, ml_orb_tang_vel, num_bins, [[0,max_radius],[min_tang_vel,max_tang_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    ml_orb_r_tv.set_xlabel("$r/R_{200m}$")
    ml_orb_r_tv.set_ylabel("$v_t/v_{200m}$")
    ml_orb_r_tv.set_title("ML Predictions")
    
    ml_orb_rv_tv = orb_fig.add_subplot(gs[0,2])
    ml_orb_rv_tv.hist2d(ml_orb_rad_vel, ml_orb_tang_vel, num_bins, [[min_rad_vel,max_rad_vel],[min_tang_vel,max_tang_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    ml_orb_rv_tv.set_xlabel("$v_r/v_{200m}$")
    ml_orb_rv_tv.set_ylabel("$v_t/v_{200m}$")
    
    act_orb_r_rv = orb_fig.add_subplot(gs[1,0])
    act_orb_r_rv.hist2d(act_orb_radius, act_orb_rad_vel, num_bins, [[0,max_radius],[min_rad_vel,max_rad_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    act_orb_r_rv.set_xlabel("$r/R_{200m}$")
    act_orb_r_rv.set_ylabel("$v_r/v_{200m}$")
    
    act_orb_r_tv = orb_fig.add_subplot(gs[1,1])
    act_orb_r_tv.hist2d(act_orb_radius, act_orb_tang_vel, num_bins, [[0,max_radius],[min_tang_vel,max_tang_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    act_orb_r_tv.set_xlabel("$r/R_{200m}$")
    act_orb_r_tv.set_ylabel("$v_t/v_{200m}$")
    act_orb_r_tv.set_title("Actual Labels")
    
    act_orb_rv_tv = orb_fig.add_subplot(gs[1,2])
    act_orb_rv_tv.hist2d(act_orb_rad_vel, act_orb_tang_vel, num_bins, [[min_rad_vel,max_rad_vel],[min_tang_vel,max_tang_vel]], False, None, min_ptl, cmap = cmap, norm = "log", vmin = min_ptl, vmax = max_ptl)
    act_orb_rv_tv.set_xlabel("$v_r/v_{200m}$")
    act_orb_rv_tv.set_ylabel("$v_t/v_{200m}$")
    
    inf_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=min_ptl, vmax=max_ptl),cmap=cmap), cax=plt.subplot(gs[:2,-1]), pad = 0.1)
    
    perr_orb_r_rv = orb_fig.add_subplot(gs[2,0])
    orb_imshow_img = perr_orb_r_rv.imshow(per_err_4, cmap = per_err_cmap, vmin = min_err, vmax = max_err, origin = 'lower', aspect = 'auto', extent = [0,max_radius,min_rad_vel,max_rad_vel])
    perr_orb_r_rv.set_xlabel("$r/R_{200m}$")
    perr_orb_r_rv.set_ylabel("$v_r/v_{200m}$")
    
    perr_orb_r_tv = orb_fig.add_subplot(gs[2,1])
    perr_orb_r_tv.imshow(per_err_5, cmap = per_err_cmap, vmin = min_err, vmax = max_err, origin = 'lower', aspect = 'auto', extent = [0,max_radius,min_tang_vel,max_tang_vel])
    perr_orb_r_tv.set_xlabel("$r/R_{200m}$")
    perr_orb_r_tv.set_ylabel("$v_t/v_{200m}$")
    perr_orb_r_tv.set_title("Percent Error")

    perr_orb_rv_tv = orb_fig.add_subplot(gs[2,2])
    perr_orb_rv_tv.imshow(per_err_6, cmap = per_err_cmap, vmin = min_err, vmax = max_err, origin = 'lower', aspect = 'auto', extent = [min_rad_vel,max_rad_vel,min_tang_vel,max_tang_vel])
    perr_orb_rv_tv.set_xlabel("$v_r/v_{200m}$")
    perr_orb_rv_tv.set_ylabel("$v_t/v_{200m}$")
    
    orb_perr_color_bar = plt.colorbar(orb_imshow_img, cax=plt.subplot(gs[2,-1]), pad = 0.1)
    
    orb_fig.savefig(save_location + "/2dhist/nu:" + str(start_nu) + " to " + str(end_nu) + "_ptls_orb.png")    

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
    create_directory(save_location + "/corr_matrix/")
    mpl.rcParams.update({'font.size': 12})

    heatmap = sns.heatmap(data.corr(), annot = True, cbar = True)
    heatmap.set_title("Feature Correlation Heatmap")
    heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=45)

    if show:
        plt.show()
    if save:
        fig = heatmap.get_figure()
        fig.set_size_inches(21, 13)
        fig.savefig(save_location + "/corr_matrix/" + title + ".png")
    plt.close()
    
def graph_acc_by_bin(pred_orb_inf, corr_orb_inf, radius, num_bins, start_nu, end_nu, plot, save, save_location):
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
    
    # plot_types = ["stairs","stairs","stairs","stairs"]
    # plot_size = (2,2)

    # x_data = np.zeros((num_bins,3,4))
    # y_data = np.zeros((num_bins,3,4))

    # x_data[:,0,0] = all_accuracy
    # x_data[:,1,0] = inf_accuracy
    # x_data[:,2,0] = orb_accuracy
    # x_data[:,0,1] = all_accuracy
    # x_data[:,1,1] = all_accuracy
    # x_data[:,2,1] = all_accuracy
    # x_data[:,0,2] = inf_accuracy
    # x_data[:,1,2] = inf_accuracy
    # x_data[:,2,2] = inf_accuracy
    # x_data[:,0,3] = orb_accuracy
    # x_data[:,1,3] = orb_accuracy
    # x_data[:,2,3] = orb_accuracy
    # x_data[:,0,0] = bins
    # x_data[:,1,0] = bins
    # x_data[:,2,0] = bins
    # x_data[:,:,1] = bins
    # x_data[:,:,2] = bins
    # x_data[:,:,3] = bins

    # y_lim = [(-0.1,1.1),(-0.1,1.1),(-0.1,1.1),(-0.1,1.1)]
    # x_label = ["radius $r/R_{200m}$","radius $r/R_{200m}$","radius $r/R_{200m}$","radius $r/R_{200m}$"]
    # y_label = ["Accuracy","Accuracy","Accuracy","Accuracy"]
    # save_location = save_location + "error_by_rad_graphs/error_by_rad_" + str(start_nu) + "_" + str(end_nu) + ".png"
    # line_labels = ["All ptl", "inf ptl", "orb ptl", "All ptl","All ptl","All ptl","inf ptl","inf ptl","inf ptl","orb ptl","orb ptl","orb ptl"]

    # accuracy_plotter = gp.plot_determiner(plot_types=plot_types, plot_size=plot_size, X=x_data, Y=y_data, ylim=y_lim, x_label=x_label, y_label=y_label, line_labels=line_labels, fig_title="Accuracy Per r/R200m Bin", save_location=save_location, save=True)
    # accuracy_plotter.plot()
    # accuracy_plotter.save()
    
    fig, ax = plt.subplots(2,2, layout="constrained")
    fig.suptitle("Accuracy by Radius for nu: " + str(start_nu) + "_" + str(end_nu))
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
        fig.savefig(save_location + "error_by_rad_graphs/error_by_rad_" + str(start_nu) + "_" + str(end_nu) + ".png")
        plt.close()
        
def feature_dist(features, labels, save_name, plot, save, save_location):
    tot_plts = features.shape[1]
    num_col = 3
    
    num_rows = tot_plts // num_col
    if tot_plts % num_col != 0:
        num_rows += 1
    
    position = np.arange(1, tot_plts + 1)
    
    fig = plt.figure(1)
    fig = plt.figure(tight_layout=True)
    
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
        
        