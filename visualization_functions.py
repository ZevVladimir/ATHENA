import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from calculation_functions import calculate_distance
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report
from data_and_loading_functions import check_pickle_exist_gadget

def compare_density_prf(radii, actual_prf_all, actual_prf_1halo, mass, orbit_assn, num, start_nu, end_nu, show_graph = False, save_graph = False):
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

    fig, ax = plt.subplots(1,1)

    ax.plot(middle_bins, calculated_prf_all, 'r-', label = "my code profile all")
    ax.plot(middle_bins, actual_prf_all, 'c--', label = "SPARTA profile all")

    ax.plot(middle_bins, calculated_prf_orb, 'b-', label = "my code profile orb")
    ax.plot(middle_bins, calculated_prf_inf, 'g-', label = "my code profile inf")
    
    ax.plot(middle_bins, actual_prf_1halo, 'm--', label = "SPARTA profile orbit")
    ax.plot(middle_bins, actual_prf_all - actual_prf_1halo, 'y--', label = "SPARTA profile inf")
    
    ax.set_title("1Halo Density Profile")
    ax.set_xlabel("radius $r/R_{200m}$")
    ax.set_ylabel("Mass of halo $M_{\odot}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    if save_graph:
        fig.savefig("/home/zvladimi/MLOIS/Random_figures/density_prf_" + str(start_nu) + "-" + str(end_nu) + "_" + str(num) + ".png")
        mpl.pyplot.close()
    if show_graph:
        plt.show()
    ax.clear()
    fig.clear()


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

def plot_radius_rad_vel_tang_vel_graphs(orb_inf, radius, radial_vel, tang_vel, correct_orb_inf, title, num_bins, start_nu, end_nu, plot, save):
    mpl.rcParams.update({'font.size': 8})
    inf_radius = radius[np.where(orb_inf == 0)]
    orb_radius = radius[np.where(orb_inf == 1)]
    inf_rad_vel = radial_vel[np.where(orb_inf == 0)]
    orb_rad_vel = radial_vel[np.where(orb_inf == 1)]
    inf_tang_vel = tang_vel[np.where(orb_inf == 0)]
    orb_tang_vel = tang_vel[np.where(orb_inf == 1)]
    
    max_radius = np.max(radius)
    max_rad_vel = np.max(radial_vel)
    min_rad_vel = np.min(radial_vel)
    max_tang_vel = np.max(tang_vel)
    min_tang_vel = np.min(tang_vel)
    
    ptl_min = 20
    cmap = plt.get_cmap("inferno")

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(title + " nu:" + str(start_nu) + " to " + str(end_nu))

    sub_fig_titles = ["Orbiting Particles", "Infalling Particles", "Orbit/Infall Ratio"]
    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(sub_fig_titles[row])

        np_hist1, x_edge1, y_edge1 = np.histogram2d(orb_radius, orb_rad_vel, bins=num_bins)
        np_hist2, x_edge2, y_edge2 = np.histogram2d(orb_radius, orb_tang_vel, bins=num_bins)
        np_hist3, x_edge3, y_edge3 = np.histogram2d(orb_tang_vel, orb_rad_vel, bins=num_bins)
        np_hist4, x_edge4, y_edge4 = np.histogram2d(inf_radius, inf_rad_vel, bins=num_bins)
        np_hist5, x_edge5, y_edge5 = np.histogram2d(inf_radius, inf_tang_vel, bins=num_bins)
        np_hist6, x_edge6, y_edge6 = np.histogram2d(inf_tang_vel, inf_rad_vel, bins=num_bins)
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
                
        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)
        if row == 0:
            
            
            hist1 = axs[0].hist2d(orb_radius, orb_rad_vel, bins = num_bins, range = [[0,max_radius],[min_rad_vel,max_rad_vel]], cmap = cmap, vmin = 0, vmax = max_ptl)
            axs[0].set_xlabel("$r/R_{200m}$")
            axs[0].set_ylabel("$v_r/v_{200m}$")
            
            hist2 = axs[1].hist2d(orb_radius, orb_tang_vel, bins = num_bins, range = [[0,max_radius],[min_tang_vel,max_tang_vel]], cmap = cmap, vmin = 0, vmax = max_ptl)
            axs[1].set_xlabel("$r/R_{200m}$")
            axs[1].set_ylabel("$v_t/v_{200m}$")

            hist3 = axs[2].hist2d(orb_tang_vel, orb_rad_vel, bins = num_bins, range = [[min_tang_vel,max_tang_vel], [min_rad_vel,max_rad_vel]], cmap = cmap, vmin = 0, vmax = max_ptl)
            axs[2].set_xlabel("$v_r/v_{200m}$")
            axs[2].set_ylabel("$v_t/v_{200m}$")
            
            subfig.colorbar(hist1[3], ax=axs[-1], pad = 0.1)

        elif row == 1:
            hist4 = axs[0].hist2d(inf_radius, inf_rad_vel, bins = num_bins, range = [[0,max_radius],[min_rad_vel,max_rad_vel]], cmap = cmap, vmin = 0, vmax = max_ptl)
            axs[0].set_xlabel("$r/R_{200m}$")
            axs[0].set_ylabel("$v_r/v_{200m}$")

            hist5 = axs[1].hist2d(inf_radius, inf_tang_vel, bins = num_bins, range = [[0,max_radius],[min_tang_vel,max_tang_vel]], cmap = cmap, vmin = 0, vmax = max_ptl)
            axs[1].set_xlabel("$r/R_{200m}$")
            axs[1].set_ylabel("$v_t/v_{200m}$")

            hist6 = axs[2].hist2d(inf_tang_vel, inf_rad_vel, bins = num_bins, range = [[min_tang_vel,max_tang_vel], [min_rad_vel,max_rad_vel]], cmap = cmap, vmin = 0, vmax = max_ptl)
            axs[2].set_xlabel("$v_r/v_{200m}$")
            axs[2].set_ylabel("$v_t/v_{200m}$")
            subfig.colorbar(hist1[3], ax=axs[-1], pad = 0.1)

        else:
            #orbital divided by infall
            ratio_rad_rvel = hist1[0]/(hist1[0] + hist4[0])
            ratio_rad_rvel[(np.isnan(ratio_rad_rvel))] = 0
            ratio_rad_rvel = np.round(ratio_rad_rvel,2)

            ratio_rad_tvel = hist2[0]/(hist2[0] + hist5[0])
            ratio_rad_tvel[(np.isnan(ratio_rad_tvel))] = 0
            ratio_rad_tvel = np.round(ratio_rad_tvel,2)

            ratio_rvel_tvel = hist3[0]/(hist3[0] + hist6[0])
            ratio_rvel_tvel[(np.isnan(ratio_rvel_tvel))] = 0
            ratio_rvel_tvel = np.round(ratio_rvel_tvel,2)

            hist7 = axs[0].imshow(np.flip(ratio_rad_rvel, axis = 1).T, cmap = cmap, extent = [0, num_bins, 0, num_bins])
            axs[0].set_xlabel("x bin")
            axs[0].set_ylabel("y bin")
            subfig.colorbar(hist7, ax=axs[0], shrink = 0.6, pad = 0.01)

            hist8 = axs[1].imshow(np.flip(ratio_rad_tvel, axis = 1).T, cmap = cmap, extent = [0, num_bins, 0, num_bins])
            axs[1].set_xlabel("x bin")
            axs[1].set_ylabel("y bin")
            subfig.colorbar(hist8, ax=axs[1], shrink = 0.6, pad = 0.01)

            hist9 = axs[2].imshow(np.flip(ratio_rvel_tvel, axis = 1).T, cmap = cmap, extent = [0, num_bins, 0, num_bins])
            # for i in range(num_bins):
            #     for j in range(num_bins):
            axs[2].set_xlabel("x bin")
            axs[2].set_ylabel("y bin")
            subfig.colorbar(hist9, ax=axs[2], shrink = 0.6, pad = 0.01)
            
    if plot:
        plt.show()
    if save:
        fig.savefig("/home/zvladimi/MLOIS/Random_figures/2d_hists/_2d_hist_" + str(start_nu) + "_" + str(end_nu) + "_" + title + ".png", dpi = 1000)
        mpl.pyplot.close()

def graph_feature_importance(feature_names, feature_importance, model_name, plot, save):
    mpl.rcParams.update({'font.size': 16})
    fig2, (plot1) = plt.subplots(1,1)
    import_idxs = np.argsort(feature_importance)
    plot1.barh(feature_names[import_idxs], feature_importance[import_idxs])
    plot1.set_xlabel = ("XGBoost feature importance")
    
    if plot:
        plt.show()
    if save:
        fig2.savefig("/home/zvladimi/MLOIS/Random_figures/feature_importance_" + model_name + ".png")
        mpl.pyplot.close()

def graph_correlation_matrix(data, feature_names):
    mpl.rcParams.update({'font.size': 12})
    heatmap = sns.heatmap(data.corr(), annot = True, cbar = True)
    heatmap.set_title("Feature Correlation Heatmap")
    plt.show()
    
def graph_err_by_bin(pred_orb_inf, corr_orb_inf, radius, num_bins, start_nu, end_nu, plot, save):
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
    
    fig, ax = plt.subplots(1,1)
    ax.stairs(all_accuracy, bins, color = "black", alpha = 0.4, label = "all ptl")    
    ax.stairs(inf_accuracy, bins, color = "blue", alpha = 0.4, label = "inf ptl")
    ax.stairs(orb_accuracy, bins, color = "red", alpha = 0.4, label = "orb ptl")
    ax.set_title("Halo: " + str(start_nu) + "_" + str(end_nu) + " Num ptls: " + str(radius.shape[0]))
    ax.set_xlabel("radius $r/R_{200m}$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.1,1.1)
    plt.legend()
    if plot:
        plt.show()
    if save:
        fig.savefig("/home/zvladimi/MLOIS/Random_figures/error_by_rad_graphs/error_by_rad_" + str(start_nu) + "_" + str(end_nu) + ".png")
        mpl.pyplot.close()
        
def feature_dist(features, labels, save_name, plot, save):
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
        fig.savefig("/home/zvladimi/MLOIS/Random_figures/feature_dist/feature_dist_" + save_name + ".png")
        mpl.pyplot.close()
        
        