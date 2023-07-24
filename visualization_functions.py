import numpy as np
import matplotlib.pyplot as plt
from calculation_functions import calculate_distance
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def plot_radius_rad_vel_tang_vel_graphs(orb_inf, radius, radial_vel, tang_vel, correct_orb_inf):
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
    
    num_bins = 30
    ptl_min = 0
    cmap = plt.get_cmap("inferno")

    fig = plt.figure(constrained_layout=True)
    fig.suptitle('ML Predictions')

    sub_fig_titles = ["Orbiting Particles", "Infalling Particles", "Orbit/Infall Ratio"]
    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(sub_fig_titles[row])

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)
        if row == 0:
            hist1 = axs[0].hist2d(orb_radius, orb_rad_vel, bins = num_bins, range = [[0,max_radius],[min_rad_vel,max_rad_vel]], cmap = cmap, cmin = ptl_min)
            axs[0].set_title("Radial Velocity vs Radius")
            axs[0].set_xlabel("radius $r/R_{200m}$")
            axs[0].set_ylabel("rad vel $v_r/v_{200m}$")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist1[3], cax=cax, orientation='vertical')
            
            hist2 = axs[1].hist2d(orb_radius, orb_tang_vel, bins = num_bins, range = [[0,max_radius],[min_tang_vel,max_tang_vel]], cmap = cmap, cmin = ptl_min)
            axs[1].set_title("Tangential Velocity vs Radius")
            axs[1].set_xlabel("radius $r/R_{200m}$")
            axs[1].set_ylabel("tang vel $v_t/v_{200m}$")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist2[3], cax=cax, orientation='vertical')

            hist3 = axs[2].hist2d(orb_tang_vel, orb_rad_vel, bins = num_bins, range = [[min_tang_vel,max_tang_vel], [min_rad_vel,max_rad_vel]], cmap = cmap, cmin = ptl_min)
            axs[2].set_title("Tangential Velocity vs Radial Velocity")
            axs[2].set_xlabel("rad vel $v_r/v_{200m}$")
            axs[2].set_ylabel("tang vel $v_t/v_{200m}$")
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist3[3], cax=cax, orientation='vertical')

        elif row == 1:
            hist4 = axs[0].hist2d(inf_radius, inf_rad_vel, bins = num_bins, range = [[0,max_radius],[min_rad_vel,max_rad_vel]], cmap = cmap, cmin = ptl_min)
            axs[0].set_title("Radial Velocity vs Radius")
            axs[0].set_xlabel("radius $r/R_{200m}$")
            axs[0].set_ylabel("rad vel $v_r/v_{200m}$")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist4[3], cax=cax, orientation='vertical')

            hist5 = axs[1].hist2d(inf_radius, inf_tang_vel, bins = num_bins, range = [[0,max_radius],[min_tang_vel,max_tang_vel]], cmap = cmap, cmin = ptl_min)
            axs[1].set_title("Tangential Velocity vs Radius")
            axs[1].set_xlabel("radius $r/R_{200m}$")
            axs[1].set_ylabel("tang vel $v_t/v_{200m}$")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist5[3], cax=cax, orientation='vertical')

            hist6 = axs[2].hist2d(inf_tang_vel, inf_rad_vel, bins = num_bins, range = [[min_tang_vel,max_tang_vel], [min_rad_vel,max_rad_vel]], cmap = cmap, cmin = ptl_min)
            axs[2].set_title("Tangential Velocity vs Radial Velocity")
            axs[2].set_xlabel("rad vel $v_r/v_{200m}$")
            axs[2].set_ylabel("tang vel $v_t/v_{200m}$")
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist6[3], cax=cax, orientation='vertical')

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
            axs[0].set_title("Radial Velocity vs Radius")
            axs[0].set_xlabel("radius $r/R_{200m}$")
            axs[0].set_ylabel("rad vel $v_r/v_{200m}$")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist7, cax=cax, orientation='vertical')

            hist8 = axs[1].imshow(np.flip(ratio_rad_tvel, axis = 1).T, cmap = cmap, extent = [0, num_bins, 0, num_bins])
            axs[1].set_title("Tangential Velocity vs Radius")
            axs[1].set_xlabel("radius $r/R_{200m}$")
            axs[1].set_ylabel("tang vel $v_t/v_{200m}$")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist8, cax=cax, orientation='vertical')

            hist9 = axs[2].imshow(np.flip(ratio_rvel_tvel, axis = 1).T, cmap = cmap, extent = [0, num_bins, 0, num_bins])
            # for i in range(num_bins):
            #     for j in range(num_bins):
            #         axs[2].text(i,j, ratio_rvel_tvel[i,j], color="w", ha="center", va="center", fontsize = "xx-small", fontweight="bold")
            axs[2].set_title("Tangential Velocity vs Radial Velocity")
            axs[2].set_xlabel("rad vel $v_r/v_{200m}$")
            axs[2].set_ylabel("tang vel $v_t/v_{200m}$")
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            subfig.colorbar(hist9, cax=cax, orientation='vertical')

    # fig1, (plot1,plot2,plot3) = plt.subplots(1,3)
    # hist1 = plot1.hist2d(orb_radius, orb_rad_vel, bins = num_bins, range = [[0,max_radius],[min_rad_vel,max_rad_vel]], cmap = cmap, cmin = ptl_min)
    # plot1.set_title("Radial Velocity vs Radius")
    # plot1.set_xlabel("radius $r/R_{200m}$")
    # plot1.set_ylabel("rad vel $v_r/v_{200m}$")
    # fig1.colorbar(hist1[3], ax = plot1)
    
    # hist2 = plot2.hist2d(orb_radius, orb_tang_vel, bins = num_bins, range = [[0,max_radius],[min_tang_vel,max_tang_vel]], cmap = cmap, cmin = ptl_min)
    # plot2.set_title("Tangential Velocity vs Radius")
    # plot2.set_xlabel("radius $r/R_{200m}$")
    # plot2.set_ylabel("tang vel $v_t/v_{200m}$")
    # fig1.colorbar(hist2[3], ax = plot2)

    # hist3 = plot3.hist2d(orb_tang_vel, orb_rad_vel, bins = num_bins, range = [[min_rad_vel,max_rad_vel],[min_tang_vel,max_tang_vel]], cmap = cmap, cmin = ptl_min)
    # plot3.set_title("Tangential Velocity vs Radial Velocity")
    # plot3.set_xlabel("rad vel $v_r/v_{200m}$")
    # plot3.set_ylabel("tang vel $v_t/v_{200m}$")
    # fig1.colorbar(hist3[3], ax = plot3)

    # fig1.suptitle("Orbital Particles")

    # fig2, (plot4,plot5,plot6) = plt.subplots(1,3)
    # hist4 = plot4.hist2d(inf_radius, inf_rad_vel, bins = num_bins, range = [[0,max_radius],[min_rad_vel,max_rad_vel]], cmap = cmap, cmin = ptl_min)
    # plot4.set_title("Radial Velocity vs Radius")
    # plot4.set_xlabel("radius $r/R_{200m}$")
    # plot4.set_ylabel("rad vel $v_r/v_{200m}$")
    # fig2.colorbar(hist4[3], ax = plot4)

    # hist5 = plot5.hist2d(inf_radius, inf_tang_vel, bins = num_bins, range = [[0,max_radius],[min_tang_vel,max_tang_vel]], cmap = cmap, cmin = ptl_min)
    # plot5.set_title("Tangential Velocity vs Radius")
    # plot5.set_xlabel("radius $r/R_{200m}$")
    # plot5.set_ylabel("tang vel $v_t/v_{200m}$")
    # fig2.colorbar(hist5[3], ax = plot5)

    # hist6 = plot6.hist2d(inf_tang_vel, inf_rad_vel, bins = num_bins, range = [[min_rad_vel,max_rad_vel],[min_tang_vel,max_tang_vel]], cmap = cmap, cmin = ptl_min)
    # plot6.set_title("Tangential Velocity vs Radial Velocity")
    # plot6.set_xlabel("rad vel $v_r/v_{200m}$")
    # plot6.set_ylabel("tang vel $v_t/v_{200m}$")
    # fig2.colorbar(hist6[3], ax = plot6)
    
    # fig2.suptitle("Infall Particles")


    # print(np.array_equal(hist1[1], hist4[1]))
    # print(np.array_equal(hist1[2], hist4[2]))
    #plt.show()

def graph_feature_importance(feature_names, feature_importance):
    mpl.rcParams.update({'font.size': 16})
    fig2, (plot1) = plt.subplots(1,1)
    import_idxs = np.argsort(feature_importance)
    print(import_idxs)
    print(feature_names)
    plot1.barh(feature_names[import_idxs], feature_importance[import_idxs])
    plot1.set_xlabel = ("XGBoost feature importance")
    plt.show()

def graph_correlation_matrix(data, feature_names):
    mpl.rcParams.update({'font.size': 12})
    heatmap = sns.heatmap(data.corr(), annot = True, cbar = True)
    heatmap.set_title("Feature Correlation Heatmap")
    plt.show()