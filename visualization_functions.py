import numpy as np
import matplotlib.pyplot as plt
from calculation_functions import calculate_distance
from data_and_loading_functions import check_pickle_exist_gadget

def compare_density_prf(radii, actual_prf_all, actual_prf_1halo, mass, orbit_assn, num, start_nu, end_nu):
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
    ax.plot(middle_bins, calculated_prf_orb, 'b-', label = "my code profile orb")
    ax.plot(middle_bins, calculated_prf_inf, 'g-', label = "my code profile inf")
    ax.plot(middle_bins, calculated_prf_all, 'r-', label = "my code profile all")
    ax.plot(middle_bins, actual_prf_all, 'c--', label = "SPARTA profile all part")
    ax.plot(middle_bins, actual_prf_1halo, 'm--', label = "SPARTA profile orbit")
    ax.plot(middle_bins, actual_prf_all - actual_prf_1halo, 'y--', label = "SPARTA profile inf")
    
    ax.set_title("1Halo Density Profile")
    ax.set_xlabel("radius $r/R_{200m}$")
    ax.set_ylabel("Mass of halo $M_{\odot}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("/home/zvladimi/MLOIS/Random_figures/density_prf_" + str(start_nu) + "-" + str(end_nu) + "_" + str(num) + ".png")
    #plt.show()
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