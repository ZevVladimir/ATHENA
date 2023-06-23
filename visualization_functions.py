import numpy as np
import matplotlib.pyplot as plt
from calculation_functions import calculate_distance

def compare_density_prf(bins, radii, actual_prf, num_prf_bins, mass):
    calculated_prf_all = np.zeros(num_prf_bins)
    start_bin = 0

    for i in range(num_prf_bins):
        end_bin = bins[i]  
        
        radii_within_range = np.where((radii >= start_bin) & (radii < end_bin))[0]
        if radii_within_range.size != 0 and i != 0:
            calculated_prf_all[i] = calculated_prf_all[i - 1] + radii_within_range.size * mass
        elif i == 0:
            calculated_prf_all[i] = radii_within_range.size * mass
        else:
            calculated_prf_all[i] = calculated_prf_all[i - 1]
            
        start_bin = end_bin
    # for j in range(calculated_prf_all.size):
    #     print("calc:", calculated_prf_all[j], "act:", actual_prf[j])
    bins = np.insert(bins,0,0)
    middle_bins = (bins[1:] + bins[:-1]) / 2

    plt.figure(1)
    plt.plot(middle_bins, calculated_prf_all, color = 'b', alpha = 0.5, label = "calculated profile")
    plt.plot(middle_bins, actual_prf, color = 'r', alpha = 0.5, label = "actual profile")
    plt.title("Density profile of halo")
    plt.xlabel("radius $r/R_{200m}$")
    plt.ylabel("Mass of halo $M_{\odot}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

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