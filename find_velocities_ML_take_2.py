import numpy as np
from pygadgetreader import readsnap, readheader
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from colossus.halo import mass_so
from colossus.utils import constants
from colossus.lss import peaks
import time
from matplotlib.pyplot import cm
import sys


t1 = time.time()

G = constants.G # kpc km2/M⊙/s2

save_location =  "/home/zvladimi/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zvladimi/ML_orbit_infall_project/particle_data/snapshot_192/snapshot_0192"

cosmol = cosmology.setCosmology("bolshoi")
#get and set constants
snapshot = 192 #set to what snapshot is being loaded in
red_shift = readheader(snapshot_path, 'redshift')
scale_factor = 1/(1+red_shift)
little_h = cosmol.h
hubble_constant = cosmol.Hz(red_shift) * 0.001 # convert to units km/s/kpc


particle_halo_assign_id = np.load(save_location + "particle_halo_assign_id.npy")
indices_change = np.where(particle_halo_assign_id[:-1,1] != particle_halo_assign_id[1:,1])[0] + 1
indices_change = np.hstack((0, indices_change)) #add zero to start
indices_change = np.append(indices_change, particle_halo_assign_id.shape[0]) #add last index

particle_halo_radius_comp = np.load(save_location + "particle_halo_radius_comp.npy")
particle_radii = particle_halo_radius_comp[:,0]
particle_dist_components = particle_halo_radius_comp[:,1:]

def calc_scaled_radii(particle_dist, halo_r200, start_indices, finish_indices):
    total_num_particles = np.sum((finish_indices - start_indices))
    scaled_radii = np.zeros(total_num_particles, dtype = np.float32)

    start = 0
    finish = 0
    
    for i in range(start_indices.size):
        finish += (finish_indices[i] - start_indices[i])
        scaled_radii[start:finish] = particle_dist[start_indices[i]:finish_indices[i]] / halo_r200[i]

        start = finish
        
    return scaled_radii


def calc_pec_vel(particle_vel, halo_vel, start_indices, finish_indices):
    # Will be as many peculiar vel as there are particles
    total_num_particles = np.sum((finish_indices - start_indices))
    peculiar_velocities = np.zeros((total_num_particles, 3), dtype = np.float32)
    
    start = 0
    finish = 0
    # Loop through each halo's indices
    for i in range(start_indices.size):
        # Track how many new particles to add to open part of array
        finish += (finish_indices[i] - start_indices[i])
        
        # Peculiar velocity is particle velocity minus the corresponding halo velocity
        peculiar_velocities[start:finish] = particle_vel[start_indices[i]:finish_indices[i]] - halo_vel[i]   

        start = finish
        
    return peculiar_velocities

def calc_rhat(x_comp, y_comp, z_comp):
    rhat = np.zeros((x_comp.size, 3), dtype = np.float32)
    # Get the magnitude for each particle
    magnitude = np.sqrt(np.square(x_comp) + np.square(y_comp) + np.square(z_comp))
    
    # Scale the components by the magnitude to get a unit vector
    rhat[:,0] = x_comp/magnitude
    rhat[:,1] = y_comp/magnitude
    rhat[:,2] = z_comp/magnitude
    
    return rhat

def calc_v200m(mass, radius):
    # calculate the v200m for a halo based on its mass and radius
    return np.sqrt((G * mass)/radius)

def calc_rad_vel(peculiar_vel, particle_dist, coord_sep, start_indices, finish_indices, halo_r200):
    total_num_particles = np.sum((finish_indices - start_indices))

    use_coord_sep = np.zeros((total_num_particles,3), dtype = np.float32)
    use_part_dist = np.zeros(total_num_particles, dtype = np.float32)
    corresponding_v200m = np.zeros((total_num_particles), dtype = np.float32)
    
    start = 0
    finish = 0
    # For how many halos there are loop through how many particles are within each
    for i in range(start_indices.size):
        finish += (finish_indices[i] - start_indices[i])
        
        # Get the corresponding components, distances, and halo v200m for every particle
        use_coord_sep[start:finish] = coord_sep[start_indices[i]:finish_indices[i]]
        use_part_dist[start:finish] = particle_dist[start_indices[i]:finish_indices[i]]
        corresponding_hubble_m200m = mass_so.R_to_M(halo_r200[i], red_shift, "200c") * little_h # convert to M⊙
        curr_v200m = calc_v200m(corresponding_hubble_m200m, halo_r200[i])
        corresponding_v200m[start:finish] = curr_v200m
        
        start = finish
        
    # calculate the unit vector of the halo to the particle  
    rhat = calc_rhat(use_coord_sep[:,0], use_coord_sep[:,1], use_coord_sep[:,2])
    
    # Hubble velocity is the hubble constant times the distance the particle is from the halo
    
    v_hubble = hubble_constant * use_part_dist   
    
    # Dot the velocity with rhat to get the radial component
    radial_component_vel = np.sum(np.multiply(peculiar_vel, rhat), axis = 1)
    
    # Add radial component and v_hubble since both are now in radial direction
    radial_vel = radial_component_vel + v_hubble

    # scale all the radial velocities by v200m of the halo
    
    return radial_vel/corresponding_v200m

def split_into_bins(num_bins, radial_vel, scaled_radii, particle_radii, halo_r200_per_part):
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
            corresponding_hubble_m200m = mass_so.R_to_M(median_hubble_r200, red_shift, "200c") * little_h # convert to M⊙
            average_val_hubble[i,1] = (median_hubble_radius * hubble_constant)/calc_v200m(corresponding_hubble_m200m, median_hubble_r200)
            
        bin_start = bin_end
    
    return average_val_part, average_val_hubble
    
def split_halo_by_mass(num_bins, start_nu, num_iter, particle_velocities, halo_masses, halo_r200, halo_velocities, halo_start_indices, halo_prop_per_part):
    color = iter(cm.rainbow(np.linspace(0, 1, num_iter)))
    # convert masses to peaks
    scaled_halo_mass = halo_masses/little_h # units M⊙/h
    peak_heights = peaks.peakHeight(scaled_halo_mass, red_shift)
    
    # For how many graphs
    for i in range(num_iter):
        c = next(color)
        end_nu = start_nu + 0.5
        print("Start split:", start_nu, "to", end_nu)
        # Get the indices of the halos that are within the desired peaks
        halos_within_range = np.where((peak_heights >= start_nu) & (peak_heights < end_nu))[0]
        
        # Get the halo r200m and velocities that are within the range
        halo_r200_within_range = halo_r200[halos_within_range]
        halo_vel_within_range = halo_velocities[halos_within_range]
        
        # Get the range of particle indices for each halo
        particle_start_indices_within_range = halo_start_indices[halos_within_range]
        particle_finish_indices_within_range = halo_start_indices[(halos_within_range + 1)]
        
        # calculate peculiar and radial velocity as well as scaled radii
        peculiar_velocity = calc_pec_vel(particle_velocities, halo_vel_within_range, particle_start_indices_within_range, particle_finish_indices_within_range)
        radial_velocity = calc_rad_vel(peculiar_velocity, particle_radii, particle_dist_components, particle_start_indices_within_range, 
                                       particle_finish_indices_within_range, halo_r200_within_range)
        radius_div_r200 = calc_scaled_radii(particle_radii, halo_r200_within_range, particle_start_indices_within_range, particle_finish_indices_within_range)
        
        graph_rad_vel, graph_val_hubble = split_into_bins(num_bins, radial_velocity, radius_div_r200, particle_radii, halo_prop_per_part[:,0])
        graph_rad_vel = graph_rad_vel[~np.all(graph_rad_vel == 0, axis=1)]
        graph_val_hubble = graph_val_hubble[~np.all(graph_val_hubble == 0, axis=1)]

        plt.plot(graph_rad_vel[:,0], graph_rad_vel[:,1], color = c, alpha = 0.7, label = r"${0} < \nu < {1}$".format(str(start_nu), str(end_nu)))
        start_nu = end_nu
        
    return graph_val_hubble
        
particles_vel = np.load(save_location + "all_part_vel.npy")        
root_a = np.sqrt(scale_factor)
use_part_vel = particles_vel * root_a        

halos_last_snap = np.load(save_location + "halo_last_snap.npy")
halos_status = np.load(save_location + "halo_status.npy")
halos_status = halos_status[:,snapshot]

indices_keep = np.zeros((halos_last_snap.size),dtype = np.int32)
indices_keep = np.where((halos_last_snap >= snapshot) & (halos_status == 10))
all_halo_masses = np.load(save_location + "all_halo_mass.npy") # don't have to do indices_keep because only found for halos used

all_halo_vel = np.load(save_location + "halo_velocity.npy")
all_halo_vel = all_halo_vel[:, snapshot, :]
all_halo_vel = all_halo_vel[indices_keep]

all_halo_r200 = np.load(save_location + "halo_R200m.npy")
all_halo_r200 = all_halo_r200[:, snapshot]
all_halo_r200 = all_halo_r200[indices_keep] * little_h # convert to kpc

correspond_halo_prop = np.load(save_location + "correspond_halo_prop.npy")     

num_bins = 50        
start_nu = 1
num_iter = 5
hubble_vel = split_halo_by_mass(num_bins, start_nu, num_iter, use_part_vel, all_halo_masses, all_halo_r200, all_halo_vel, indices_change, correspond_halo_prop)

arr1inds = hubble_vel[:,0].argsort()
hubble_vel[:,0] = hubble_vel[arr1inds,0]
hubble_vel[:,1] = hubble_vel[arr1inds,1]
plt.plot(hubble_vel[:,0], hubble_vel[:,1], color = "purple", alpha = 0.5, linestyle = "dashed", label = r"Hubble Flow")
plt.title("average radial velocity vs position all particles")
plt.xlabel("position $r/R_{200m}$")
plt.ylabel("average rad vel $v_r/v_{200m}$")
plt.xscale("log")    
plt.ylim([-.5,1])
plt.xlim([0.01,15])
plt.legend()

t2 = time.time()
print("velocity finished: ", (t2 - t1)," seconds")

plt.show()
