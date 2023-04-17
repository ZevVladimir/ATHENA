import numpy as np
from pygadgetreader import *
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from colossus.halo import mass_so
from colossus.utils import constants
from matplotlib.pyplot import cm
import time

save_location =  "/home/zvladimi/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zvladimi/ML_orbit_infall_project/particle_data/snapshot_192/snapshot_0192"

cosmol = cosmology.setCosmology("bolshoi")

snapshot = 192 #set to what snapshot is being loaded in
red_shift = readheader(snapshot_path, 'redshift')
scale_factor = 1/(1+red_shift)
hubble_constant = cosmol.Hz(red_shift) * 0.001 # convert to units km/s/kpc

particle_halo_assign_id = np.load(save_location + "particle_halo_assign_id.npy")
particle_halo_radius_comp = np.load(save_location + "particle_halo_radius_comp.npy")
all_halo_mass = np.load(save_location + "all_halo_mass.npy")
particles_vel = np.load(save_location + "particle_vel_192.npy")
particles_per_halo = np.load(save_location + "particles_per_halo.npy")

halos_vel = np.load(save_location + "halo_velocity.npy")
halos_vel = halos_vel[:,snapshot,:]

t1 = time.time()

particle_distances = particle_halo_radius_comp[:,0]
particle_distances_div_r200 = particle_halo_radius_comp[:,1]
# how many particles are we workin with
num_particles_identified = particle_halo_assign_id.shape[0]
# where are the separations between halos
indices_change = np.where(particle_halo_assign_id[:-1,1] != particle_halo_assign_id[1:,1])[0] + 1
indices_change = np.append(indices_change, particle_halo_assign_id.shape[0]) 
mass_hist = np.histogram(all_halo_mass, 1000)
halos_mass_indices = np.where((all_halo_mass > mass_hist[1][3]) & (all_halo_mass < mass_hist[1][4]))[0]
halos_use_indices_finish = indices_change[(halos_mass_indices + 1)]
halos_use_indices_start = indices_change[(halos_mass_indices)]
halos_use_indices = np.column_stack((halos_use_indices_start,halos_use_indices_finish))



# take indices from search and use to get velocities
use_particle_vel = np.zeros((num_particles_identified,3))
particle_indices = particle_halo_assign_id[:,2].astype(int)
use_particle_vel = particles_vel[particle_indices,:] 
root_a = np.sqrt(scale_factor)
use_particle_vel = use_particle_vel * root_a

particles_vel_pec = np.zeros((num_particles_identified,3))


# calculate peculiar velocity by subtracting halo velocity from particle velocity
# TODO rename start/finish to reflect radius too
radius_div_r200 = np.zeros((particle_distances.size))
start_vel_pec = 0
for i in range(indices_change.size):
    finish_vel_pec = indices_change[i]
    particles_vel_pec[start_vel_pec:finish_vel_pec,:] = halos_vel[i,:] - use_particle_vel[start_vel_pec:finish_vel_pec,:]  
    #radius_div_r200[start_vel_pec:finish_vel_pec] = particle_distances[start_vel_pec:finish_vel_pec]/halos_r200[i]
    start_vel_pec = finish_vel_pec

def calc_rhat(x_dist, y_dist, z_dist):
    rhat = np.zeros((x_dist.size,3))
    # get unit vector by dividing components by magnitude
    magnitude = np.sqrt(np.square(x_dist) + np.square(y_dist) + np.square(z_dist))
    rhat[:,0] = x_dist/magnitude
    rhat[:,1] = y_dist/magnitude
    rhat[:,2] = z_dist/magnitude

    return rhat

particles_per_halo = particles_per_halo.astype(int)
all_rhat = np.zeros((particles_vel_pec.shape[0],3), dtype = np.float32)
particles_vel_phys = np.zeros((particles_vel_pec.shape[0],3), dtype = np.float32)
particles_vel_tan = np.zeros((particles_vel_pec.shape[0],3), dtype = np.float32)
particles_vel_rad = np.zeros((particles_vel_pec.shape[0]), dtype = np.float32)



all_rhat = calc_rhat(particle_halo_radius_comp[:,2], particle_halo_radius_comp[:,3],particle_halo_radius_comp[:,4])

# convert peculiar velocity to physical velocity by adding scale factor * h * dist from particle to halo
start_vel_phys = 0
for i in range(indices_change.size):
    finish_vel_phys = indices_change[i]
    v_hubble = np.zeros(finish_vel_phys - start_vel_phys)
    v_hubble = np.reshape((hubble_constant * particle_distances[start_vel_phys:finish_vel_phys]),((finish_vel_phys - start_vel_phys),1))
    
    particles_vel_phys[start_vel_phys:finish_vel_phys,:] =  (v_hubble * all_rhat[start_vel_phys:finish_vel_phys])

    particles_vel_rad[start_vel_phys:finish_vel_phys] = np.sum((particles_vel_phys[start_vel_phys:finish_vel_phys] * all_rhat[start_vel_phys:finish_vel_phys]), axis = 1) 
    
    start_vel_phys = finish_vel_phys
    
#MAKE SURE TO BIN LOGARITHMICALLY
def make_bins(num_bins, radius, vel_rad):
    # remove the blank parts of the radius
    radius = radius[radius != 0]
    radiusinds = radius.argsort()
    sorted_radius = radius[radiusinds[::]]
    sorted_vel_rad = vel_rad[radiusinds[::]]
    print(np.min(sorted_vel_rad))
    print(np.max(sorted_vel_rad))

    hist = np.histogram(sorted_radius, num_bins)
    bins = hist[1]
    bin_start = 0
    average_val = np.zeros((num_bins,2))
    
    for i in range(num_bins - 1):
        bin_size = bins[i + 1] - bins[i]
        bin_finish = bin_start + bin_size

        # make sure there are points within the bins
        indices = np.where((sorted_radius >= bin_start) & (sorted_radius <= bin_finish))
        if indices[0].size != 0:

            use_vel_rad = sorted_vel_rad[indices[0]]
            average_val[i,0] = np.mean(np.array([bin_start,bin_finish]))
            average_val[i,1] = np.mean(use_vel_rad) 

        bin_start = bin_finish
    return average_val

print("start binning")
num_bins = 1000
mass_bin_radius = np.zeros(particle_distances.size)
mass_bin_vel_rad = np.zeros(particles_vel_rad.size)


start_mass_bin = 0
finish_mass_bin = 0
for i in range(halos_use_indices.shape[0]):
    finish_mass_bin += (halos_use_indices[i][1] - halos_use_indices[i][0])
    mass_bin_radius[start_mass_bin:finish_mass_bin] = particle_distances[halos_use_indices[i][0]:halos_use_indices[i][1]]
    mass_bin_vel_rad[start_mass_bin:finish_mass_bin] = particles_vel_rad[halos_use_indices[i][0]:halos_use_indices[i][1]]
    start_mass_bin = finish_mass_bin

mass_bin_radius = mass_bin_radius[mass_bin_radius != 0]
avg_vel_rad = make_bins(num_bins, mass_bin_radius, mass_bin_vel_rad)
#avg_vel_rad = make_bins(num_bins, particle_distances, particles_vel_rad)
avg_vel_rad = avg_vel_rad[~np.all(avg_vel_rad == 0, axis=1)]

graph1, (plot1) = plt.subplots(1,1)
plot1.plot(avg_vel_rad[:,0], hubble_constant * avg_vel_rad[:,0], color = "blue", label = "hubble flow")
plot1.plot(avg_vel_rad[:,0], avg_vel_rad[:,1], color = "purple", label = "particles")
plot1.set_title("average radial velocity vs position all particles")
plot1.set_xlabel("position $kpc$")
plot1.set_ylabel("average rad vel $km/s$")
plot1.set_xscale("log")    

plot1.legend()
t2 = time.time()
print("velocity finished: ", (t2 - t1)," seconds")

plt.show()