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
#get and set constants
snapshot = 192 #set to what snapshot is being loaded in
red_shift = readheader(snapshot_path, 'redshift')
scale_factor = 1/(1+red_shift)
hubble_constant = cosmol.Hz(red_shift) * 0.001 # convert to units km/s/kpc

# get particle info
particle_halo_assign_id = np.load(save_location + "particle_halo_assign_id.npy")
particle_halo_radius_comp = np.load(save_location + "particle_halo_radius_comp.npy")
particles_vel = np.load(save_location + "particle_vel_192.npy")
particles_per_halo = np.load(save_location + "particles_per_halo.npy")
correspond_halo_prop = np.load(save_location + "correspond_halo_prop.npy")

#get halo info
halos_vel = np.load(save_location + "halo_velocity.npy")
halos_vel = halos_vel[:,snapshot,:]
all_halo_mass = np.load(save_location + "all_halo_mass.npy")
halos_v200 = np.load(save_location + "halos_v200.npy")
halos_r200 = np.load(save_location + "halo_R200m.npy")
halos_r200 = halos_r200[:,snapshot]
halos_last_snap = np.load(save_location + "halo_last_snap.npy")
halos_id = np.load(save_location + "halo_id.npy")
halos_id = halos_id[:,snapshot]
halos_status = np.load(save_location + "halo_status.npy")
halos_status = halos_status[:,snapshot]

indices_keep = np.zeros((halos_id.size))
indices_keep = np.where((halos_last_snap >= snapshot) & (halos_status == 10))

halos_vel = halos_vel[indices_keep]
halos_r200 = halos_r200[indices_keep]
halos_id = halos_id[indices_keep]

t1 = time.time()

particle_distances = particle_halo_radius_comp[:,0]
# how many particles are we workin with
num_particles_identified = particle_halo_assign_id.shape[0]
# where are the separations between halos
indices_change = np.where(particle_halo_assign_id[:-1,1] != particle_halo_assign_id[1:,1])[0] + 1
indices_change = np.append(indices_change, particle_halo_assign_id.shape[0]) 

# choose a halo mass bin
mass_hist = np.histogram(all_halo_mass, 1000)
use_mass_start = 30
use_mass_finish = 31
print("mass range:", mass_hist[1][use_mass_start], "to", mass_hist[1][use_mass_finish], "solar masses")

halos_mass_indices = np.where((all_halo_mass > mass_hist[1][use_mass_start]) & (all_halo_mass < mass_hist[1][use_mass_finish]))[0]
#print(np.where(mass_hist[0] != 0))
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

radius_div_r200 = np.zeros((particle_distances.size))
start_vel_pec = 0
for i in range(indices_change.size):
    finish_vel_pec = indices_change[i]
    particles_vel_pec[start_vel_pec:finish_vel_pec,:] = use_particle_vel[start_vel_pec:finish_vel_pec,:] - halos_vel[i,:]
    radius_div_r200[start_vel_pec:finish_vel_pec] = particle_distances[start_vel_pec:finish_vel_pec] / halos_r200[i]
    start_vel_pec = finish_vel_pec

def calc_rhat(x_dist, y_dist, z_dist):
    rhat = np.zeros((x_dist.size,3))
    # get unit vector by dividing components by magnitude
    magnitude = np.sqrt(np.square(x_dist) + np.square(y_dist) + np.square(z_dist))
    rhat[:,0] = x_dist/magnitude
    rhat[:,1] = y_dist/magnitude
    rhat[:,2] = z_dist/magnitude

    return rhat

# set arrays
particles_per_halo = particles_per_halo.astype(int)
all_rhat = np.zeros((particles_vel_pec.shape[0],3), dtype = np.float32)
particles_vel_phys = np.zeros((particles_vel_pec.shape[0],3), dtype = np.float32)
particles_vel_tan = np.zeros((particles_vel_pec.shape[0],3), dtype = np.float32)
particles_vel_rad = np.zeros((particles_vel_pec.shape[0]), dtype = np.float32)
second_particles_vel_rad = np.zeros((particles_vel_pec.shape[0]), dtype = np.float32)

# find the unit direction vectors of all halos to particles
all_rhat = calc_rhat(particle_halo_radius_comp[:,1], particle_halo_radius_comp[:,2],particle_halo_radius_comp[:,3])

# convert peculiar velocity to physical velocity by adding scale factor * h * dist from particle to halo
start_vel_phys = 0
# loop through each halo's particles
for i in range(indices_change.size):
    finish_vel_phys = indices_change[i]
    # calculate the hubble flow velocity at the distances of each particle
    v_hubble = np.zeros((finish_vel_phys - start_vel_phys))
    v_hubble = np.reshape((hubble_constant * radius_div_r200[start_vel_phys:finish_vel_phys]),((finish_vel_phys - start_vel_phys),1))
    # print(i)
    # print(v_hubble * all_rhat[start_vel_phys:finish_vel_phys,:])
    # print(particles_vel_pec[start_vel_phys:finish_vel_phys,:])
    # calculate physical velocity by adding the hubble velocity in the radial direction

    particles_vel_phys[start_vel_phys:finish_vel_phys,:] = (particles_vel_pec[start_vel_phys:finish_vel_phys,:] + (v_hubble * all_rhat[start_vel_phys:finish_vel_phys]))/halos_v200[i]
    
    # calculate radial velocity by just taking the radial compenent of the physical velocity
    particles_vel_rad[start_vel_phys:finish_vel_phys] = np.sum((particles_vel_phys[start_vel_phys:finish_vel_phys] * all_rhat[start_vel_phys:finish_vel_phys]), axis = 1) 
    
    # calculating radial velocity with the radial part of the peculiar velocity with the hubble flow added
    second_particles_vel_rad[start_vel_phys:finish_vel_phys] = (np.sum((particles_vel_pec[start_vel_phys:finish_vel_phys] * all_rhat[start_vel_phys:finish_vel_phys]), axis = 1) + np.reshape(v_hubble,(v_hubble.size)))
    
    
    # if (np.allclose(particles_vel_rad, second_particles_vel_rad, rtol = 1e-4, atol = 1e-4)) is False:
    #     print(i)
    #     print(np.round(particles_vel_rad[finish_vel_phys-5:finish_vel_phys],4))
    #     print(np.round(second_particles_vel_rad[finish_vel_phys-5:finish_vel_phys],4))

    # print(np.sum((particles_vel_pec[start_vel_phys:finish_vel_phys] * all_rhat[start_vel_phys:finish_vel_phys]), axis = 1)[v_hubble.size - 5: v_hubble.size])
    # print(np.reshape(v_hubble,(v_hubble.size))[v_hubble.size - 5: v_hubble.size])
    
    start_vel_phys = finish_vel_phys
    
#MAKE SURE TO BIN LOGARITHMICALLY
def make_bins(num_bins, radius, radius_r200, vel_rad):
    # remove the blank parts of the radius

    radius = radius[radius != 0]
    radius_r200 = radius_r200[radius_r200 != 0]
    
    min_rad = np.min(radius_r200)
    max_rad = np.max(radius_r200)
    print(min_rad)
    print(max_rad)
    bins = np.logspace(np.log10(min_rad), np.log10(max_rad), num_bins)

    bin_start = 0
    average_val_part = np.zeros((num_bins,2))
    average_val_hubble = np.zeros((num_bins,2))
    
    for i in range(num_bins - 1):

        bin_size = bins[i + 1] - bins[i]
        bin_finish = bin_start + bin_size
        # make sure there are points within the bins
        indices = np.where((radius_r200 >= bin_start) & (radius_r200 <= bin_finish))

        if indices[0].size != 0:

            use_vel_rad = vel_rad[indices[0]]
            average_val_part[i,0] = np.mean(np.array([bin_start,bin_finish]))
            average_val_part[i,1] = np.mean(use_vel_rad) 

            average_r200 = np.mean(correspond_halo_prop[indices,0])
            average_radius = np.mean(radius[indices])
            print("r200", average_r200)
            print("radius", average_radius)
            print("r/r200", average_radius/average_r200)
            print("hub vel", average_radius * hubble_constant)
            print("v/v200", average_radius * hubble_constant/np.mean(np.unique(correspond_halo_prop[indices,1])[0]) )
            average_val_hubble[i,0] = average_radius/average_r200
            average_val_hubble[i,1] = average_radius * hubble_constant/np.mean(np.unique(correspond_halo_prop[indices,1])[0]) 

        bin_start = bin_finish
    return average_val_part, average_val_hubble

print("start binning")
num_bins = 100
mass_bin_radius = np.zeros(particle_distances.size)
mass_bin_radius_div_r200 = np.zeros(particle_distances.size)
mass_bin_vel_rad = np.zeros(particles_vel_rad.size)

start_mass_bin = 0
finish_mass_bin = 0
for i in range(halos_use_indices.shape[0]):
    finish_mass_bin += (halos_use_indices[i][1] - halos_use_indices[i][0])
    mass_bin_radius[start_mass_bin:finish_mass_bin] = particle_distances[halos_use_indices[i][0]:halos_use_indices[i][1]]
    mass_bin_radius_div_r200[start_mass_bin:finish_mass_bin] = radius_div_r200[halos_use_indices[i][0]:halos_use_indices[i][1]]
    mass_bin_vel_rad[start_mass_bin:finish_mass_bin] = particles_vel_rad[halos_use_indices[i][0]:halos_use_indices[i][1]]
    start_mass_bin = finish_mass_bin
    

# mass_bin_radius = mass_bin_radius[mass_bin_radius != 0]
avg_vel_rad_part, avg_vel_rad_hub = make_bins(num_bins, mass_bin_radius, mass_bin_radius_div_r200, mass_bin_vel_rad)
#avg_vel_rad_part, avg_vel_rad_hub = make_bins(num_bins, radius_div_r200, particles_vel_rad)

avg_vel_rad_part = avg_vel_rad_part[~np.all(avg_vel_rad_part == 0, axis=1)]
avg_vel_rad_hub = avg_vel_rad_hub[~np.all(avg_vel_rad_hub == 0, axis=1)]

graph1, (plot1) = plt.subplots(1,1)
plot1.plot(avg_vel_rad_hub[:,0], avg_vel_rad_hub[:,1], color = "purple", label = "hubble flow")
plot1.plot(avg_vel_rad_part[:,0], avg_vel_rad_part[:,1], color = "blue", label = "particles")
plot1.set_title("average radial velocity vs position all particles")
plot1.set_xlabel("position $r/R_{200m}$")
plot1.set_ylabel("average rad vel $v_r/v_{200m}$")
plot1.set_xscale("log")    

plot1.legend()
t2 = time.time()
print("velocity finished: ", (t2 - t1)," seconds")

plt.show()