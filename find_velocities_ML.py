import numpy as np
from pygadgetreader import readsnap, readheader
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from colossus.halo import mass_so
from colossus.utils import constants
from colossus.lss import peaks
import time
from matplotlib.pyplot import cm

G = constants.G

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

indices_keep = np.zeros((halos_id.size),dtype = np.int32)
indices_keep = np.where((halos_last_snap >= snapshot) & (halos_status == 10))

halos_vel = halos_vel[indices_keep]
halos_r200 = halos_r200[indices_keep]
halos_id = halos_id[indices_keep]

t1 = time.time()

particle_distances = particle_halo_radius_comp[:,0]
# how many particles are we workin with
num_particles_identified = particle_halo_assign_id.shape[0]

# indices where halo starts
indices_change = np.where(particle_halo_assign_id[:-1,1] != particle_halo_assign_id[1:,1])[0] + 1
indices_change = np.append(indices_change, particle_halo_assign_id.shape[0]) #add last index

# take indices from search and use to get velocities
use_particle_vel = np.zeros((num_particles_identified,3), dtype = np.float32)
particle_indices = particle_halo_assign_id[:,2].astype(int)
use_particle_vel = particles_vel[particle_indices,:] 
root_a = np.sqrt(scale_factor)
use_particle_vel = use_particle_vel * root_a

particles_vel_pec = np.zeros((num_particles_identified,3), dtype = np.float32)

# calculate peculiar velocity by subtracting halo velocity from particle velocity
radius_div_r200 = np.zeros((particle_distances.size), dtype = np.float32)
start_vel_pec = 0
for i in range(indices_change.size):
    finish_vel_pec = indices_change[i]
    
    particles_vel_pec[start_vel_pec:finish_vel_pec,:] = use_particle_vel[start_vel_pec:finish_vel_pec,:] - halos_vel[i,:]
    # if i == 15:
    #     for j in range(finish_vel_pec - start_vel_pec):
    #         if j % 200 == 0:
    #             print(j)
    #             print(use_particle_vel[start_vel_pec+j,:])
    #             print(halos_vel[i,:])
    #             print(particles_vel_pec[start_vel_pec + j,:])
    # get the corresponding r/r200 values for each particle

    radius_div_r200[start_vel_pec:finish_vel_pec] = particle_distances[start_vel_pec:finish_vel_pec] / halos_r200[i]
    start_vel_pec = finish_vel_pec

def calc_rhat(x_dist, y_dist, z_dist):
    rhat = np.zeros((x_dist.size,3), dtype = np.float32)
    # get unit vector by dividing components by magnitude
    magnitude = np.sqrt(np.square(x_dist) + np.square(y_dist) + np.square(z_dist))
    rhat[:,0] = x_dist/magnitude
    rhat[:,1] = y_dist/magnitude
    rhat[:,2] = z_dist/magnitude

    return rhat

# calculate the v200 value given a halos mass (M200m) and radius (R200m)
def calc_v200(mass, radius):
    return np.sqrt((G * mass)/radius)

# set arrays
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
#for i in range(1):

    finish_vel_phys = indices_change[i]
    # calculate the hubble flow velocity at the distances of each particle
    v_hubble = np.zeros((finish_vel_phys - start_vel_phys))
    v_hubble = np.reshape((hubble_constant * particle_distances[start_vel_phys:finish_vel_phys]),((finish_vel_phys - start_vel_phys),1))
    
    
    # calculate physical velocity by adding the hubble velocity in the radial direction

    particles_vel_phys[start_vel_phys:finish_vel_phys,:] = (particles_vel_pec[start_vel_phys:finish_vel_phys,:] + 
                                                            (v_hubble * all_rhat[start_vel_phys:finish_vel_phys]))/halos_v200[i]
    
    # calculate radial velocity by just taking the radial compenent of the physical 
    
    particles_vel_rad[start_vel_phys:finish_vel_phys] = np.sum((particles_vel_phys[start_vel_phys:finish_vel_phys,:] * all_rhat[start_vel_phys:finish_vel_phys]), axis = 1)

    
    # calculating radial velocity with the radial part of the peculiar velocity with the hubble flow added
    second_particles_vel_rad[start_vel_phys:finish_vel_phys] = ((np.sum((particles_vel_pec[start_vel_phys:finish_vel_phys,:] * 
                                                                         all_rhat[start_vel_phys:finish_vel_phys]), axis = 1) + 
                                                                 np.reshape(v_hubble,(v_hubble.size))))
    
    
    
    # for j in range(finish_vel_phys - start_vel_phys):
    #     if radius_div_r200[j] > 1:
    #         print(j)
    #         print(radius_div_r200[j])
    #         print(np.sum((particles_vel_pec[start_vel_phys+j,:] * all_rhat[start_vel_phys+j])))
    #         print(v_hubble[j])
    #         print(second_particles_vel_rad[start_vel_phys + j])
            
    
    # check that the two methods of calculation are equal
    # if (np.allclose(particles_vel_rad, second_particles_vel_rad, rtol = 1e-4, atol = 1e-4)) is False:
    #     print(i)
    #     print(np.round(particles_vel_rad[finish_vel_phys-5:finish_vel_phys],4))
    #     print(np.round(second_particles_vel_rad[finish_vel_phys-5:finish_vel_phys],4))
    
    start_vel_phys = finish_vel_phys
print(hubble_constant)

def make_bins(num_bins, radius, radius_r200, vel_rad): 
    count = 0

    min_rad = np.min(radius_r200)
    max_rad = np.max(radius_r200)
    bins = np.logspace(np.log10(min_rad), np.log10(max_rad), num_bins)

    bin_start = 0
    average_val_part = np.zeros((num_bins,2), dtype = np.float32)
    average_val_hubble = np.zeros((num_bins,2), dtype = np.float32)
    
    for i in range(num_bins - 1):
        bin_size = bins[i + 1] - bins[i]
        bin_finish = bin_start + bin_size
        # make sure there are points within the bins
        indices = np.where((radius_r200 >= bin_start) & (radius_r200 <= bin_finish))

        if indices[0].size != 0:
            
            use_vel_rad = vel_rad[indices[0]]
            average_val_part[i,0] = np.average(np.array([bin_start,bin_finish]))
            average_val_part[i,1] = np.average(use_vel_rad) 
            #print(average_val_part[i])
            
            # get median radius and r200 value for this bin
            med_r200 = np.median(correspond_halo_prop[indices,0])
            med_radius = np.median(radius[indices])
            average_val_hubble[i,0] = med_radius/med_r200
            
            # calculate the corresponding m200m and then use that to calculate v200
            corresponding_m200m = mass_so.R_to_M(med_r200, red_shift, "200c")

            average_val_hubble[i,1] = (med_radius * hubble_constant)/calc_v200(corresponding_m200m,med_r200)

        bin_start = bin_finish
        
    return average_val_part, average_val_hubble

print("start binning")
peak_heights = peaks.peakHeight(all_halo_mass, red_shift,)
mass_hist = np.histogram(peak_heights, 50)

def split_by_mass(start_nu, finish_nu, num_bins):
    
    # get all the indices where the halo masses are within the selected bounds
    halos_mass_indices = np.where((peak_heights > start_nu) & (peak_heights < finish_nu))[0]
    print(halos_mass_indices.shape)
    # Record the particle indices that the halo starts at until the next halo starts
    halos_use_indices_start = indices_change[(halos_mass_indices)]
    halos_use_indices_finish = indices_change[(halos_mass_indices + 1)]
    halos_use_indices = np.column_stack((halos_use_indices_start,halos_use_indices_finish))

    total_part_use = np.sum((halos_use_indices_finish - halos_use_indices_start))
    mass_bin_radius = np.zeros(total_part_use, dtype = np.float32)
    mass_bin_radius_div_r200 = np.zeros(total_part_use, dtype = np.float32)
    mass_bin_vel_rad = np.zeros(total_part_use, dtype = np.float32)
    
    start_mass_bin = 0
    finish_mass_bin = 0
    for i in range(halos_use_indices.shape[0]):
        finish_mass_bin += (halos_use_indices[i][1] - halos_use_indices[i][0])
        
        mass_bin_radius[start_mass_bin:finish_mass_bin] = particle_distances[halos_use_indices[i][0]:halos_use_indices[i][1]]
        mass_bin_radius_div_r200[start_mass_bin:finish_mass_bin] = radius_div_r200[halos_use_indices[i][0]:halos_use_indices[i][1]]
        mass_bin_vel_rad[start_mass_bin:finish_mass_bin] = particles_vel_rad[halos_use_indices[i][0]:halos_use_indices[i][1]]
        
        start_mass_bin = finish_mass_bin
    
    avg_vel_rad_part, avg_vel_rad_hub = make_bins(num_bins, mass_bin_radius, mass_bin_radius_div_r200, mass_bin_vel_rad)
    avg_vel_rad_part = avg_vel_rad_part[~np.all(avg_vel_rad_part == 0, axis=1)]
    avg_vel_rad_hub = avg_vel_rad_hub[~np.all(avg_vel_rad_hub == 0, axis=1)]
    
    arr1inds = avg_vel_rad_hub[:,0].argsort()
    avg_vel_rad_hub[:,0] = avg_vel_rad_hub[arr1inds[:],0]
    avg_vel_rad_hub[:,1] = avg_vel_rad_hub[arr1inds[:],1]
    
    return avg_vel_rad_part, avg_vel_rad_hub

# start and finish track where we are in the array
plt.rcParams['text.usetex'] = True
num_bins = 50
start_nu = 1
num_iter = 5
color = iter(cm.rainbow(np.linspace(0, 1, num_iter)))
for i in range(num_iter):
    c = next(color)
    finish_nu = start_nu + .5
    avg_vel_rad_part, avg_vel_rad_hub = split_by_mass(start_nu, finish_nu, num_bins)
    
    plt.plot(avg_vel_rad_part[:,0], avg_vel_rad_part[:,1], color = c, label = r"${0} < \nu < {1}$".format(str(start_nu), str(finish_nu)))
    start_nu += .5
    

plt.plot(avg_vel_rad_hub[:,0], avg_vel_rad_hub[:,1], color = "purple", label = "hubble flow")
plt.title("average radial velocity vs position all particles")
plt.xlabel("position $r/R_{200m}$")
plt.ylabel("average rad vel $v_r/v_{200m}$")
plt.xscale("log")    
# plt.ylim([-.5,.5])
plt.xlim([0.01,10])
plt.legend()
t2 = time.time()
print("velocity finished: ", (t2 - t1)," seconds")

plt.show()