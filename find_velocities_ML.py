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

G = constants.G # kpc km2/M⊙/s2

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

#get halo info
halos_vel = np.load(save_location + "halo_velocity.npy")
halos_vel = halos_vel[:, snapshot, :]
halos_v200 = np.load(save_location + "halos_v200.npy")
halos_r200 = np.load(save_location + "halo_R200m.npy")
halos_r200 = halos_r200[:, snapshot]
halos_last_snap = np.load(save_location + "halo_last_snap.npy")
halos_status = np.load(save_location + "halo_status.npy")
halos_status = halos_status[:,snapshot]

indices_keep = np.zeros((halos_last_snap.size),dtype = np.int32)
indices_keep = np.where((halos_last_snap >= snapshot) & (halos_status == 10))

halos_vel = halos_vel[indices_keep]
halos_r200 = halos_r200[indices_keep]

t1 = time.time()

particle_distances = particle_halo_radius_comp[:,0]

# how many particles are we workin with
num_particles_identified = particle_halo_assign_id.shape[0]

# indices where halo starts
# TODO check that this is counting the right amount of particles per halo
# Could just do a mask where halo id matches and do it in a loop
indices_change = np.where(particle_halo_assign_id[:-1,1] != particle_halo_assign_id[1:,1])[0] + 1
indices_change = np.hstack((0, indices_change)) #add zero to start
indices_change = np.append(indices_change, particle_halo_assign_id.shape[0]) #add last index
num_halos = indices_change.size - 1

# take indices from search and use to get velocities
use_particle_vel = np.zeros((num_particles_identified,3), dtype = np.float32)
particle_indices = particle_halo_assign_id[:,2].astype(int)
use_particle_vel = particles_vel[particle_indices,:] 

root_a = np.sqrt(scale_factor)
use_particle_vel = use_particle_vel * root_a


# calculate peculiar velocity by subtracting halo velocity from particle velocity
def calc_vel_pec(part_vel, halo_vel, part_dist, part_in_halo_indices, halo_indices):
    particles_vel_pec = np.zeros((part_vel.shape[0],3), dtype = np.float32)
    radius_div_r200 = np.zeros((part_dist.size), dtype = np.float32)
    num_halos = halo_indices.shape[0]
    
    start_vel_pec = 0
    for i in range(num_halos):
        finish_vel_pec = start_vel_pec + (part_in_halo_indices[i, 1] - part_in_halo_indices[i,0])
        #TODO change so accessing halo v200, r200 same way
        particles_vel_pec[start_vel_pec:finish_vel_pec,:] =  part_vel[start_vel_pec:finish_vel_pec,:] - halo_vel[i,:] 

        radius_div_r200[start_vel_pec:finish_vel_pec] = part_dist[start_vel_pec:finish_vel_pec] / halos_r200[halo_indices[i]]
        
        start_vel_pec = finish_vel_pec
    # print(particles_vel_pec)
    # print(particles_vel_pec.shape)
    return particles_vel_pec, radius_div_r200

def calc_rhat(x_dist, y_dist, z_dist):
    rhat = np.zeros((x_dist.size,3), dtype = np.float32)
    
    # get unit vector by dividing components by magnitude
    magnitude = np.sqrt(np.add(np.add(np.square(x_dist),np.square(y_dist)),np.square(z_dist)), dtype = np.float32)
    rhat[:,0] = np.divide(x_dist,magnitude)
    rhat[:,1] = np.divide(y_dist,magnitude)
    rhat[:,2] = np.divide(z_dist,magnitude)

    return rhat

# calculate the v200 value given a halos mass (M200m) and radius (R200m)
def calc_v200(mass, radius):
    return np.sqrt((G * mass)/radius)


# Calculates the radial velocity
def calc_rad_vel(part_dist, part_vel_pec, comps, part_in_halo_indices, halo_indices):
    all_rhat = np.zeros((part_vel_pec.shape[0],3), dtype = np.float32)
    rad_vel_no_hub = np.zeros((part_vel_pec.shape[0]), dtype = np.float32)
    particles_vel_rad = np.zeros((part_vel_pec.shape[0]), dtype = np.float32)
    
    # find the unit direction vectors of all halos to particles
    all_rhat = calc_rhat(comps[:,1], comps[:,2],comps[:,3])
   
    num_halos = part_in_halo_indices.shape[0]

    # loop through each halo's particles
    start_vel_phys = 0
    for i in range(num_halos):
        finish_vel_phys = start_vel_phys + (part_in_halo_indices[i, 1] - part_in_halo_indices[i,0])
        
        # calculate the hubble flow velocity at the distances of each particle
        v_hubble = np.zeros((finish_vel_phys - start_vel_phys))
        v_hubble = np.multiply(hubble_constant, part_dist[start_vel_phys:finish_vel_phys])
        
        # calculate radial component of the peculiar velocitiy by dotting with rhat 
        rad_vel_no_hub[start_vel_phys:finish_vel_phys] = np.sum((np.multiply(part_vel_pec[start_vel_phys:finish_vel_phys,:], all_rhat[start_vel_phys:finish_vel_phys])), axis = 1)
        
        # calculate radial velocity by adding the hubble velocity to the radial component of peculiar vel
        # Then scale by the halo's r200 value
        particles_vel_rad[start_vel_phys:finish_vel_phys] = np.divide((rad_vel_no_hub[start_vel_phys:finish_vel_phys] + v_hubble), halos_v200[halo_indices[i]])


        if i == 5:
            for j in range(1, 16):
        
                #print("orig vel_pec", part_vel_pec[-j])
                print("Radius:", (part_dist[-j] / halos_r200[halo_indices[i]]), "radial hubble: ",v_hubble[-j], "radial vel pec:", rad_vel_no_hub[finish_vel_phys-j])
                print("rad vel:", particles_vel_rad[finish_vel_phys-j])

        start_vel_phys = finish_vel_phys
    return particles_vel_rad

correspond_halo_prop = np.load(save_location + "correspond_halo_prop.npy")

def make_bins(num_bins, radius, radius_r200, vel_rad, curr_halo_prop): 
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
        indices = np.where((radius_r200 >= bin_start) & (radius_r200 < bin_finish))

        if indices[0].size != 0:
            
            use_vel_rad = vel_rad[indices[0]]
            use_radius_r200 = radius_r200[indices[0]]
            average_val_part[i,0] = np.average(use_radius_r200)
            average_val_part[i,1] = np.average(use_vel_rad) 
            
            # get median radius and r200 value for this bin
            med_r200 = np.median(curr_halo_prop[indices])
            med_radius = np.median(radius[indices])
            average_val_hubble[i,0] = med_radius/med_r200
            
            # calculate the corresponding m200m and then use that to calculate v200
            corresponding_m200m = mass_so.R_to_M(med_r200, red_shift, "200c")
            #TODO plot for each with dashed lines
            #TODO save data as pickle then if it exists use it
            average_val_hubble[i,1] = (med_radius * hubble_constant)/calc_v200(corresponding_m200m,med_r200)

        bin_start = bin_finish
        
    return average_val_part, average_val_hubble

print("start binning")
all_halo_mass = np.load(save_location + "all_halo_mass.npy")
peak_heights = peaks.peakHeight(all_halo_mass, red_shift,)
mass_hist = np.histogram(peak_heights, 50)

def split_by_mass(start_nu, finish_nu, num_bins):
    # get all the indices where the halo masses are within the selected bounds
    halos_mass_indices = np.where((peak_heights >= start_nu) & (peak_heights < finish_nu))[0] 
    print("Num halos:", halos_mass_indices.shape[0])
    # Record the particle indices that the halo starts at until the next halo starts
    halos_use_indices_start = indices_change[halos_mass_indices]
    halos_use_indices_finish = indices_change[(halos_mass_indices + 1)]
    # TODO rename
    halos_use_indices = np.column_stack((halos_use_indices_start,halos_use_indices_finish))

    #create arrays for the all the information we need
    total_part_use = np.sum((halos_use_indices_finish - halos_use_indices_start))
    correspond_halo_r200 = np.zeros((total_part_use), dtype = np.float32)
    mass_particle_vel = np.zeros((total_part_use,3), dtype = np.float32)
    mass_particles_vel_pec = np.zeros((total_part_use,3), dtype = np.float32)
    mass_particle_distances = np.zeros((total_part_use), dtype = np.float32)
    mass_particle_halo_radius_comp = np.zeros((total_part_use,4), dtype = np.float32)
    
    # get the corresponding halo velocities
    mass_halos_vel = halos_vel[halos_mass_indices]
    
    start_mass_bin = 0
    finish_mass_bin = 0
    
    # Loop through all of the halos for this mass bin and assign corresponding values to smaller arrays
    for i in range(halos_use_indices.shape[0]):
        finish_mass_bin += (halos_use_indices[i][1] - halos_use_indices[i][0])
        mass_particle_vel[start_mass_bin:finish_mass_bin] = use_particle_vel[halos_use_indices[i][0]:halos_use_indices[i][1]]
        mass_particle_distances[start_mass_bin:finish_mass_bin] = particle_distances[halos_use_indices[i][0]:halos_use_indices[i][1]]
        mass_particle_halo_radius_comp[start_mass_bin:finish_mass_bin] = particle_halo_radius_comp[halos_use_indices[i][0]:halos_use_indices[i][1],:]
        correspond_halo_r200[start_mass_bin:finish_mass_bin] = correspond_halo_prop[halos_use_indices[i][0]:halos_use_indices[i][1],0]
        
        start_mass_bin = finish_mass_bin
        
    mass_particles_vel_pec, mass_radius_div_r200 = calc_vel_pec(mass_particle_vel, mass_halos_vel, mass_particle_distances, halos_use_indices, halos_mass_indices)
    mass_particles_vel_rad = calc_rad_vel(mass_particle_distances, mass_particles_vel_pec, mass_particle_halo_radius_comp, halos_use_indices, halos_mass_indices)


    start_mass_bin = 0
    finish_mass_bin = 0
    # for i in range(halos_use_indices.shape[0]):
    #     finish_mass_bin += (halos_use_indices[i][1] - halos_use_indices[i][0])
        
    #     # mass_bin_radius[start_mass_bin:finish_mass_bin] = mass_particle_distances[halos_use_indices[i][0]:halos_use_indices[i][1]]
    #     #mass_bin_radius_div_r200[start_mass_bin:finish_mass_bin] = mass_radius_div_r200[halos_use_indices[i][0]:halos_use_indices[i][1]]
    #     mass_bin_vel_rad[start_mass_bin:finish_mass_bin] = mass_particles_vel_rad[halos_use_indices[i][0]:halos_use_indices[i][1]]
    #     
    #     start_mass_bin = finish_mass_bin
    
    avg_vel_rad_part, avg_vel_rad_hub = make_bins(num_bins, mass_particle_distances, mass_radius_div_r200, mass_particles_vel_rad, correspond_halo_r200)
    avg_vel_rad_part = avg_vel_rad_part[~np.all(avg_vel_rad_part == 0, axis=1)]
    avg_vel_rad_hub = avg_vel_rad_hub[~np.all(avg_vel_rad_hub == 0, axis=1)]
    
    arr1inds = avg_vel_rad_hub[:,0].argsort()
    avg_vel_rad_hub[:,0] = avg_vel_rad_hub[arr1inds[:],0]
    avg_vel_rad_hub[:,1] = avg_vel_rad_hub[arr1inds[:],1]
    
    return avg_vel_rad_part, avg_vel_rad_hub

# start and finish track where we are in the array
#plt.rcParams['text.usetex'] = True
num_bins = 50
start_nu = 1
num_iter = 4
color = iter(cm.rainbow(np.linspace(0, 1, num_iter)))

for i in range(num_iter):
    c = next(color)
    finish_nu = start_nu + .5
    print(start_nu, "to", finish_nu)
    avg_vel_rad_part, avg_vel_rad_hub = split_by_mass(start_nu, finish_nu, num_bins)  
    plt.plot(avg_vel_rad_hub[:,0], avg_vel_rad_hub[:,1], color = "purple", linestyle = "dashed",
             label = r"Hubble Flow ${0} < \nu < {1}$".format(str(start_nu), str(finish_nu)))
    plt.plot(avg_vel_rad_part[:,0], avg_vel_rad_part[:,1], color = c, label = r"${0} < \nu < {1}$".format(str(start_nu), str(finish_nu)))
    start_nu += .5
    

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