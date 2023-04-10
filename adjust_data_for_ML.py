import numpy as np
from pygadgetreader import *
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from colossus.halo import mass_so

file = "/home/zvladimi/ML_orbit_infall_project/np_arrays/"
save_location =  "/home/zvladimi/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zvladimi/ML_orbit_infall_project/particle_data/snapshot_192/snapshot_0192"

# get constants from pygadgetreader
snapshot = 192 #set to what snapshot is being loaded in
red_shift = readheader(snapshot_path, 'redshift')
scale_factor = 1/(1+red_shift)
cosmol = cosmology.setCosmology("bolshoi")
rho_m = cosmol.rho_m(red_shift)
h = readheader(snapshot_path, 'h')

box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
box_size = box_size * 10**3 * scale_factor #convert to Kpc/h physical

#load particle info
particles_pid = np.load(file + "particle_pid_192.npy")
particles_vel = np.load(file + "particle_vel_192.npy")
particles_pos = np.load(file + "particle_pos_192.npy") 
particles_pos = particles_pos * 10**3 * scale_factor #convert to kpc and physical
particles_mass = np.load(file + "all_particle_mass_192.npy")
mass = particles_mass[0] * 10**10 #units M_sun/h

#load all halo info at snapshot
halos_pos = np.load(file + "halo_position.npy")
halos_pos = halos_pos[:,snapshot,:] * 10**3 * scale_factor #convert to kpc and physical
halos_vel = np.load(file + "halo_velocity.npy")
halos_vel = halos_vel[:,snapshot,:]
halos_last_snap = np.load(file + "halo_last_snap.npy")
halos_r200 = np.load(file + "halo_R200m.npy")
halos_r200 = halos_r200[:,snapshot]
halos_id = np.load(file + "halo_id.npy")
halos_id = halos_id[:,snapshot]
halos_status = np.load(file + "halo_status.npy")
halos_status = halos_status[:,snapshot]

num_particles = particles_pid.size

# remove all halos for any halo that doesn't exist beyond snapshot 
# remove all halos that aren't main halos (identified with tag = 10)
indices_keep = np.zeros((halos_id.size))
indices_keep = np.where((halos_last_snap >= snapshot) & (halos_status == 10))
halos_pos = halos_pos[indices_keep]
halos_vel = halos_vel[indices_keep]
halos_r200 = halos_r200[indices_keep]
halos_id = halos_id[indices_keep]
num_halos = halos_r200.size #num of halos remaining

#construct a search tree iwth all of the particle positions
particle_tree = cKDTree(data = particles_pos, leafsize = 3, balanced_tree = False, boxsize = box_size)

#calculate distance of particle from halo
def calculate_distance(halo_x, halo_y, halo_z, particle_x, particle_y, particle_z, new_particles):
    x_dist = particle_x - halo_x
    y_dist = particle_y - halo_y
    z_dist = particle_z - halo_z
    
    coord_diff = np.zeros((new_particles, 3))
    coord_diff[:,0] = x_dist
    coord_diff[:,1] = y_dist
    coord_diff[:,2] = z_dist

    half_box_size = box_size/2
    
    #handles periodic boundary conditions by checking if you were to add or subtract a boxsize would it then
    #be within half a box size of the halo
    #do this for x, y, and z coords
    x_within_plus = np.where((x_dist + box_size) < half_box_size)
    x_within_minus = np.where((x_dist - box_size) > -half_box_size)
    
    particle_x[x_within_plus] = particle_x[x_within_plus] + box_size
    particle_x[x_within_minus] = particle_x[x_within_minus] - box_size
    
    coord_diff[x_within_plus,0] = x_dist[x_within_plus] + box_size
    coord_diff[x_within_minus,0] = x_dist[x_within_minus] - box_size
    
    y_within_plus = np.where((y_dist + box_size) < half_box_size)
    y_within_minus = np.where((y_dist - box_size) > -half_box_size)
    
    particle_y[y_within_plus] = particle_y[y_within_plus] + box_size
    particle_y[y_within_minus] = particle_y[y_within_minus] - box_size
    
    coord_diff[y_within_plus,0] = y_dist[y_within_plus] + box_size
    coord_diff[y_within_minus,0] = y_dist[y_within_minus] - box_size
    
    z_within_plus = np.where((z_dist + box_size) < half_box_size)
    z_within_minus = np.where((z_dist - box_size) > -half_box_size)
    
    particle_z[z_within_plus] = particle_z[z_within_plus] + box_size
    particle_z[z_within_minus] = particle_z[z_within_minus] - box_size
    
    coord_diff[z_within_plus,0] = z_dist[z_within_plus] + box_size
    coord_diff[z_within_minus,0] = z_dist[z_within_minus] - box_size

    #calculate distance with standard sqrt((x_1-x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2)
    distance = np.zeros((new_particles,1))
    distance = np.sqrt(np.square((halo_x - particle_x)) + np.square((halo_y - particle_y)) + np.square((halo_z - particle_z)))
    
    return distance, coord_diff

#calculates density within sphere of given radius with given mass and calculating volume at each particle's radius
def calculate_density(masses, radius):
    volume = (4/3) * np.pi * np.power(radius,3)
    return masses/volume

#returns indices where density goes below overdensity value (200 * rho_c)
def check_where_r200(my_density):
    return np.where(my_density < (200 * rho_m))

#brute force method for finding particles around a halo
def brute_force(curr_particles_pos, r200, halo_x, halo_y, halo_z):
    #find particles within a 3D box with dimensions of r200 around halo
    within_box = curr_particles_pos[np.where((curr_particles_pos[:,0] < r200 + halo_x) & (curr_particles_pos[:,0] > r200 - halo_x) & (curr_particles_pos[:,1] < r200 + halo_y) & (curr_particles_pos[:,1] > r200 - halo_y) & (curr_particles_pos[:,2] < r200 + halo_z) & (curr_particles_pos[:,2] > r200 - halo_z))]
    #for those particles calculate all the radii    
    brute_radii = calculate_distance(halo_x, halo_y, halo_z, within_box[:,0], within_box[:,1], within_box[:,2], within_box.shape[0])
    #only return the particles that are within the r200 radius
    return within_box[np.where(brute_radii <= r200)]

calculated_r200 = np.zeros(halos_r200.size)
# particle_halo_assign: pid, halo_id, radius, x_dist, y_dist, z_dist, indices
particle_halo_assign = np.zeros((particles_pid.size,7))
particles_per_halo = np.zeros(num_halos)
start = 0

for i in range(num_halos):
    #find the indices of the particles within the expected r200 radius multiplied by 1.4 
    #value of 1.4 determined by guessing if just r200 value or 1.1 miss a couple halo r200 values but 1.4 gets them all
    indices = particle_tree.query_ball_point(halos_pos[i,:], r = 1.1 * halos_r200[i])

    # how many new particles being added
    num_new_particles = len(indices)
    particles_per_halo[i] = num_new_particles

    # sort the particles
    new_particles = np.sort(indices)

    #for how many new particles create an array of how much mass there should be within that particle radius
    use_mass = np.arange(1, num_new_particles + 1, 1) * mass

    #Only take the particle positions that where found with the tree
    current_particles_pos = particles_pos[indices,:]
    current_particles_pid = particles_pid[indices]
    current_halos_pos = halos_pos[i,:]
    
    #assign particles to their corresponding halos
    particle_halo_assign[start:start+num_new_particles,0] = current_particles_pid
    particle_halo_assign[start:start+num_new_particles,1] = i
        
    #calculate the radii of each particle based on the distance formula
    radius, coord_dist = calculate_distance(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], current_particles_pos[:,0],
                                current_particles_pos[:,1], current_particles_pos[:,2], num_new_particles)

    #sort the radii and positions to allow for creation of plots and to correctly assign how much mass there is
    arrsortrad = radius.argsort()
    radius = radius[arrsortrad[::1]]
    current_particles_pos = current_particles_pos[arrsortrad[::1]]
    coord_dist = coord_dist[arrsortrad[::1]]
    
    # divide radius by halo_r200 to scale
    # also save the coord_dist to use later to calculate unit vectors
    particle_halo_assign[start:start+num_new_particles,2] = radius/halos_r200[i]
    particle_halo_assign[start:start+num_new_particles,3] = coord_dist[:,0]
    particle_halo_assign[start:start+num_new_particles,4] = coord_dist[:,1]
    particle_halo_assign[start:start+num_new_particles,5] = coord_dist[:,2]
    particle_halo_assign[start:start+num_new_particles,6] = np.array(indices, dtype = int)  
    
    #calculate the density at each particle
    calculated_densities = np.zeros(num_new_particles)
    calculated_densities = calculate_density(use_mass, radius)
    
    #determine indices of particles where the expected r200 value is 
    indices_r200_met = check_where_r200(calculated_densities)
    
    #code used to get 3D visual of particles around a halo
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # sizes = np.full(num_new_particles, 10)
    # ax1.scatter(current_particles_pos[:,0], current_particles_pos[:,1], current_particles_pos[:,2], alpha = 0.08, marker = ".", c = 'r')
    # ax1.scatter(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], color = 'b', marker = "+", s = 25)
    # ax1.title.set_text("Tree Search")
    
    # brute_part = brute_force(particles_pos, halos_r200[i], current_halos_pos[0], current_halos_pos[1], current_halos_pos[2])
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.scatter(brute_part[:,0], brute_part[:,1], brute_part[:,2], alpha = 0.08, marker = ".", c = 'r')
    # ax2.scatter(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], color = 'b', marker = "+", s = 25)
    # ax2.title.set_text("Brute Force Search")
    # plt.show()
    
    #if only one index is less than 200 * rho_c then that is the r200 radius
    if indices_r200_met[0].size == 1:
        calculated_r200[i] = radius[indices_r200_met[0][0]]
    #if there are none then the radius is 0
    elif indices_r200_met[0].size == 0:
        calculated_r200[i] = 0
    #if multiple indices choose the first two and average them
    else:
        calculated_r200[i] = (radius[indices_r200_met[0][0]] + radius[indices_r200_met[0][1]])/2
    start += num_new_particles
    
difference_r200 = halos_r200 - calculated_r200

print("finish position")

# remove rows with zeros
particle_halo_assign = particle_halo_assign[~np.all(particle_halo_assign == 0, axis=1)]
# how many particles are we workin with
num_particles_identified = particle_halo_assign.shape[0]
# where are the separations between halos
indices_change = np.where(particle_halo_assign[:-1,1] != particle_halo_assign[1:,1])[0]  

# take indices from search and use to get velocities
use_particle_vel = np.zeros((num_particles_identified,3))
particle_indices = particle_halo_assign[:,6].astype(int)
use_particle_vel = particles_vel[particle_indices,:]
root_a = np.sqrt(scale_factor)
use_particle_vel = use_particle_vel * root_a

particles_vel_pec = np.zeros((num_particles_identified,3))
start = 0

# calculate peculiar velocity by subtracting halo velocity from particle velocity
for i in range(indices_change.size):
    finish = start + indices_change[i]
    particles_vel_pec[start:finish,:] = use_particle_vel[start:finish,:] - halos_vel[i,:]  
    start = finish

def calc_rhat(x_dist, y_dist, z_dist):
    rhat = np.zeros((x_dist.size,3))
    # get unit vector by dividing components by magnitude
    magnitude = np.sqrt(np.square(x_dist) + np.square(y_dist) + np.square(z_dist))
    rhat[:,0] = x_dist/magnitude
    rhat[:,1] = y_dist/magnitude
    rhat[:,2] = z_dist/magnitude

    return rhat


all_rhat = np.zeros((particles_vel_pec.shape[0],3))
particles_per_halo = particles_per_halo.astype(int)
particles_vel_phys = np.zeros((particles_vel_pec.shape[0],3))
particles_vel_tan = np.zeros((particles_vel_pec.shape[0],3))


# convert peculiar velocity to physical velocity
for i in range(particles_vel_pec.shape[0]):
    particles_vel_phys[i,:] = particles_vel_pec[i,:] + scale_factor * h * particle_halo_assign[i,2]


# calculate tangent velocity which is v_physical * the unit vector for that particle/halo
start = 0   
for i in range(particles_per_halo.size):
    finish = start + particles_per_halo[i]

    # calclulate all of the direction (unit) vectors
    all_rhat[start:finish,:] = calc_rhat(particle_halo_assign[start:finish,3], particle_halo_assign[start:finish,4],
                         particle_halo_assign[start:finish,5])
    
    particles_vel_tan[start:finish,:] = particles_vel_phys[start:finish,:] * all_rhat[start:finish,:]
    
    start = finish
# calculate radial velocity by projecting the physical velocity onto the corresponding unit vector
# (because unit vector don't bother dividing by the magnitude of the unit vector)
particles_vel_rad = (particles_vel_phys * all_rhat) * all_rhat

use_radius = particle_halo_assign[:,2]

#MAKE SURE TO BIN LOGARITHMICALLY
def make_bins(num_bins, radius, vel_rad):
    # remove the blank parts of the radius
    radius = radius[radius != 0]
    
    # sort the radius and radial velocity based off the radius
    arrsortrad = radius.argsort()
    radius = radius[arrsortrad[::1]]
    vel_rad = vel_rad[arrsortrad[::1]]

    min_dist = radius[0]
    max_dist = radius[-1]
    hist = np.histogram(radius, num_bins,range = (min_dist,max_dist))
    bins = hist[1]
    start = 0
    average_val = np.zeros((num_bins,4))
    for i in range(num_bins - 1):
        bin_size = bins[i + 1] - bins[i]
        finish = start + bin_size

        # make sure there are points within the bins
        indices = np.where((radius >= start) & (radius <= finish))
        if indices[0].size != 0:
            start_index = indices[0][0]
            end_index = indices[0][-1]

            # if there is only one point that meets the criteria just use that point
            if start_index == end_index:
                average_val[i,0] = np.mean(radius[start_index], dtype=np.float64)
                average_val[i,1] = vel_rad[start_index,0]
                average_val[i,2] = vel_rad[start_index,1]
                average_val[i,3] = vel_rad[start_index,2]
            # otherwise find the mean
            else:
                average_val[i,0] = np.mean(radius[start_index:end_index], dtype=np.float64)
                average_val[i,1] = np.nanmean(vel_rad[start_index:end_index,0]) 
                average_val[i,2] = np.nanmean(vel_rad[start_index:end_index,1])
                average_val[i,3] = np.nanmean(vel_rad[start_index:end_index,2])
                
        start = finish

    return average_val


num_bins = 100
avg_vel_rad = make_bins(num_bins, use_radius, particles_vel_rad)

graph1, (plot1) = plt.subplots(1,1)

plot1.set_title("average radial velocity vs position")
plot1.set_xlabel("position")
plot1.set_ylabel("average rad vel")
plot1.set_xscale("log")


total_avg_vel = np.sqrt(np.square(avg_vel_rad[:,1]) + np.square(avg_vel_rad[:,2]) + np.square(avg_vel_rad[:,2]))
plot1.scatter(avg_vel_rad[:,0], avg_vel_rad[:,1], color = 'r', label = "x comp")
plot1.scatter(avg_vel_rad[:,0], avg_vel_rad[:,2], color = 'b', label = "y comp")
plot1.scatter(avg_vel_rad[:,0], avg_vel_rad[:,3], color = 'g', label = "z comp")
plot1.scatter(avg_vel_rad[:,0], total_avg_vel, color = 'c', label = "magnitude")
plot1.legend()
plt.show()
