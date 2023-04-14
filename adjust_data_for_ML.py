import numpy as np
from pygadgetreader import *
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from colossus.halo import mass_so
from colossus.utils import constants
from matplotlib.pyplot import cm
import time

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
hubble_constant = cosmol.Hz(red_shift) * 0.001 # convert to units km/s/kpc
G = constants.G

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

halo_dict = {}

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

def calc_v200(mass, radius):
    return np.sqrt((G * mass)/radius)

calculated_r200 = np.zeros(halos_r200.size)
# particle_halo_assign: pid, halo_id, radius, x_dist, y_dist, z_dist, indices
particle_halo_assign_id = np.zeros((num_particles * 5, 3), dtype = np.int32)
particle_halo_radius_comp = np.zeros((num_particles * 5, 4), dtype = np.float32)
halos_v200 = np.zeros(num_halos, dtype = np.float32)
all_halo_mass = np.zeros(num_halos, dtype = np.float32)
particles_per_halo = np.zeros(num_halos, dtype = np.int32)
start = 0

t1 = time.time()
print("start particle assign")
for i in range(num_halos):
    #find the indices of the particles within the expected r200 radius multiplied by 1.4 
    #value of 1.4 determined by guessing if just r200 value or 1.1 miss a couple halo r200 values but 1.4 gets them all
    indices = particle_tree.query_ball_point(halos_pos[i,:], r = 6 * halos_r200[i])

    # how many new particles being added
    num_new_particles = len(indices)

    particles_per_halo[i] = num_new_particles

    # sort the particles
    new_particles = np.sort(indices)

    #for how many new particles create an array of how much mass there should be within that particle radius
    use_mass = np.arange(1, num_new_particles + 1, 1) * mass

    #Only take the particle positions that where found with the tree
    current_particles_pos = particles_pos[indices,:]
    current_particles_vel = particles_vel[indices,:]
    current_particles_pid = particles_pid[indices]
    current_halos_pos = halos_pos[i,:]
    
    #assign particles to their corresponding halos
    particle_halo_assign_id[start:start+num_new_particles,0] = current_particles_pid
    particle_halo_assign_id[start:start+num_new_particles,1] = i
    particle_halo_assign_id[start:start+num_new_particles,2] = np.array(indices, dtype = np.int32) 
        
    #calculate the radii of each particle based on the distance formula
    radius, coord_dist = calculate_distance(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], current_particles_pos[:,0],
                                current_particles_pos[:,1], current_particles_pos[:,2], num_new_particles)

    #sort the radii and positions to allow for creation of plots and to correctly assign how much mass there is
    arrsortrad = radius.argsort()
    radius = radius[arrsortrad[::1]]
    current_particles_pos = current_particles_pos[arrsortrad[::1]]
    current_particles_vel = current_particles_vel[arrsortrad[::1]]
    coord_dist = coord_dist[arrsortrad[::1]]
    
    halo_v200 = calc_v200(use_mass[-1], halos_r200[i])

    # divide radius by halo_r200 to scale
    # also save the coord_dist to use later to calculate unit vectors
    particle_halo_radius_comp[start:start+num_new_particles,0] = radius
    particle_halo_radius_comp[start:start+num_new_particles,1] = coord_dist[:,0]
    particle_halo_radius_comp[start:start+num_new_particles,2] = coord_dist[:,1]
    particle_halo_radius_comp[start:start+num_new_particles,3] = coord_dist[:,2]
     
    halos_v200[i] = halo_v200
    

    all_halo_mass[i] = use_mass[-1]
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
t2 = time.time()
print("finish particle assign: ", (t2 - t1), " seconds")
###########################################################################################################

###########################################################################################################

# remove rows with zeros
particle_halo_assign_id = particle_halo_assign_id[~np.all(particle_halo_assign_id == 0, axis=1)]
particle_halo_radius_comp = particle_halo_radius_comp[~np.all(particle_halo_radius_comp == 0, axis=1)]
particle_distances = particle_halo_radius_comp[:,0]

# how many particles are we workin with
num_particles_identified = particle_halo_assign_id.shape[0]
# where are the separations between halos
indices_change = np.where(particle_halo_assign_id[:-1,1] != particle_halo_assign_id[1:,1])[0]  
indices_change = np.append(indices_change, particle_halo_assign_id.shape[0])
mass_hist = np.histogram(all_halo_mass, 1000)
halos_mass_indices = np.where((all_halo_mass > mass_hist[1][5]) & (all_halo_mass < mass_hist[1][6]))[0]
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
    particles_vel_pec[start_vel_pec:finish_vel_pec,:] = use_particle_vel[start_vel_pec:finish_vel_pec,:] - halos_vel[i,:] 
    #radius_div_r200[start_vel_pec:finish_vel_pec] = particle_distances[start_vel_pec:finish_vel_pec]/halos_r200[i]
    start_vel_pec = finish_vel_pec
print(1)
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


# convert peculiar velocity to physical velocity by adding scale factor * h * dist from particle to halo
start_vel_phys = 0
for i in range(indices_change.size):
    finish_vel_phys = indices_change[i]
    particles_vel_phys[start_vel_phys:finish_vel_phys,:] = particles_vel_pec[start_vel_phys:finish_vel_phys,:] + np.reshape((hubble_constant * particle_distances[start_vel_phys:finish_vel_phys]),((finish_vel_phys - start_vel_phys),1))
    # print(particles_vel_pec[start_vel_phys:finish_vel_phys,:])
    # print(np.reshape((scale_factor * h * particle_distances[start_vel_phys:finish_vel_phys]),((finish_vel_phys - start_vel_phys),1)))
    start_vel_phys = finish_vel_phys
# calculate tangent velocity which is v_physical * the unit vector for that particle/halo
print(2)
# start_vel_tan = 0   
# for i in range(particles_per_halo.size):
#     finish_vel_tan = start_vel_tan + particles_per_halo[i]

#     # calclulate all of the direction (unit) vectors
#     all_rhat[start_vel_tan:finish_vel_tan,:] = calc_rhat(particle_halo_assign[start_vel_tan:finish_vel_tan,3], particle_halo_assign[start_vel_tan:finish_vel_tan,4],
#                          particle_halo_assign[start_vel_tan:finish,5])
    
#     particles_vel_tan[start_vel_tan:finish_vel_tan,:] = particles_vel_phys[start_vel_tan:finish_vel_tan,:] * all_rhat[start_vel_tan:finish_vel_tan,:]
    
#     start_vel_tan = finish_vel_tan

all_rhat = calc_rhat(particle_halo_radius_comp[:,1], particle_halo_radius_comp[:,2],particle_halo_radius_comp[:,3])

# calculate radial velocity by projecting the physical velocity onto the corresponding unit vector
start_vel_rad = 0

for i in range(indices_change.size):
    finish_vel_rad = indices_change[i]
    #curr_halo_v200 = np.ones((finish_vel_rad - start_vel_rad)) * halos_v200[i]
    
    particles_vel_rad[start_vel_rad:finish_vel_rad] = np.sum((particles_vel_phys[start_vel_rad:finish_vel_rad] * all_rhat[start_vel_rad:finish_vel_rad]), axis = 1) 
    #/ curr_halo_v200
    start_vel_rad = finish_vel_rad
print(3)
#MAKE SURE TO BIN LOGARITHMICALLY
def make_bins(num_bins, radius, vel_rad):
    # remove the blank parts of the radius
    radius = radius[radius != 0]
    sorted_radius = np.sort(radius)
    min_dist = sorted_radius[0]
    max_dist = sorted_radius[-1]
    hist = np.histogram(sorted_radius, num_bins,range = (min_dist,max_dist))
    bins = hist[1]
    bin_start = 0
    average_val = np.zeros((num_bins,2))
    
    for i in range(num_bins - 1):
        bin_size = bins[i + 1] - bins[i]
        bin_finish = bin_start + bin_size

        # make sure there are points within the bins
        indices = np.where((radius >= bin_start) & (radius <= bin_finish))
        if indices[0].size != 0:
            use_vel_rad = vel_rad[indices]
            average_val[i,0] = np.mean(np.array([bin_start,bin_finish]))
            average_val[i,1] = np.nanmean(use_vel_rad) 

        bin_start = bin_finish
    return average_val

print("start binning")
num_bins = 1000
mass_bin_radius = np.zeros(particle_distances.size)
mass_bin_vel_rad = np.zeros(particles_vel_rad.size)


start_mass_bin = 0
finish_mass_bin = 0
# for i in range(halos_use_indices.shape[0]):
#     finish_mass_bin += (halos_use_indices[i][1] - halos_use_indices[i][0])
#     mass_bin_radius[start_mass_bin:finish_mass_bin] = particle_distances[halos_use_indices[i][0]:halos_use_indices[i][1]]
#     mass_bin_vel_rad[start_mass_bin:finish_mass_bin] = particles_vel_rad[halos_use_indices[i][0]:halos_use_indices[i][1]]
#     start_mass_bin = finish_mass_bin

mass_bin_radius = mass_bin_radius[mass_bin_radius != 0]
#avg_vel_rad = make_bins(num_bins, mass_bin_radius, mass_bin_vel_rad)
avg_vel_rad = make_bins(num_bins, particle_distances, particles_vel_rad)
avg_vel_rad = avg_vel_rad[~np.all(avg_vel_rad == 0, axis=1)]

graph1, (plot1) = plt.subplots(1,1)
plot1.plot(avg_vel_rad[:,0], hubble_constant * avg_vel_rad[:,0], color = "blue", label = "hubble flow")
plot1.plot(avg_vel_rad[:,0], avg_vel_rad[:,1], color = "purple", label = "particles")
plot1.set_title("average radial velocity vs position all particles")
plot1.set_xlabel("position $kpc$")
plot1.set_ylabel("average rad vel $km/s$")
plot1.set_xscale("log")    
#plot1.set_ylim([-.3,.3])
plot1.legend()
t3 = time.time()
print("velocity finished: ", (t3 - t2)," seconds")
print("total time: ", (t3-t1), " seconds")
plt.show()
