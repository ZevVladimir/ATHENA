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
particle_halo_assign_id = np.zeros((num_particles * 10, 3), dtype = np.int32)
particle_halo_radius_comp = np.zeros((num_particles * 10, 4), dtype = np.float32)
correspond_halo_prop = np.zeros((num_particles*5,2),dtype=np.float32)
halos_v200 = np.zeros(num_halos, dtype = np.float32)
all_halo_mass = np.zeros(num_halos, dtype = np.float32)
particles_per_halo = np.zeros(num_halos, dtype = np.int32)
start = 0

t1 = time.time()
print("start particle assign")
for i in range(num_halos):
    #find the indices of the particles within the expected r200 radius multiplied by 1.4 
    #value of 1.4 determined by guessing if just r200 value or 1.1 miss a couple halo r200 values but 1.4 gets them all
    indices = particle_tree.query_ball_point(halos_pos[i,:], r = 7 * halos_r200[i])

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
    halos_v200[i] = halo_v200
    # divide radius by halo_r200 to scale
    # also save the coord_dist to use later to calculate unit vectors
    particle_halo_radius_comp[start:start+num_new_particles,0] = radius
    particle_halo_radius_comp[start:start+num_new_particles,1] = coord_dist[:,0]
    particle_halo_radius_comp[start:start+num_new_particles,2] = coord_dist[:,1]
    particle_halo_radius_comp[start:start+num_new_particles,3] = coord_dist[:,2]
     
    correspond_halo_prop[start:start+num_new_particles,0] = halos_r200[i]
    correspond_halo_prop[start:start+num_new_particles,1] = halo_v200
    

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

# remove rows with zeros
particle_halo_assign_id = particle_halo_assign_id[~np.all(particle_halo_assign_id == 0, axis=1)]
particle_halo_radius_comp = particle_halo_radius_comp[~np.all(particle_halo_radius_comp == 0, axis=1)]
correspond_halo_prop = correspond_halo_prop[~np.all(correspond_halo_prop == 0, axis=1)]

np.save(save_location + "particle_halo_assign_id", particle_halo_assign_id)
np.save(save_location + "particle_halo_radius_comp", particle_halo_radius_comp)
np.save(save_location + "all_halo_mass", all_halo_mass)
np.save(save_location + "particles_per_halo", particles_per_halo)
np.save(save_location + "halos_v200", halos_v200)
np.save(save_location + "correspond_halo_prop", correspond_halo_prop)
