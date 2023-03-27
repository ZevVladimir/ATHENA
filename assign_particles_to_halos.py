import numpy as np
import time
from pygadgetreader import *
from scipy.spatial import cKDTree
from scipy import constants
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from colossus.halo import mass_so
from colour import Color



file = "/home/zvladimi/ML_orbit_infall_project/np_arrays/"
save_location =  "/home/zvladimi/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zvladimi/ML_orbit_infall_project/particle_data/snapshot_192/snapshot_0192"

# get constants from pygadgetreader
snapshot = 192
red_shift = readheader(snapshot_path, 'redshift')
print(red_shift)
scale_factor = 1/(1+red_shift)
cosmol = cosmology.setCosmology("bolshoi")
rho_c = cosmol.rho_c(red_shift)
rho_m = cosmol.rho_m(red_shift)
h = readheader(snapshot_path, 'h')


box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
box_size = box_size * 10**3 * scale_factor #convert to Kpc/h physical
print("box_size", box_size)

#load particle info
particles_pid = np.load(file + "particle_pid_192.npy")
particles_vel = np.load(file + "particle_vel_192.npy")
particles_pos = np.load(file + "particle_pos_192.npy") 
particles_pos = particles_pos * 10**3 * scale_factor #convert to kpc and physical
particles_mass = np.load(file + "all_particle_mass_192.npy")
mass = particles_mass[0] * 10**10 #units M_sun/h

# print("total mass", mass * particles_mass.size)
# print("colossus total mass", (box_size **3)*rho_m)


#load halo info at snapshot
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

num_particles = particles_pid.size

# remove all halos for any halo that doesn't exist beyond snapshot 
# remove all halos that aren't main halos (identified with tag = 10)
indices_keep = np.zeros((halos_id.size))
indices_keep = np.where((halos_last_snap >= snapshot) & (halos_status[:,snapshot] == 10))
halos_pos = halos_pos[indices_keep]
halos_vel = halos_vel[indices_keep]
halos_r200 = halos_r200[indices_keep]
halos_id = halos_id[indices_keep]
num_halos = halos_r200.size

#construct a search tree iwth all of the particle positions
particle_tree = cKDTree(data = particles_pos, leafsize = 3, balanced_tree = False, boxsize = box_size)


#calculate distance of particle from halo
def calculate_distance(halo_x, halo_y, halo_z, particle_x, particle_y, particle_z, new_particles):
    x_dist = particle_x - halo_x
    #print(x_dist)
    y_dist = particle_y - halo_y
    #print(y_dist)
    z_dist = particle_z - halo_z
    #print(z_dist)
    half_box_size = box_size/2
    
    #handles periodic boundary conditions by checking if you were to add or subtract a boxsize would it then
    #be within half a box size of the halo
    #do this for x, y, and z coords
    x_within_plus = np.where((x_dist + box_size) < half_box_size)
    x_within_minus = np.where((x_dist + box_size) < -half_box_size)
    
    particle_x[x_within_plus] = particle_x[x_within_plus] + box_size
    particle_x[x_within_minus] = particle_x[x_within_minus] - box_size
    
    y_within_plus = np.where((y_dist + box_size) < half_box_size)
    y_within_minus = np.where((y_dist + box_size) < -half_box_size)
    
    particle_y[y_within_plus] = particle_y[y_within_plus] + box_size
    particle_y[y_within_minus] = particle_y[y_within_minus] - box_size
    
    z_within_plus = np.where((z_dist + box_size) < half_box_size)
    z_within_minus = np.where((z_dist + box_size) < -half_box_size)
    
    particle_z[z_within_plus] = particle_z[z_within_plus] + box_size
    particle_z[z_within_minus] = particle_z[z_within_minus] - box_size

    #calculate distance with standard sqrt((x_1-x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2)
    distance = np.zeros((new_particles,1))
    distance = np.sqrt(np.square((halo_x - particle_x)) + np.square((halo_y - particle_y)) + np.square((halo_z - particle_z)))
    
    return distance
    
def calc_mass_overdensity(radius):
    return (4/3) * np.pi * (200 * rho_m) * np.power(radius,3)
#calculates density within sphere of given radius with given mass and calculating volume at each particle's radius
def calculate_density(masses, radius):
    volume = (4/3) * np.pi * np.power(radius,3)
    # print("radius", radius[:10])
    #print("mass", masses[-10:])
    #print("volume ", volume[-10:])
    return masses/volume

#returns indices where density goes below overdensity value (200 * rho_c)
def check_where_r200(my_density):
    return np.where(my_density < (200 * rho_m))

def brute_force(curr_particles_pos, r200, halo_x, halo_y, halo_z):
    within_box = curr_particles_pos[np.where((curr_particles_pos[:,0] < r200 + halo_x) & (curr_particles_pos[:,0] > r200 - halo_x) & (curr_particles_pos[:,1] < r200 + halo_y) & (curr_particles_pos[:,1] > r200 - halo_y) & (curr_particles_pos[:,2] < r200 + halo_z) & (curr_particles_pos[:,2] > r200 - halo_z))]
    brute_radii = calculate_distance(halo_x, halo_y, halo_z, within_box[:,0], within_box[:,1], within_box[:,2], within_box.shape[0])
    return within_box[np.where(brute_radii <= r200)]

calculated_r200 = np.zeros(halos_r200.size)
t1 = time.time()
count = 1

mass_overdensity = calc_mass_overdensity(halos_r200)
num_iter = 50
ratio_mass = np.zeros(num_iter)

for i in range(num_halos):
#for i in range(num_iter):
    # i = 104
    #find the indices of the particles within the expected r200 radius
    indices = particle_tree.query_ball_point(halos_pos[i,:], r = 1.4 * halos_r200[i])

    #new particles being added
    num_new_particles = len(indices)
    actual_mass = mass_so.R_to_M(halos_r200[i], red_shift, "200m")
    
    #print(num_new_particles, " new particles")
    new_particles = np.sort(indices)

    #for how many new particles create an array of how much mass there should be
    use_mass = np.arange(1, num_new_particles + 1, 1) * mass
    # print("found mass: ", use_mass[num_new_particles - 1])
    # print("actual mass: ", actual_mass)
    # print(use_mass[num_new_particles - 1]/mass)
    # print(actual_mass/mass)
    #ratio_mass[i] = use_mass[num_new_particles-1]

    #Only take the positions that where found with the tree
    current_particles_pos = particles_pos[indices,:]
    
    current_halos_pos = halos_pos[i,:]
    
    #calculate the radii of each particle based on the distance formula
    radius = calculate_distance(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], current_particles_pos[:,0],
                                current_particles_pos[:,1], current_particles_pos[:,2], num_new_particles)
     
    #radius = np.sort(radius)
    #sort the radii and positions to allow for creation of plots and to correctly assign how much mass there is
    arrsortrad = radius.argsort()
    radius = radius[arrsortrad[::1]]
    current_particles_pos = current_particles_pos[arrsortrad[::1]]
    #print("masses: ", use_mass)
    
    #calculate the density at each particle
    calculated_densities = np.zeros(num_new_particles)
    calculated_densities = calculate_density(use_mass, radius)
    
    #determine where the expected r200 value is 
    indices_r200_met = check_where_r200(calculated_densities)
    
    # print("Where calculated density is below: ", indices_r200_met[0][:8])
    # print("radius: ",radius[indices_r200_met[0][:8]])
    # print("densities: ", calculated_densities[indices_r200_met[0][:8]])
    
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
        print(indices_r200_met)
        print(calculated_densities)
        print(200 * rho_m)
        calculated_r200[i] = 0
    #if multiple indices choose the first two and average them
    else:
        calculated_r200[i] = (radius[indices_r200_met[0][0]] + radius[indices_r200_met[0][1]])/2
    
    # if (i % 1500 == 0 and i != 0):
    #     t2 = time.time()
    #     print(i, (t2 - t1))
    #     print("remaining: ", (num_halos - (count * 1500)))
    #     count += 1
    
    

#print(np.divide(ratio_mass,mass_overdensity[:num_iter]))
# for i in range(num_iter):

#   print("actual r200:",halos_r200[i])
#   print("calc r200:",calculated_r200[i])
#   print(i, "difference in r200:", halos_r200[i] - calculated_r200[i])
difference_r200 = halos_r200 - calculated_r200
print(np.where(calculated_r200 == 0))
print("largest difference: ", np.max(difference_r200))
print("smallest difference: ", np.min(difference_r200))
print("median difference: ", np.median(difference_r200))
print("average difference: ", np.average(difference_r200))
#plt.show()