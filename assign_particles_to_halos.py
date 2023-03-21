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
particles_pos = particles_pos * 10**3 * scale_factor
particles_mass = np.load(file + "all_particle_mass_192.npy")
mass = particles_mass[0] * 10**10 #units M_sun/h

# print("total mass", mass * particles_mass.size)
# print("colossus total mass", (box_size **3)*rho_m)


#load halo info
halos_pos = np.load(file + "halo_position.npy")
halos_pos = halos_pos[:,snapshot,:] * 10**3 * scale_factor
halos_vel = np.load(file + "halo_velocity.npy")
halos_vel = halos_vel[:,snapshot,:]
halos_last_snap = np.load(file + "halo_last_snap.npy")
halos_r200 = np.load(file + "halo_R200m.npy")
halos_r200 = halos_r200[:,snapshot]
halos_id = np.load(file + "halo_id.npy")
halos_id = halos_id[:,snapshot]
halos_status = np.load(file + "halo_status.npy")

num_particles = particles_pid.size
num_halos = halos_last_snap.size

# remove all halos for any halo that doesn't exist beyond snapshot 190
indices_keep = np.zeros((num_halos))
indices_keep = np.where((halos_last_snap >= snapshot) & (halos_status[:,snapshot] == 10))
halos_pos = halos_pos[indices_keep]
halos_vel = halos_vel[indices_keep]
halos_r200 = halos_r200[indices_keep]
halos_id = halos_id[indices_keep]
new_num_halos = halos_r200.size

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

    distance = np.zeros((new_particles,1))
    distance = np.sqrt(np.square((halo_x - particle_x)) + np.square((halo_y - particle_y)) + np.square((halo_z - particle_z)))
    
    return distance
    
def calc_mass_overdensity(radius):
    return (4/3) * np.pi * (200 * rho_c) * np.power(radius,3)
#calculates density within sphere of given radius with given mass 
def calculate_density(masses, radius):
    volume = (4/3) * np.pi * np.power(radius,3)
    # print("radius", radius[:10])
    #print("mass", masses[-10:])
    #print("volume ", volume[-10:])
    return masses/volume

#returns indices where density goes below overdensity value
def check_where_r200(my_density):
    #print("200* rho critical: ", 200 * rho_c)
    return np.where(my_density < (200 * rho_c))

calculated_r200 = np.zeros(halos_r200.size)
t1 = time.time()
count = 1

mass_overdensity = calc_mass_overdensity(halos_r200)
num_iter = 25
ratio_mass = np.zeros(num_iter)


#for i in range(len(indices)):
for i in range(num_iter):
    #find the indices of the particles within the radius
    indices = particle_tree.query_ball_point(halos_pos[i,:], r = 1.1 * halos_r200[i])

    #print(1.5 * halos_r200[i])
    #new particles being added
    num_new_particles = len(indices)
    
    print(num_new_particles, " new particles")
    new_particles = np.sort(indices)

    #print("r200 volume:", (4/3 * np.pi * np.power(halos_r200[i],3)))
    #for how many new particles create an array of how much mass there should be
    use_mass = np.arange(1, num_new_particles + 1, 1) * mass
    
    ratio_mass[i] = use_mass[num_new_particles-1]
    # print(use_mass)
    #Only take the positions that where found with the tree
    current_particles_pos = particles_pos[indices,:]
    
    current_halos_pos = halos_pos[i,:]
    
    radius = calculate_distance(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], current_particles_pos[:,0],
                                current_particles_pos[:,1], current_particles_pos[:,2], num_new_particles)
     
    #radius = np.sort(radius)
    arrsortrad = radius.argsort()
    radius = radius[arrsortrad[::1]]
    current_particles_pos = current_particles_pos[arrsortrad[::1]]
    #print("masses: ", use_mass)
    
    calculated_densities = np.zeros(num_new_particles)
    calculated_densities = calculate_density(use_mass, radius)
    
    print(200 * rho_c)
    
    indices_r200_met = check_where_r200(calculated_densities)
    print("Where calculated density is below: ", indices_r200_met[0][:8])
    print("radius: ",radius[indices_r200_met[0][:8]])
    print("densities: ", calculated_densities[indices_r200_met[0][:8]])
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(current_particles_pos[:,0], current_particles_pos[:,1], current_particles_pos[:,2], cmap = 'gray', marker = ".", c = 'r')
    # ax.scatter(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], color = 'b', marker = "*")
    # # plt.xlim([0,box_size])
    # # plt.ylim([0,box_size])
    
    # plt.show()
    
    
    if indices_r200_met[0].size == 1:
        calculated_r200[i] = radius[indices_r200_met[0][0]]
    elif indices_r200_met[0].size == 0:
        calculated_r200[i] = 0
    else:
        calculated_r200[i] = (radius[indices_r200_met[0][0]] + radius[indices_r200_met[0][1]])/2
    
    if (i % 2500 == 0 and i != 0):
        t2 = time.time()
        print(i, (t2 - t1))
        print("remaining: ", (len(indices) - (count * 2500)))
    
    

print(np.divide(ratio_mass,mass_overdensity[:num_iter]))

# for i in range(num_iter):
#     print("calculated R200: ", calculated_r200[i])
#     print("Actual R200: ", halos_r200[i])
#   print(halos_r200[i] - calculated_r200[i])