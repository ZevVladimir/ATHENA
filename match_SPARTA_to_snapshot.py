import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pygadgetreader import *
import time
from scipy import constants 
from colossus.cosmology import cosmology
from colossus.halo import mass_so

file = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zeevvladimir/ML_orbit_infall_project/Original_Data/snapshot_190/snapshot_0190"

cosmol = cosmology.setCosmology("illustris")
red_shift = readheader(snapshot_path, 'redshift')

rho_c = cosmol.rho_c(red_shift)
rho_m = cosmol.rho_m(red_shift)

#Shape particle_with_halo: pid, halo_pos_x, halo_pos_y, halo_pos_z, halo_vel_x, halo_vel_y, halo_vel_z, halo_number, halo_r200
particles_with_halo = np.load(file + "particle_with_halo.npy")


particles_pid = np.load(file + "particle_pid.npy")
particles_vel = np.load(file + "particle_vel.npy")
particles_pos = np.load(file + "particle_pos.npy") 
particles_mass = np.load(file + "all_particle_mass.npy")
mass = particles_mass[0] * 10**10 #units M_sun/h

scale_factor = 1/(1+red_shift)

# match_indices = np.intersect1d(particle_with_halo[:,0], particles_pid, return_indices=True)
# #match_indices: sorted pids, indices for matches in particle_with_halo, indices for matches in particles_pid
# matched_particles_with_halo = particle_with_halo[match_indices[1]]


#only take pos and vel that are matched
# matched_particles_pid = particles_pid[match_indices[2]]
# matched_particles_pos = particles_pos[match_indices[2]]
particles_pos_x = particles_pos[:,0] * 10**3 * scale_factor
particles_pos_y = particles_pos[:,1] * 10**3 * scale_factor
particles_pos_z = particles_pos[:,2] * 10**3 * scale_factor
print(particles_pos_x.shape)
print(particles_pos_y.shape)
print(particles_pos_z.shape)

halo_pos_x = particles_with_halo[:,1] * 10**3 * scale_factor
halo_pos_y = particles_with_halo[:,2] * 10**3 * scale_factor
halo_pos_z = particles_with_halo[:,3] * 10**3 * scale_factor
print(halo_pos_x.shape)
print(halo_pos_y.shape)
print(halo_pos_z.shape)

box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
box_size = box_size * 10**3 * scale_factor #convert to Kpc/h physical


#calculate distance of particle from halo
def calculate_distance(halo_x, halo_y, halo_z, particle_x, particle_y, particle_z):
    distance = np.zeros((particles_with_halo[1].size,1))
    distance = np.sqrt(np.square((halo_x - particle_x)) + np.square((halo_y - particle_y)) + np.square((halo_z - particle_z)))
    
    return distance 



# for i in range(halo_pos_x.size):
#     #if the distance (x,y,z) goes over the boundary to the left add the box_size
#     #if the distance (x,y,z) goes over the boundary to the right subtract the box_size
#     if halo_pos_x[i] - particles_pos_x[i] > box_size/2:
#         particles_pos_x[i] = particles_pos_x[i] + box_size
#     elif halo_pos_x[i] - particles_pos_x[i] < -box_size/2:
#         particles_pos_x[i] = particles_pos_x[i] - box_size
    
#     if halo_pos_y[i] - particles_pos_y[i] > box_size/2:
#         particles_pos_y[i] = particles_pos_y[i] + box_size
#     elif halo_pos_y[i] - particles_pos_y[i] < -box_size/2:
#         particles_pos_y[i] = particles_pos_y[i] - box_size

#     if halo_pos_z[i] - particles_pos_z[i] > box_size/2:
#         particles_pos_z[i] = particles_pos_z[i] + box_size
#     elif halo_pos_z[i] - particles_pos_z[i] < -box_size/2:
#         particles_pos_z[i] = particles_pos_z[i] - box_size

radius = calculate_distance(halo_pos_x, halo_pos_y, halo_pos_z, particles_pos_x, particles_pos_y, particles_pos_z)

print("Box size: " + str(box_size))
print("distances")
print(np.sort(radius))



#create array which has the points where a new halo starts
indices_split = np.where(particles_with_halo[:,7][:-1] != particles_with_halo[:,7][1:])[0]
indices_split = indices_split + 1 #so the values stored here are the starting values for each halo
halo_splitting_point = np.zeros((indices_split.size + 1), dtype=int)
halo_splitting_point[0] = 0
halo_splitting_point[1:] = indices_split


# current_R200 = 0
# start_particle = 0
# final_particle = 0
# r200_values =  particles_with_halo[:,8]
# r200_values = r200_values[halo_splitting_point]
# # Checking how many particles are around each halo
# for index in range(1, halo_splitting_point.size):
#     new_particles = halo_splitting_point[index] - halo_splitting_point[index-1]
#     final_particle = final_particle + new_particles
#     current_R200 = r200_values[index]

#     current_halo_particles = np.zeros((int(halo_splitting_point[index]),3))
#     current_halo_particles[0:new_particles,0] = particles_with_halo[start_particle:final_particle,0] #add PIDs
#     current_halo_particles[0:new_particles,1] = radius[start_particle:final_particle] #add how far the particle is from the halo
#     current_halo_particles = current_halo_particles[~np.all(current_halo_particles == 0, axis=1)] #remove zeros
#     current_halo_particles = current_halo_particles[current_halo_particles[:,1].argsort()] #sort the particles by distance
    
#     mass_values = np.zeros(new_particles)
#     mass_values = mass * np.arange(1, new_particles + 1,1) 
#     mass_values.reshape(new_particles,1)
#     current_halo_particles[0:new_particles,2] = mass_values

#     mass_within_act_R200 = np.where(current_halo_particles[:,1] < current_R200)
#     mass_within_act_R200 = current_halo_particles[mass_within_act_R200,2]
#     total_mass_within = np.sum(mass_within_act_R200)

#     actual_mass = mass_so.R_to_M(current_R200, red_shift, "200m")
#     #print("calculated mass: " + str(total_mass_within))
#     #print("actual mass: " + str(actual_mass))
#     start_particle = final_particle


start_particle = 0
final_particle = 0
current_particle = 0
calculated_R200 = np.zeros(halo_splitting_point.size - 1)
actual_calculated_indices = np.zeros(halo_splitting_point.size - 1)
actual_count = 0
halo_count = 0
miss = 0
t1 = time.time()
array_two_r200 = np.zeros(2,)

print("Start")
#calculate for each halo the M200m and then keep the largest one
#for index in range(1, halo_splitting_point.size):
for index in range(1, 10000):
    #add the difference between the new and old so only get new particles
    new_particles = halo_splitting_point[index] - halo_splitting_point[index-1]
    final_particle = final_particle + new_particles
    
    #create array for each halos particles with their PIDs, the particles distances, and the mass
    current_halo_particles = np.zeros((int(halo_splitting_point[index]),3))
    current_halo_particles[0:new_particles,0] = particles_with_halo[start_particle:final_particle,0] #add PIDs
    current_halo_particles[0:new_particles,1] = radius[start_particle:final_particle] #add how far the particle is from the halo
    current_halo_particles = current_halo_particles[~np.all(current_halo_particles == 0, axis=1)] #remove zeros
    current_halo_particles = current_halo_particles[current_halo_particles[:,1].argsort()] #sort the particles by distance
    

    #for each particle give how many particles there are (including it) within its radius then multiply by mass
    mass_values = np.zeros(new_particles)
    mass_values = mass * np.arange(1, new_particles + 1,1) 
    mass_values.reshape(new_particles,1)

    current_halo_particles[0:new_particles,2] = mass_values

    #density = mass of particles within radius/((4/3)*pi*particle radius^3)
    calculated_density = np.divide((current_halo_particles[0:new_particles,2]),((4/3)*np.pi*(np.power(current_halo_particles[0:new_particles,1],3))))
    indices_density_reached = np.where(calculated_density >= (200*rho_c))

    if(indices_density_reached[0].size > 1):
        # If we have multiple particles at and beyond R200 average the first 2's radii
        array_two_r200[0] = current_halo_particles[indices_density_reached[0][0],1]
        array_two_r200[1] = current_halo_particles[indices_density_reached[0][1],1]
        calculated_R200[actual_count] = np.mean(array_two_r200)
        actual_calculated_indices[actual_count] = halo_count #keep track of which halos actaully got an R200
        actual_count += 1
    elif(indices_density_reached[0].size == 1):
        # If there is only one particle at R200 just use that radius
        calculated_R200[actual_count] = current_halo_particles[indices_density_reached[0][0],1]
        actual_calculated_indices[actual_count] = halo_count
        actual_count += 1
        
    
    halo_count += 1
    
    if (index % 100000 == 0 and index != 0):
        print(index)
        t2 = time.time()
        print(t2 - t1)
    start_particle = final_particle
print("calculated R200")

print(calculated_R200[:10])
# print(np.max(calculated_R200))
# print(np.min(calculated_R200))

print("actual R200")

r200_indices_split = np.where(particles_with_halo[:,8][:-1] != particles_with_halo[:,8][1:])[0]
r200_indices_split = r200_indices_split + 1 #so the values stored here are the starting values for each halo
r200_splitting_point = np.zeros((r200_indices_split.size + 1), dtype=int)
r200_splitting_point[0] = 0
r200_splitting_point[1:] = r200_indices_split

actual_calculated_indices = actual_calculated_indices[actual_calculated_indices != 0]
actual_calculated_indices = (np.rint(actual_calculated_indices)).astype(int)
actual_R200 = particles_with_halo[:,8]
actual_R200 = actual_R200[r200_splitting_point]
actual_R200 = actual_R200[actual_calculated_indices]



print(actual_R200[:10])
print(np.max(actual_R200))
print(np.min(actual_R200))

# #subtract velocitions
# matched_particles_vel[:,0] = matched_particles_vel[:,0] - matched_particles_with_halo[:,1]
# matched_particles_vel[:,1] = matched_particles_vel[:,1] - matched_particles_with_halo[:,2]
# matched_particles_vel[:,2] = matched_particles_vel[:,2] - matched_particles_with_halo[:,3]