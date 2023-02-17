import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pygadgetreader import *
import time
from scipy import constants 
from colossus.cosmology import cosmology

file = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zeevvladimir/ML_orbit_infall_project/Original_Data/snapshot_190/snapshot_0190"

cosmol = cosmology.setCosmology("illustris")
red_shift = readheader(snapshot_path, 'redshift')

rho_c = cosmol.rho_c(red_shift)
rho_m = cosmol.rho_m(red_shift)

#Shape particle_with_halo: pid, halo_pos_x, halo_pos_y, halo_pos_z, halo_vel_x, halo_vel_y, halo_vel_z, halo_number, halo_r200
particle_with_halo = np.load(file + "particle_with_halo.npy")
particle_with_halo[:,1:3] = particle_with_halo[:,1:3] #in units of kpc/h
particles_pid = np.load(file + "particle_pid.npy")
particles_vel = np.load(file + "particle_vel.npy")
particles_pos = np.load(file + "particle_pos.npy") 
particles_pos = particles_pos #units kpc/h
particles_mass = np.load(file + "all_particle_mass.npy")
mass = particles_mass[0]  #units M_sun/h


match_indices = np.intersect1d(particle_with_halo[:,0], particles_pid, return_indices=True)
#match_indices: sorted pids, indices for matches in particle_with_halo, indices for matches in particles_pid
matched_particles_with_halo = particle_with_halo[match_indices[1]]

#only take pos and vel that are matched
matched_particles_pid = particles_pid[match_indices[2]]
matched_particles_pos = particles_pos[match_indices[2]]
matched_particles_pos_x = matched_particles_pos[:,0]
matched_particles_pos_y = matched_particles_pos[:,1]
matched_particles_pos_z = matched_particles_pos[:,2]
matched_particles_vel = particles_vel[match_indices[2]]

matched_halo_pos_x = matched_particles_with_halo[:,1]
matched_halo_pos_y = matched_particles_with_halo[:,2]
matched_halo_pos_z = matched_particles_with_halo[:,3]

box_size = readheader(snapshot_path, 'boxsize') #units kpc/h

#calculate distance of particle from halo
def calculate_distance(halo_x, halo_y, halo_z, particle_x, particle_y, particle_z):
    distance = np.zeros((matched_particles_with_halo[1].size,1))
    distance = np.sqrt(np.square((halo_x - particle_x)) + np.square((halo_y - particle_y)) + np.square((halo_z - particle_z)))
    
    return distance 

radius = calculate_distance(matched_halo_pos_x, matched_halo_pos_y, matched_halo_pos_z, matched_particles_pos_x, matched_particles_pos_y, matched_particles_pos_z)


for i in range(radius.size):
    #if the radius is too large
    if radius[i] > box_size/2:
        #if the distance (x,y,z) goes over the boundary to the left add the box_size
        #if the distance (x,y,z) goes over the boundary to the right subtract the box_size
        if matched_halo_pos_x[i] - matched_particles_pos_x[i] > box_size/2:
            matched_particles_pos_x[i] = matched_particles_pos_x[i] + box_size
        elif matched_halo_pos_x[i] - matched_particles_pos_x[i] < -box_size/2:
            matched_particles_pos_x[i] = matched_particles_pos_x[i] - box_size
        
        if matched_halo_pos_y[i] - matched_particles_pos_y[i] > box_size/2:
            matched_particles_pos_y[i] = matched_particles_pos_y[i] + box_size
        elif matched_halo_pos_y[i] - matched_particles_pos_y[i] < -box_size/2:
            matched_particles_pos_y[i] = matched_particles_pos_y[i] - box_size

        if matched_halo_pos_z[i] - matched_particles_pos_z[i] > box_size/2:
            matched_particles_pos_z[i] = matched_particles_pos_z[i] + box_size
        elif matched_halo_pos_z[i] - matched_particles_pos_z[i] < -box_size/2:
            matched_particles_pos_z[i] = matched_particles_pos_z[i] - box_size

radius = calculate_distance(matched_halo_pos_x, matched_halo_pos_y, matched_halo_pos_z, matched_particles_pos_x, matched_particles_pos_y, matched_particles_pos_z)

print("distances")
print(np.sort(radius))



#create array which has the points where a new halo starts
indices_split = np.where(matched_particles_with_halo[:,7][:-1] != matched_particles_with_halo[:,7][1:])[0]
indices_split = indices_split + 1 #so the values stored here are the starting values for each halo
halo_splitting_point = np.zeros((indices_split.size + 1), dtype=int)
halo_splitting_point[0] = 0
halo_splitting_point[1:] = indices_split

start_particle = 0
final_particle = 0
current_particle = 0
calculated_R200 = np.zeros(halo_splitting_point.size - 1)
count = 0
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
    current_halo_particles[0:new_particles,0] = matched_particles_with_halo[start_particle:final_particle,0] #add PIDs
    current_halo_particles[0:new_particles,1] = radius[start_particle:final_particle] #add how far the particle is from the halo
    current_halo_particles = current_halo_particles[~np.all(current_halo_particles == 0, axis=1)] #remove zeros
    current_halo_particles = current_halo_particles[current_halo_particles[:,1].argsort()] #sort the particles by distance
    

    #for each particle give how many particles there are (including it) within its radius then multiply by mass
    mass_values = np.zeros(new_particles)
    mass_values = mass * np.arange(1, new_particles + 1,1) 
    mass_values.reshape(new_particles,1)

    current_halo_particles[0:new_particles,2] = mass_values

    #density = mass of particles within radius/((4/3)*pi*particle radius^3)
    # print("mass: ")
    #print(current_halo_particles[0:new_particles,2])
    # print("volume: ")
    # print((4/3)*np.pi*(np.power(current_halo_particles[0:new_particles,1],3)))
    calculated_density = np.divide((current_halo_particles[0:new_particles,2]),((4/3)*np.pi*(np.power(current_halo_particles[0:new_particles,1],3))))
    indices_density_reached = np.where(calculated_density >= (200*rho_c))
    


    if(indices_density_reached[0].size > 1):
        array_two_r200[0] = current_halo_particles[indices_density_reached[0][0],1]
        array_two_r200[1] = current_halo_particles[indices_density_reached[0][1],1]
        calculated_R200[count] = np.mean(array_two_r200)
    count += 1
     
    
    if (index % 100000 == 0 and index != 0):
        print(index)
        t2 = time.time()
        print(t2 - t1)
    start_particle = final_particle
print("calculated R200")

zeros_R200 = np.where(calculated_R200 != 0)
no_zero_calculated_R200 = calculated_R200[zeros_R200]
print(np.max(no_zero_calculated_R200))
print(no_zero_calculated_R200)

print("actual R200")
actual_R200 = matched_particles_with_halo[:,8]
actual_R200 = actual_R200[indices_split]
no_zero_actual_R200 = actual_R200[zeros_R200]


print(actual_R200)

# #subtract velocitions
# matched_particles_vel[:,0] = matched_particles_vel[:,0] - matched_particles_with_halo[:,1]
# matched_particles_vel[:,1] = matched_particles_vel[:,1] - matched_particles_with_halo[:,2]
# matched_particles_vel[:,2] = matched_particles_vel[:,2] - matched_particles_with_halo[:,3]