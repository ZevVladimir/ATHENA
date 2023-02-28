import numpy as np
import time
from pygadgetreader import *

file = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"
save_location = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zeevvladimir/ML_orbit_infall_project/Original_Data/snapshot_190/snapshot_0190"

snapshot = 190
red_shift = readheader(snapshot_path, 'redshift')
scale_factor = 1/(1+red_shift)

box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
box_size = box_size * 10**3 * scale_factor #convert to Kpc/h physical

#load particle info
particles_pid = np.load(file + "particle_pid.npy")
particles_vel = np.load(file + "particle_vel.npy")
particles_pos = np.load(file + "particle_pos.npy") 
particles_pos = particles_pos * 10**3 * scale_factor
particles_mass = np.load(file + "all_particle_mass.npy")
mass = particles_mass[0] * 10**10 #units M_sun/h

#load halo info
halo_position = np.load(file + "halo_position.npy")
halo_position = halo_position[:,snapshot,:] * 10**3 * scale_factor
halo_velocity = np.load(file + "halo_velocity.npy")
halo_velocity = halo_velocity[:,snapshot,:]
halo_last_snap = np.load(file + "halo_last_snap.npy")
halo_r200 = np.load(file + "halo_R200m.npy")
halo_r200 = halo_r200[:,snapshot]


num_particles = particles_pid.size
num_halos = halo_last_snap.size


#particle_with_halo:
#CONTAINS: PID, particle_pos_x, particle_pos_y, particle_pos_z, total mass to this point, particle_vel_x, particle_vel_y, particle_vel_z,
#halo_pos_x, halo_pos_y, halo_pos_z, halo_vel_x, halo_vel_y, halo_vel_z, halo_r200
#SIZE: (256^3,15)
particle_with_halo = np.zeros((num_particles,15))

#calculate distance of particle from halo
def calculate_distance(halo_x, halo_y, halo_z, particle_x, particle_y, particle_z):
    x_dist = particle_x - halo_x
    y_dist = particle_y - halo_y
    z_dist = particle_z - halo_z
    half_box_size = box_size/2
    #make this a np.where? instead want to get indices not values idk...
    x_within_plus = np.zeros(x_dist.size)
    x_within_minus = np.zeros(x_dist.size)
    x_within = np.zeros(x_dist.size)
    x_within = x_dist[x_dist < half_box_size]
    x_within_plus = x_dist[(x_dist + box_size) < half_box_size]
    x_within_minus = x_dist[(x_dist + box_size) < -half_box_size]
    
    x_pos = np.zeros((x_within.size + x_within_plus.size + x_within_minus.size))
    x_pos[:x_within.size] = x_within
    x_pos[x_within.size : x_within.size + x_within_plus.size] = x_within_plus
    x_pos[x_within.size + x_within_plus.size: x_within.size + x_within_plus.size + x_within_minus.size] = x_within_minus

    y_within_plus = np.zeros(x_dist.size)
    y_within_minus = np.zeros(x_dist.size)
    y_within = np.zeros(x_dist.size)
    y_within = y_dist[y_dist < half_box_size]
    y_within_plus = y_dist[(y_dist + box_size) < half_box_size]
    y_within_minus = y_dist[(y_dist + box_size) < -half_box_size]
    
    y_pos = np.zeros((y_within.size + y_within_plus.size + y_within_minus.size))
    y_pos[:y_within.size] = y_within
    y_pos[y_within.size : y_within.size + y_within_plus.size] = y_within_plus
    y_pos[y_within.size + y_within_plus.size: y_within.size + y_within_plus.size + y_within_minus.size] = y_within_minus

    z_within_plus = np.zeros(x_dist.size)
    z_within_minus = np.zeros(x_dist.size)
    z_within = np.zeros(x_dist.size)
    z_within_plus = z_dist[(z_dist + box_size) < half_box_size]
    z_within_minus = z_dist[(z_dist + box_size) < -half_box_size]
    z_within = z_dist[z_dist < half_box_size]
    
    z_pos = np.zeros((z_within.size + z_within_plus.size + z_within_minus.size))
    z_pos[:z_within.size] = z_within
    z_pos[z_within.size : z_within.size + z_within_plus.size] = z_within_plus
    z_pos[z_within.size + z_within_plus.size: z_within.size + z_within_plus.size + z_within_minus.size] = z_within_minus


    x_y_satis = np.intersect1d(x_pos, y_pos)
    print(x_y_satis)
    all_satis = np.intersect1d(x_y_satis, z_pos)
    print(all_satis)
    #Conditions to check if there are particles in the wrong place due to periodic boundary conditions
    # for i in range(particle_x.size):
    # #if the distance (x,y,z) goes over the boundary to the left add the box_size
    # #if the distance (x,y,z) goes over the boundary to the right subtract the box_size
    #     if halo_x - particle_x[i] > box_size/2:
    #         particle_x[i] = particle_x[i] + box_size
    #     elif halo_x - particle_x[i] < -box_size/2:
    #         particle_x[i] = particle_x[i] - box_size
        
    #     if halo_y - particle_y[i] > box_size/2:
    #         particle_y[i] = particle_y[i] + box_size
    #     elif halo_y - particle_y[i] < -box_size/2:
    #         particle_y[i] = particle_y[i] - box_size

    #     if halo_z - particle_z[i] > box_size/2:
    #         particle_z[i] = particle_z[i] + box_size
    #     elif halo_z - particle_z[i] < -box_size/2:
    #         particle_z[i] = particle_z[i] - box_size

    #calculate the actual distance
    distance = np.zeros((num_particles,1))
    distance = np.sqrt(np.square((halo_x - particle_x)) + np.square((halo_y - particle_y)) + np.square((halo_z - particle_z)))
    
    return distance 


start_particle = 0
final_particle = 0
t1 = time.time()
for halo in range(num_halos):
    if(halo_last_snap[halo] >= 190):
        distance_from_halo = calculate_distance(halo_position[halo,0], halo_position[halo,1],halo_position[halo,2], 
                                                particles_pos[:,0], particles_pos[:,1], particles_pos[:,2])
        mask_within_r200 = np.where(distance_from_halo <= halo_r200[halo])
        print(mask_within_r200[0].size)
        print(np.sort(distance_from_halo))
        print(halo_r200[halo])
        
        new_particles = mask_within_r200[0].size
        final_particle = final_particle + new_particles

        use_particle_pid = np.zeros(new_particles)
        use_particle_pos = np.zeros((new_particles,3))
        use_particle_vel = np.zeros((new_particles,3))

        use_particle_pid = particles_pid[mask_within_r200]
        use_particle_pos = particles_pos[mask_within_r200]
        use_particle_vel = particles_vel[mask_within_r200]
        use_mass = np.arange(1, new_particles+1, 1) * mass

        use_halo_pos = halo_position[halo]
        use_halo_vel = halo_velocity[halo]
        use_halo_r200 = halo_r200[halo]


        particle_with_halo[start_particle:final_particle, 0] = use_particle_pid
        particle_with_halo[start_particle:final_particle, 1] = use_particle_pos[:,0]
        particle_with_halo[start_particle:final_particle, 2] = use_particle_pos[:,1]
        particle_with_halo[start_particle:final_particle, 3] = use_particle_pos[:,2]
        particle_with_halo[start_particle:final_particle, 4] = use_mass
        particle_with_halo[start_particle:final_particle, 5] = use_particle_vel[:,0]
        particle_with_halo[start_particle:final_particle, 6] = use_particle_vel[:,1]
        particle_with_halo[start_particle:final_particle, 7] = use_particle_vel[:,2]
        particle_with_halo[start_particle:final_particle, 8] = use_halo_pos[0]
        particle_with_halo[start_particle:final_particle, 9] = use_halo_pos[1]
        particle_with_halo[start_particle:final_particle, 10] = use_halo_pos[2]
        particle_with_halo[start_particle:final_particle, 11] = use_halo_vel[0]
        particle_with_halo[start_particle:final_particle, 12] = use_halo_vel[1]
        particle_with_halo[start_particle:final_particle, 13] = use_halo_vel[2]
        particle_with_halo[start_particle:final_particle, 14] = use_halo_r200


        start_particle = final_particle
    if (halo % 5000 == 0 and halo != 0):
        print(halo)
        t2 = time.time()
        print(t2 - t1)

start_particle = 0
final_particle = 0
count = 0
difference = 0
#loop through halos 

# for halo in range(halo_n.size):
#     final_particle += halo_n[halo]
#     #if it is a halo we want
#     if halo_last_snap[halo] >= snapshot:
#         difference = final_particle - start_particle
#         for particle in range(difference): 
#             particle_with_halo[count][0] = particles_pid[start_particle + particle] #making sure to skip any particles that belong to other halos
#             particle_with_halo[count][1] = halo_position[halo][snapshot][0]
#             particle_with_halo[count][2] = halo_position[halo][snapshot][1]
#             particle_with_halo[count][3] = halo_position[halo][snapshot][2]
#             particle_with_halo[count][4] = halo_velocity[halo][snapshot][0] 
#             particle_with_halo[count][5] = halo_velocity[halo][snapshot][1]
#             particle_with_halo[count][6] = halo_velocity[halo][snapshot][2]
#             particle_with_halo[count][7] = halo_n[halo]
#             particle_with_halo[count][8] = halo_r200[halo][snapshot]
#             count += 1
#     start_particle = final_particle
#     if (halo % 5000 == 0 and halo != 0):
#         print(halo)
#         t2 = time.time()
#         print(t2 - t1)


print(particle_with_halo)
np.save(save_location + "particle_with_halo", particle_with_halo)