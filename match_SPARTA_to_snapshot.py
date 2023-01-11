import numpy as np
import time
import matplotlib.pyplot as plt
import math

file = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"

particle_with_halo = np.load(file + "particle_with_halo.npy")

particles_pid = np.load(file + "particle_pid.npy")
particles_vel = np.load(file + "particle_vel.npy")
particles_pos = np.load(file + "particle_pos.npy")

#choose 10000 random positions to graph the cosmic web
# slice_particles_pos = particles_pos[particles_pos[:,2] < 1]
# x_column = slice_particles_pos[:,0]
# y_column = slice_particles_pos[:,1]


# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(x_column, y_column)

#plt.show()

match_indices = np.intersect1d(particle_with_halo[:,0], particles_pid, return_indices=True)
#match_indices: sorted pids, indices for matches in particle_with_halo, indices for matches in particles_pid
#matched_particles_with_halo layout: pid, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
matched_particles_with_halo = particle_with_halo[match_indices[1]]

#subtract halo position/velocity from particles
matched_particles_pos = particles_pos[match_indices[2]]
matched_particles_vel = particles_vel[match_indices[2]]

#subtract positions
matched_particles_pos[:,0] = matched_particles_pos[:,0] - matched_particles_with_halo[:,1]  
matched_particles_pos[:,1] = matched_particles_pos[:,1] - matched_particles_with_halo[:,2]  
matched_particles_pos[:,2] = matched_particles_pos[:,2] - matched_particles_with_halo[:,3]  

#subtract velocitions
matched_particles_vel[:,0] = matched_particles_vel[:,0] - matched_particles_with_halo[:,1]  
matched_particles_vel[:,1] = matched_particles_vel[:,1] - matched_particles_with_halo[:,2]  
matched_particles_vel[:,2] = matched_particles_vel[:,2] - matched_particles_with_halo[:,3]   

radius = np.zeros((matched_particles_with_halo[1].size,1))

radius = np.square((matched_particles_with_halo[:,1] - matched_particles_pos[:,0])**2 + (matched_particles_with_halo[:,2] - matched_particles_pos[:,1])**2 + 
(matched_particles_with_halo[:,3] - matched_particles_pos[:,2])**2)
print(radius)