import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pygadgetreader import *


file = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zeevvladimir/ML_orbit_infall_project/Original_Data/snapshot_190/snapshot_0190"

#Shape particle_with_halo pid, halo_pos_x, halo_pos_y, halo_pos_z, halo_vel_x, halo_vel_y, halo_vel_z, halo_r200
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
matched_particles_with_halo = particle_with_halo[match_indices[1]]

#subtract halo position/velocity from particles
matched_particles_pos = particles_pos[match_indices[2]]
matched_particles_vel = particles_vel[match_indices[2]] 

#subtract positions
box_size = readheader(snapshot_path, 'boxsize')

#calculate distance of particle from halo
distance = np.zeros((matched_particles_with_halo[1].size,1))
distance = np.sqrt((matched_particles_with_halo[:,1] - matched_particles_pos[:,0])**2 + (matched_particles_with_halo[:,2] - matched_particles_pos[:,1])**2 + 
(matched_particles_with_halo[:,3] - matched_particles_pos[:,2])**2)

mask_larger = distance > box_size/2
distance[mask_larger] = distance[mask_larger] - box_size
dinstance_div_r200 = distance/matched_particles_with_halo[:,7]
print(dinstance_div_r200)



# #subtract velocitions
# matched_particles_vel[:,0] = matched_particles_vel[:,0] - matched_particles_with_halo[:,1]  
# matched_particles_vel[:,1] = matched_particles_vel[:,1] - matched_particles_with_halo[:,2]  
# matched_particles_vel[:,2] = matched_particles_vel[:,2] - matched_particles_with_halo[:,3]   
