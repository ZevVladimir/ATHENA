import numpy as np
import time

t1 = time.time()

file = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"
save_location = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"

snapshot = 190

#load halo info
halo_n = np.load(file + "halo_n.npy")
print(halo_n.shape)
halo_position = np.load(file + "halo_position.npy")
print(halo_position.shape)
halo_velocity = np.load(file + "halo_velocity.npy")
print(halo_velocity.shape)

#load particle id
particle_id = np.load(file + "tracer_id.npy")
#particle_id, halo_pos_1, halo_pos_2, halo_pos_3, halo_velocity_1, halo_velocity_2, halo_velocity_3
particle_info = np.zeros((particle_id.size,1,1,1,1,1,1), dtype='float32')

particle_count_start = 0
# loop through all halos
for halo_count in range(halo_n.size):
    #determine the end and beginning particles and then add these particles to the array with the corresponding halo data
    particle_count_end = particle_count_start + halo_n[halo_count]
    np.append(particle_info,(particle_id[particle_count_start:particle_count_end], halo_position[halo_count, snapshot, 0],  halo_position[halo_count, snapshot, 1],
         halo_position[halo_count, snapshot, 2], halo_velocity[halo_count, snapshot, 0], halo_velocity[halo_count, snapshot, 1],
         halo_velocity[halo_count, snapshot, 2]))
    particle_count_start = particle_count_end

    #keep track of how long it takes to go through 100 halos and extrapolate for the entrie process
    if halo_count % 100 == 0:
        print(halo_count)
        t2 = time.time()
        t = t2 - t1
        t1 = t2
        print("time: %f" % t)
        batches = halo_n.size
        print("estimated " + str((((batches - halo_count) * (t/100))/60)) + " minutes reamining")

np.save(save_location + "particle_info", particle_info)
