import numpy as np
import time

file = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"
save_location = "/home/zeevvladimir/ML_orbit_infall_project/np_arrays/"

#load halo info
halo_n = np.load(file + "halo_n.npy")
halo_position = np.load(file + "halo_position.npy")
halo_velocity = np.load(file + "halo_velocity.npy")
halo_last_snap = np.load(file + "halo_last_snap.npy")
halo_r200 = np.load(file + "halo_R200m.npy")
#load particle info
particle_id = np.load(file + "tracer_id.npy")



snapshot = 190

# loop through all halos to find how many particles there are after 190
max_size = 0
for i in range(halo_n.size):
    if halo_last_snap[i] >= snapshot:
        max_size += halo_n[i]
particle_with_halo = np.zeros((max_size,9))


start_particle = 0
final_particle = 0
count = 0
difference = 0
#loop through halos 
t1 = time.time()
for halo in range(halo_n.size):
    final_particle += halo_n[halo]
    #if it is a halo we want
    if halo_last_snap[halo] >= snapshot:
        difference = final_particle - start_particle
        for particle in range(difference): 
            particle_with_halo[count][0] = particle_id[start_particle + particle] #making sure to skip any particles that belong to other halos
            particle_with_halo[count][1] = halo_position[halo][snapshot][0]
            particle_with_halo[count][2] = halo_position[halo][snapshot][1]
            particle_with_halo[count][3] = halo_position[halo][snapshot][2]
            particle_with_halo[count][4] = halo_velocity[halo][snapshot][0] 
            particle_with_halo[count][5] = halo_velocity[halo][snapshot][1]
            particle_with_halo[count][6] = halo_velocity[halo][snapshot][2]
            particle_with_halo[count][7] = halo_n[halo]
            particle_with_halo[count][8] = halo_r200[halo][snapshot]
            count += 1
    start_particle = final_particle
    if (halo % 5000 == 0 and halo != 0):
        print(halo)
        t2 = time.time()
        print(t2 - t1)
print(particle_with_halo)
np.save(save_location + "particle_with_halo", particle_with_halo)