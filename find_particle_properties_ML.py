import numpy as np
from pygadgetreader import readsnap, readheader
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from colossus.utils import constants
from colossus.halo import mass_so
from colossus.lss import peaks
from matplotlib.pyplot import cm
import time
import h5py
import os
import pickle

curr_snapshot = "190"
curr_hdf5_file = "sparta_190.hdf5"
hdf5_file = "/home/zvladimi/ML_orbit_infall_project/SPARTA_data/" + curr_hdf5_file
save_location =  "/home/zvladimi/ML_orbit_infall_project/calculated_info/" + curr_hdf5_file + "/"
snapshot_path = "/home/zvladimi/ML_orbit_infall_project/particle_data/snapshot_" + curr_snapshot + "/snapshot_0" + curr_snapshot

# get constants from pygadgetreader
snapshot_index = int(curr_snapshot) #set to what snapshot is being loaded in
red_shift = readheader(snapshot_path, 'redshift')
scale_factor = 1/(1+red_shift)
cosmol = cosmology.setCosmology("bolshoi")
rho_m = cosmol.rho_m(red_shift)
little_h = cosmol.h 
hubble_constant = cosmol.Hz(red_shift) * 0.001 # convert to units km/s/kpc
G = constants.G

box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
box_size = box_size * 10**3 * scale_factor * little_h #convert to Kpc physical

def check_pickle_exist_gadget(path, ptl_property, snapshot):
    file_path = path + ptl_property + "_" + snapshot + ".pickle" 
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            particle_info = pickle.load(pickle_file)
    else:
        particle_info = readsnap(snapshot_path, ptl_property, 'dm')
        with open(file_path, "wb") as pickle_file:
            pickle.dump(particle_info, pickle_file)
    return particle_info

def check_pickle_exist_hdf5_prop(path, first_group, second_group, third_group, hdf5_path):
    file_path = path + first_group + "_" + second_group + "_" + third_group + ".pickle" 
    if os.path.isfile(file_path):
        with open(file_path, "rb") as pickle_file:
            halo_info = pickle.load(pickle_file)
    else:
        print(hdf5_path)
        with h5py.File(hdf5_path, 'r') as file:
            if third_group != "":
                halo_info = file[first_group][second_group][third_group][:]
            else:
                halo_info = file[first_group][second_group][:]
        with open(file_path, "wb") as pickle_file:
            pickle.dump(halo_info, pickle_file)
    return halo_info

def load_or_pickle_data(path, snapshot, hdf5):
    if os.path.exists(path) != True:
        os.makedirs(path)
    ptl_pid = check_pickle_exist_gadget(path, "pid", snapshot)
    ptl_vel = check_pickle_exist_gadget(path, "vel", snapshot)
    ptl_pos = check_pickle_exist_gadget(path, "pos", snapshot)
    ptl_mass = check_pickle_exist_gadget(path, "mass", snapshot)
    
    halo_pos = check_pickle_exist_hdf5_prop(path, "halos", "position", "", hdf5)
    halo_vel = check_pickle_exist_hdf5_prop(path, "halos", "velocity", "", hdf5)
    halo_last_snap = check_pickle_exist_hdf5_prop(path, "halos", "last_snap", "", hdf5)
    halo_r200m = check_pickle_exist_hdf5_prop(path, "halos", "R200m", "", hdf5)
    halo_id = check_pickle_exist_hdf5_prop(path, "halos", "id", "", hdf5)
    halo_status = check_pickle_exist_hdf5_prop(path, "halos", "status", "", hdf5)
    
    density_prf_all = check_pickle_exist_hdf5_prop(path, "anl_prf", "M_all", "", hdf5)
    density_prf_1halo = check_pickle_exist_hdf5_prop(path, "anl_prf", "M_1halo", "", hdf5)
    
    num_pericenter = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "n_pericenter", hdf5)
    tracer_id = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "tracer_id", hdf5)
    n_is_lower_limit = check_pickle_exist_hdf5_prop(path, "tcr_ptl", "res_oct", "n_is_lower_limit", hdf5)

    return ptl_pid, ptl_vel, ptl_pos, ptl_mass, halo_pos, halo_vel, halo_last_snap, halo_r200m, halo_id, halo_status, num_pericenter, tracer_id, n_is_lower_limit, density_prf_all, density_prf_1halo

#load particle info
particles_pid, particles_vel, particles_pos, particles_mass, halos_pos, halos_vel, halos_last_snap, halos_r200m, halos_id, halos_status, num_pericenter, tracer_id, n_is_lower_limit, density_prf_all, density_prf_1halo = load_or_pickle_data(save_location, curr_snapshot, hdf5_file)

particles_pos = particles_pos * 10**3 * scale_factor * little_h #convert to kpc and physical
mass = particles_mass[0] * 10**10 * little_h #units M_sun

#load all halo info at snapshot
halos_pos = halos_pos[:,snapshot_index,:] * 10**3 * scale_factor * little_h #convert to kpc and physical
halos_vel = halos_vel[:,snapshot_index,:]
halos_r200m = halos_r200m[:,snapshot_index] * little_h # convert to kpc
halos_id = halos_id[:,snapshot_index]
halos_status = halos_status[:,snapshot_index]
density_prf_all = density_prf_all[:,snapshot_index,:]
density_prf_1halo = density_prf_1halo[:,snapshot_index,:]

num_particles = particles_pid.size

# remove all halos for any halo that doesn't exist beyond snapshot 
# remove all halos that aren't main halos (identified with tag = 10)
indices_keep = np.zeros((halos_id.size))
indices_keep = np.where((halos_last_snap >= snapshot_index) & (halos_status == 10))
halos_pos = halos_pos[indices_keep]
halos_vel = halos_vel[indices_keep]
halos_r200m = halos_r200m[indices_keep]
halos_id = halos_id[indices_keep]
density_prf_all = density_prf_all[indices_keep]
density_prf_1halo = density_prf_1halo[indices_keep]
total_num_halos = halos_r200m.size #num of halos remaining

# create array that tracks the ids and if a tracer is orbiting or infalling.
orbit_assn_tracers = np.zeros((tracer_id.size, 2), dtype = np.int32)
orbit_assn_tracers[:,0] = tracer_id
num_pericenter[num_pericenter > 0] = 1 # if there is more than one pericenter count as orbiting (1) and infalling is 0
orbit_assn_tracers[:,1] = num_pericenter

# However particle can also be orbiting if n_is_lower_limit is 1
indices_pass_n_low_lim = np.where(n_is_lower_limit == 1)
orbit_assn_tracers[indices_pass_n_low_lim,1] = 1

# create array that tracks the pids and if a particle is orbiting or infalling. (assume particle is infalling until proven otherwise)
orbit_assn_pids = np.zeros((particles_pid.size, 2), dtype = np.int32)
orbit_assn_pids[:,0] = particles_pid
match_ids = np.intersect1d(particles_pid, tracer_id, return_indices = True)[1:] # only take the matching indices for particle pids and orbit_assn
orbit_assn_pids[match_ids[0],1] = orbit_assn_tracers[match_ids[1],1] # assign the pids to the orbit/infall assignment of the matching tracers

# Create bins for the density profile calculation
num_prf_bins = density_prf_all.shape[1]
start_prf_bins = 0.01
end_prf_bins = 3.0
prf_bins = np.logspace(np.log10(start_prf_bins), np.log10(end_prf_bins), num_prf_bins)

#construct a search tree iwth all of the particle positions
particle_tree = cKDTree(data = particles_pos, leafsize = 3, balanced_tree = False, boxsize = box_size)

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
    
    coord_diff[:,0] = particle_x - halo_x
    
    y_within_plus = np.where((y_dist + box_size) < half_box_size)
    y_within_minus = np.where((y_dist - box_size) > -half_box_size)
    
    particle_y[y_within_plus] = particle_y[y_within_plus] + box_size
    particle_y[y_within_minus] = particle_y[y_within_minus] - box_size
    
    coord_diff[:,1] = particle_y - halo_y
    
    z_within_plus = np.where((z_dist + box_size) < half_box_size)
    z_within_minus = np.where((z_dist - box_size) > -half_box_size)
    
    particle_z[z_within_plus] = particle_z[z_within_plus] + box_size
    particle_z[z_within_minus] = particle_z[z_within_minus] - box_size
    
    coord_diff[:,2] = particle_z - halo_z

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

def calc_pec_vel(particle_vel, halo_vel):   
    # Peculiar velocity is particle velocity minus the corresponding halo velocity
    peculiar_velocities = particle_vel - halo_vel   

    return peculiar_velocities

def calc_rhat(x_comp, y_comp, z_comp):
    rhat = np.zeros((x_comp.size, 3), dtype = np.float32)
    # Get the magnitude for each particle
    magnitude = np.sqrt(np.square(x_comp) + np.square(y_comp) + np.square(z_comp))
    
    # Scale the components by the magnitude to get a unit vector
    rhat[:,0] = x_comp/magnitude
    rhat[:,1] = y_comp/magnitude
    rhat[:,2] = z_comp/magnitude
    
    return rhat

def calc_v200m(mass, radius):
    # calculate the v200m for a halo based on its mass and radius
    return np.sqrt((G * mass)/radius)

def calc_rad_vel(peculiar_vel, particle_dist, coord_sep, halo_r200):
    
    # Get the corresponding components, distances, and halo v200m for every particle
    v_hubble = np.zeros(particle_dist.size, dtype = np.float32)
    corresponding_hubble_m200m = mass_so.R_to_M(halo_r200, red_shift, "200c") * little_h # convert to M⊙
    curr_v200m = calc_v200m(corresponding_hubble_m200m, halo_r200)
        
    # calculate the unit vector of the halo to the particle  
    rhat = calc_rhat(coord_sep[:,0], coord_sep[:,1], coord_sep[:,2])
    
    # Hubble velocity is the hubble constant times the distance the particle is from the halo
    v_hubble = hubble_constant * particle_dist   
    
    v_hubble = rhat * v_hubble[:, np.newaxis] 
    
    physical_vel = peculiar_vel + v_hubble    

    radial_vel_comp = physical_vel * rhat
    radial_vel = np.sum(radial_vel_comp, axis = 1)
    
    # Dot the velocity with rhat to get the radial component
    #radial_component_vel = np.sum(np.multiply(peculiar_vel, rhat), axis = 1)
    
    # Add radial component and v_hubble since both are now in radial direction
    #radial_vel = radial_component_vel + v_hubble

    # scale all the radial velocities by v200m of the halo
    return radial_vel, radial_vel_comp, curr_v200m, physical_vel, rhat

def calc_tang_vel(radial_vel, physical_vel, rhat):
    component_rad_vel = rhat * radial_vel[:, np.newaxis] 
    tangential_vel = physical_vel - component_rad_vel
    
    return tangential_vel

def compare_density_prf(masses, radii, actual_prf):
    calculated_prf_all = np.zeros(num_prf_bins)
    
    for i in range(num_prf_bins - 1):
        start_bin = prf_bins[i]
        end_bin = prf_bins[i+1]  
        
        radii_within_range = np.where((radii >= start_bin) & (radii < end_bin))[0]
        if radii_within_range.size != 0:
            total_mass_within_bin = masses[radii_within_range[-1]]
            calculated_prf_all[i] = total_mass_within_bin
        else:
            calculated_prf_all[i] = 0
    for j in range(calculated_prf_all.size):
        print("calc:", calculated_prf_all[j], "act:", actual_prf[j])
    print(calculated_prf_all/mass)
    print(actual_prf/mass)
    print(np.allclose(calculated_prf_all, actual_prf, atol = 0.001))

def initial_search(halo_positions, search_radius, halo_r200m):
    num_halos = halo_positions.shape[0]
    particles_per_halo = np.zeros(num_halos, dtype = np.int32)
    all_halo_mass = np.zeros(num_halos, dtype = np.float32)
    
    for i in range(num_halos):
        #find how many particles we are finding
        indices = particle_tree.query_ball_point(halo_positions[i,:], r = search_radius * halo_r200m[i])

        # how many new particles being added and correspondingly how massive the halo is
        num_new_particles = len(indices)
        all_halo_mass[i] = num_new_particles * mass
        particles_per_halo[i] = num_new_particles

    print("Total num particles: ", np.sum(particles_per_halo))
    print("Total num halos: ", num_halos)
    
    return particles_per_halo, all_halo_mass

def search_halos(halo_positions, halo_r200m, search_radius, total_particles, dens_prf_all):
    num_halos = halo_positions.shape[0]
    halo_indices = np.zeros((num_halos,2), dtype = np.int32)
    all_part_vel = np.zeros((total_particles,3), dtype = np.float32)

    calculated_r200m = np.zeros(halo_r200m.size)
    calculated_radial_velocities = np.zeros((total_particles), dtype = np.float32)
    calculated_radial_velocities_comp = np.zeros((total_particles,3), dtype = np.float32)
    calculated_tangential_velocities_comp = np.zeros((total_particles,3), dtype = np.float32)
    all_radii = np.zeros((total_particles), dtype = np.float32)
    all_scaled_radii = np.zeros(total_particles, dtype = np.float32)
    r200m_per_part = np.zeros(total_particles, dtype = np.float32)
    all_orbit_asn = np.zeros((total_particles,2), dtype = np.int32)

    start = 0
    for i in range(num_halos):
        # find the indices of the particles within the expected r200 radius multiplied by times_r200 
        # value of 1.4 determined by guessing if just r200 value or 1.1 miss a couple halo r200 values but 1.4 gets them all
        indices = particle_tree.query_ball_point(halo_positions[i,:], r = search_radius * halo_r200m[i])

        #Only take the particle positions that where found with the tree
        current_particles_pos = particles_pos[indices,:]
        current_particles_vel = particles_vel[indices,:]
        current_orbit_assn = orbit_assn_pids[indices]
        current_halos_pos = halo_positions[i,:]        

        # how many new particles being added
        num_new_particles = len(indices)
        halo_indices[i,0] = start
        halo_indices[i,1] = start + num_new_particles

        #for how many new particles create an array of how much mass there should be within that particle radius
        use_mass = np.arange(1, num_new_particles + 1, 1) * mass       
        
        all_part_vel[start:start+num_new_particles] = current_particles_vel
            
        #calculate the radii of each particle based on the distance formula
        unsorted_particle_radii, unsorted_coord_dist = calculate_distance(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], current_particles_pos[:,0],
                                    current_particles_pos[:,1], current_particles_pos[:,2], num_new_particles)
        
            
        #sort the radii, positions, velocities, coord separations to allow for creation of plots and to correctly assign how much mass there is
        arrsortrad = unsorted_particle_radii.argsort()
        particle_radii = unsorted_particle_radii[arrsortrad]
        current_particles_pos = current_particles_pos[arrsortrad]
        current_particles_vel = current_particles_vel[arrsortrad]
        coord_dist = unsorted_coord_dist[arrsortrad]
        
        #calculate the density at each particle
        calculated_densities = np.zeros(num_new_particles)
        calculated_densities = calculate_density(use_mass, particle_radii)
        
        #determine indices of particles where the expected r200 value is 
        indices_r200_met = check_where_r200(calculated_densities)
        
        #if only one index is less than 200 * rho_c then that is the r200 radius
        if indices_r200_met[0].size == 1:
            calculated_r200m[i] = particle_radii[indices_r200_met[0][0]]
        #if there are none then the radius is 0
        elif indices_r200_met[0].size == 0:
            calculated_r200m[i] = 0
        #if multiple indices choose the first two and average them
        else:
            calculated_r200m[i] = (particle_radii[indices_r200_met[0][0]] + particle_radii[indices_r200_met[0][1]])/2
        
        #correspond_halo_prop[start:start+num_new_particles,1] = halo_v200
        
        peculiar_velocity = calc_pec_vel(current_particles_vel, halos_vel[i])
        calculated_radial_velocities[start:start+num_new_particles], calculated_radial_velocities_comp[start:start+num_new_particles], curr_v200m, physical_vel, rhat = calc_rad_vel(peculiar_velocity, particle_radii, coord_dist, halo_r200m[i])
        calculated_tangential_velocities_comp[start:start+num_new_particles] = calc_tang_vel(calculated_radial_velocities[start:start+num_new_particles], physical_vel, rhat)/curr_v200m
        calculated_radial_velocities[start:start+num_new_particles] = calculated_radial_velocities[start:start+num_new_particles]/curr_v200m
        calculated_radial_velocities_comp[start:start+num_new_particles] = calculated_radial_velocities_comp[start:start+num_new_particles]/curr_v200m
        all_radii[start:start+num_new_particles] = particle_radii
        all_scaled_radii[start:start+num_new_particles] = particle_radii/halo_r200m[i]
        r200m_per_part[start:start+num_new_particles] = halo_r200m[i]
        all_orbit_asn[start:start+num_new_particles] = current_orbit_assn
        
        if i == 1:
            compare_density_prf(use_mass, particle_radii/halo_r200m[i], dens_prf_all[i])
        
        start += num_new_particles
    
    return all_orbit_asn, calculated_radial_velocities, all_radii, all_scaled_radii, r200m_per_part, calculated_radial_velocities_comp, calculated_tangential_velocities_comp
    
def split_into_bins(num_bins, radial_vel, scaled_radii, particle_radii, halo_r200_per_part):
    start_bin_val = 0.001
    finish_bin_val = np.max(scaled_radii)
    
    bins = np.logspace(np.log10(start_bin_val), np.log10(finish_bin_val), num_bins)
    
    bin_start = 0
    average_val_part = np.zeros((num_bins,2), dtype = np.float32)
    average_val_hubble = np.zeros((num_bins,2), dtype = np.float32)
    
    # For each bin
    for i in range(num_bins - 1):
        bin_end = bins[i]
        
        # Find which particle belong in that bin
        indices_in_bin = np.where((scaled_radii >= bin_start) & (scaled_radii < bin_end))[0]
 
        if indices_in_bin.size != 0:
            # Get all the scaled radii within this bin and average it
            use_scaled_particle_radii = scaled_radii[indices_in_bin]
            average_val_part[i, 0] = np.average(use_scaled_particle_radii)
            
            # Get all the radial velocities within this bin and average it
            use_vel_rad = radial_vel[indices_in_bin]
            average_val_part[i, 1] = np.average(use_vel_rad)
            
            # get all the radii within this bin
            hubble_radius = particle_radii[indices_in_bin]

            # Find the median value and then the median value for the corresponding R200m values
            median_hubble_radius = np.median(hubble_radius)
            median_hubble_r200 = np.median(halo_r200_per_part[indices_in_bin])
            median_scaled_hubble = median_hubble_radius/median_hubble_r200
            
            # Calculate the v200m value for the corresponding R200m value found
            average_val_hubble[i,0] = median_scaled_hubble
            corresponding_hubble_m200m = mass_so.R_to_M(median_hubble_r200, red_shift, "200c") * little_h # convert to M⊙
            average_val_hubble[i,1] = (median_hubble_radius * hubble_constant)/calc_v200m(corresponding_hubble_m200m, median_hubble_r200)
            
        bin_start = bin_end
    
    return average_val_part, average_val_hubble    
    
def split_halo_by_mass(num_bins, start_nu, num_iter, times_r200m, halo_r200m, file):        
    color = iter(cm.rainbow(np.linspace(0, 1, num_iter)))
        
    print("\nstart initial search")    
    # get the halo_masses and number of particles
    num_particles_per_halo, halo_masses = initial_search(halos_pos, times_r200m, halo_r200m)
    print("finish initial search")
    total_num_particles = np.sum(num_particles_per_halo)
    # convert masses to peaks
    scaled_halo_mass = halo_masses/little_h # units M⊙/h
    peak_heights = peaks.peakHeight(scaled_halo_mass, red_shift)
    
    # Determine if we are creating a completely new file (file doesn't have any keys) or if we are accessing a different number of particles
    if len(file.keys()) < 3:
        new_file = True
    elif file["Scaled_radii"].shape[0] != total_num_particles:
        del file["Scaled_radii"]
        del file["Radial_vel"]
        del file["Tangential_vel"]
        new_file = True
    else:
        new_file = False
        
    # For how many graphs
    for j in range(num_iter):
        c = next(color)
        end_nu = start_nu + 0.5
        print("\nStart split:", start_nu, "to", end_nu)
        # Get the indices of the halos that are within the desired peaks
    
        halos_within_range = np.where((peak_heights >= start_nu) & (peak_heights < end_nu))[0]
        print("Num halos: ", halos_within_range.shape[0])
        if halos_within_range.shape[0] > 0:
            use_halo_pos = halos_pos[halos_within_range]
            use_halo_r200m = halo_r200m[halos_within_range]
            use_density_prf_all = density_prf_all[halos_within_range]
            use_density_prf_1halo = density_prf_1halo[halos_within_range]
            use_num_particles = num_particles_per_halo[halos_within_range]
            total_num_use_particles = np.sum(use_num_particles)
            
            print("Num particles: ", total_num_use_particles)

            orbital_assign, radial_velocities, radii, scaled_radii, r200m_per_part, radial_velocities_comp, tangential_velocities_comp = search_halos(use_halo_pos, use_halo_r200m, times_r200m, total_num_use_particles, use_density_prf_all)   

            # with a new file and just started create all the datasets
            if new_file and len(list(file.keys())) == 0:
                file.create_dataset("PIDS", data = orbital_assign[:,0], chunks = True, maxshape=(total_num_particles,))
                file.create_dataset("Orbit_Infall", data = orbital_assign[:,1], chunks = True, maxshape=(total_num_particles,))
                file.create_dataset("Scaled_radii", data = scaled_radii, chunks = True, maxshape=(total_num_particles,))
                file.create_dataset("Radial_vel", data = radial_velocities_comp, chunks = True, maxshape=(total_num_particles, 3))
                file.create_dataset("Tangential_vel", data = tangential_velocities_comp, chunks = True, maxshape=(total_num_particles, 3))
            # with a new file adding on additional data to the datasets
            elif new_file and len(list(file.keys())) != 0:
                file["PIDS"].resize((file["PIDS"].shape[0] + orbital_assign[:,0].shape[0]), axis = 0)
                file["PIDS"][-orbital_assign[:,0].shape[0]:] = orbital_assign[:,0]
                
                file["Orbit_Infall"].resize((file["Orbit_Infall"].shape[0] + orbital_assign[:,1].shape[0]), axis = 0)
                file["Orbit_Infall"][-orbital_assign[:,1].shape[0]:] = orbital_assign[:,1]
                
                file["Scaled_radii"].resize((file["Scaled_radii"].shape[0] + scaled_radii.shape[0]), axis = 0)
                file["Scaled_radii"][-scaled_radii.shape[0]:] = scaled_radii
                
                file["Radial_vel"].resize((file["Radial_vel"].shape[0] + radial_velocities_comp.shape[0]), axis = 0)
                file["Radial_vel"][-radial_velocities_comp.shape[0]:] = radial_velocities_comp
                
                file["Tangential_vel"].resize((file["Tangential_vel"].shape[0] + tangential_velocities_comp.shape[0]), axis = 0)
                file["Tangential_vel"][-tangential_velocities_comp.shape[0]:] = tangential_velocities_comp
                
            file_counter = 0
            # if not a new file and same num of particles will just replace the previous information
            if not new_file:
                file["Scaled_radii"][file_counter:file_counter + scaled_radii.shape[0]] = scaled_radii
                file["Radial_vel"][file_counter:file_counter + radial_velocities_comp.shape[0]] = radial_velocities_comp
                file["Tangential_vel"][file_counter:file_counter + tangential_velocities_comp.shape[0]] = tangential_velocities_comp
            
            file_counter = file_counter + scaled_radii.shape[0]
               
            graph_rad_vel, graph_val_hubble = split_into_bins(num_bins, radial_velocities, scaled_radii, radii, r200m_per_part)
            graph_rad_vel = graph_rad_vel[~np.all(graph_rad_vel == 0, axis=1)]
            graph_val_hubble = graph_val_hubble[~np.all(graph_val_hubble == 0, axis=1)]

            plt.plot(graph_rad_vel[:,0], graph_rad_vel[:,1], color = c, alpha = 0.7, label = r"${0} < \nu < {1}$".format(str(start_nu), str(end_nu)))
        start_nu = end_nu
        
    return graph_val_hubble     
    
num_bins = 50        
start_nu = 0
num_iter = 7
t1 = time.time()
print("start particle assign")

times_r200 = 14
with h5py.File((save_location + "all_particle_properties" + curr_snapshot + ".hdf5"), 'a') as all_particle_properties:
    hubble_vel = split_halo_by_mass(num_bins, start_nu, num_iter, times_r200, halos_r200m, all_particle_properties)    
    

arr1inds = hubble_vel[:,0].argsort()
hubble_vel[:,0] = hubble_vel[arr1inds,0]
hubble_vel[:,1] = hubble_vel[arr1inds,1]
plt.plot(hubble_vel[:,0], hubble_vel[:,1], color = "purple", alpha = 0.5, linestyle = "dashed", label = r"Hubble Flow")
plt.title("average radial velocity vs position all particles")
plt.xlabel("position $r/R_{200m}$")
plt.ylabel("average rad vel $v_r/v_{200m}$")
plt.xscale("log")    
plt.ylim([-.5,1])
plt.xlim([0.01,15])
plt.legend()
    
t2 = time.time()
print("finish binning: ", (t2- t1), " seconds")
plt.show()

