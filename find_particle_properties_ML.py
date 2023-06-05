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

file = "/home/zvladimi/ML_orbit_infall_project/np_arrays/"
save_location =  "/home/zvladimi/ML_orbit_infall_project/np_arrays/"
snapshot_path = "/home/zvladimi/ML_orbit_infall_project/particle_data/snapshot_192/snapshot_0192"

# get constants from pygadgetreader
snapshot = 192 #set to what snapshot is being loaded in
red_shift = readheader(snapshot_path, 'redshift')
scale_factor = 1/(1+red_shift)
cosmol = cosmology.setCosmology("bolshoi")
rho_m = cosmol.rho_m(red_shift)
little_h = cosmol.h 
hubble_constant = cosmol.Hz(red_shift) * 0.001 # convert to units km/s/kpc

G = constants.G

box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
box_size = box_size * 10**3 * scale_factor * little_h #convert to Kpc physical

#load particle info
particles_pid = np.load(file + "particle_pid_192.npy")
particles_vel = np.load(file + "particle_vel_192.npy")
particles_pos = np.load(file + "particle_pos_192.npy") 
particles_pos = particles_pos * 10**3 * scale_factor * little_h #convert to kpc and physical
particles_mass = np.load(file + "all_particle_mass_192.npy")
mass = particles_mass[0] * 10**10 * little_h #units M_sun

#load all halo info at snapshot
halos_pos = np.load(file + "halo_position.npy")
halos_pos = halos_pos[:,snapshot,:] * 10**3 * scale_factor * little_h #convert to kpc and physical
halos_vel = np.load(file + "halo_velocity.npy")
halos_vel = halos_vel[:,snapshot,:]
halos_last_snap = np.load(file + "halo_last_snap.npy")
halos_r200 = np.load(file + "halo_R200m.npy")
halos_r200 = halos_r200[:,snapshot] * little_h # convert to kpc
halos_id = np.load(file + "halo_id.npy")
halos_id = halos_id[:,snapshot]
halos_status = np.load(file + "halo_status.npy")
halos_status = halos_status[:,snapshot]

num_particles = particles_pid.size

# remove all halos for any halo that doesn't exist beyond snapshot 
# remove all halos that aren't main halos (identified with tag = 10)
indices_keep = np.zeros((halos_id.size))
indices_keep = np.where((halos_last_snap >= snapshot) & (halos_status == 10))
halos_pos = halos_pos[indices_keep]
halos_vel = halos_vel[indices_keep]
halos_r200 = halos_r200[indices_keep]
halos_id = halos_id[indices_keep]
total_num_halos = halos_r200.size #num of halos remaining

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

def search_halos(halo_positions, halo_r200m, search_radius, total_particles):
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

    start = 0
    for i in range(num_halos):
        # find the indices of the particles within the expected r200 radius multiplied by times_r200 
        # value of 1.4 determined by guessing if just r200 value or 1.1 miss a couple halo r200 values but 1.4 gets them all
        indices = particle_tree.query_ball_point(halo_positions[i,:], r = search_radius * halo_r200m[i])

        # how many new particles being added
        num_new_particles = len(indices)
        halo_indices[i,0] = start
        halo_indices[i,1] = start + num_new_particles

        #for how many new particles create an array of how much mass there should be within that particle radius
        use_mass = np.arange(1, num_new_particles + 1, 1) * mass

        #Only take the particle positions that where found with the tree
        current_particles_pos = particles_pos[indices,:]
        current_particles_vel = particles_vel[indices,:]
        current_halos_pos = halo_positions[i,:]
        
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
        
        start += num_new_particles
        
    return calculated_radial_velocities, all_radii, all_scaled_radii, r200m_per_part, calculated_radial_velocities_comp, calculated_tangential_velocities_comp
    
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
    
def split_halo_by_mass(num_bins, start_nu, num_iter, times_r200m, halo_r200m):
    color = iter(cm.rainbow(np.linspace(0, 1, num_iter)))
        
    # get the halo_masses and number of particles
    num_particles_per_halo, halo_masses = initial_search(halos_pos, times_r200m, halo_r200m)
    
    # convert masses to peaks
    scaled_halo_mass = halo_masses/little_h # units M⊙/h
    peak_heights = peaks.peakHeight(scaled_halo_mass, red_shift)
    
    # For how many graphs
    for j in range(num_iter):
        c = next(color)
        end_nu = start_nu + 0.5
        print("Start split:", start_nu, "to", end_nu)
        # Get the indices of the halos that are within the desired peaks
    
        halos_within_range = np.where((peak_heights >= start_nu) & (peak_heights < end_nu))[0]
        print("Num halos: ", halos_within_range.shape[0])
        if halos_within_range.shape[0] > 0:
            use_halo_pos = halos_pos[halos_within_range]
            use_halo_r200m = halo_r200m[halos_within_range]
            use_num_particles = num_particles_per_halo[halos_within_range]
            total_num_particles = np.sum(use_num_particles)
            
            print("Num particles: ", total_num_particles)
            all_part_prop = np.zeros((total_num_particles,7), dtype = np.float32)

            radial_velocities, radii, scaled_radii, r200m_per_part, radial_velocities_comp, tangential_velocities_comp = search_halos(use_halo_pos, use_halo_r200m, times_r200m, total_num_particles)
            all_part_prop[:, 0] = scaled_radii
            all_part_prop[:, 1:4] = radial_velocities_comp
            all_part_prop[:, 4:] = tangential_velocities_comp
            
            np.save(save_location + "all_part_prop_" + str(start_nu) + "_to_" + str(end_nu), all_part_prop)
            
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
hubble_vel, all_part_prop = split_halo_by_mass(num_bins, start_nu, num_iter, times_r200, halos_r200)    
    

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

