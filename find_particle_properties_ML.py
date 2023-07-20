import numexpr as ne
import numpy as np
from pairing import depair
from pygadgetreader import readsnap, readheader
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from colossus.halo import mass_so
from colossus.lss import peaks
from matplotlib.pyplot import cm
import time
import h5py
from data_and_loading_functions import load_or_pickle_SPARTA_data, load_or_pickle_ptl_data, save_to_hdf5, find_closest_snap
from calculation_functions import *
from visualization_functions import compare_density_prf, rad_vel_vs_radius_plot
from sparta import sparta

def initial_search(halo_positions, search_radius, halo_r200m, tree):
    num_halos = halo_positions.shape[0]
    particles_per_halo = np.zeros(num_halos, dtype = np.int32)
    all_halo_mass = np.zeros(num_halos, dtype = np.float32)
    
    for i in range(num_halos):
        #find how many particles we are finding
        indices = tree.query_ball_point(halo_positions[i,:], r = search_radius * halo_r200m[i])

        # how many new particles being added and correspondingly how massive the halo is
        num_new_particles = len(indices)
        all_halo_mass[i] = num_new_particles * mass
        particles_per_halo[i] = num_new_particles

    print("Total num particles: ", np.sum(particles_per_halo))
    print("Total num halos: ", num_halos)
    
    return particles_per_halo, all_halo_mass

def search_halos(particle_tree, halo_idxs, halo_positions, halo_r200m, search_radius, total_particles, dens_prf_all, dens_prf_1halo, start_nu, end_nu, curr_halo_id, sparta_file_path, snapshot_list, comp_snap, box_size, particles_pos, particles_vel, particles_pid, rho_m, halos_vel, red_shift, hubble_constant):
    p_snap = snapshot_list[0]
    global halo_start_idx
    num_halos = halo_positions.shape[0]
    halo_indices = np.zeros((num_halos,2), dtype = np.int32)
    all_part_vel = np.zeros((total_particles,3), dtype = np.float32)

    calculated_r200m = np.zeros(halo_r200m.size)
    calculated_radial_velocities = np.zeros((total_particles), dtype = np.float32)
    calculated_radial_velocities_comp = np.zeros((total_particles,3), dtype = np.float32)
    calculated_tangential_velocities_comp = np.zeros((total_particles,3), dtype = np.float32)
    calculated_tangential_velocities  = np.zeros((total_particles), dtype = np.float32)
    all_radii = np.zeros((total_particles), dtype = np.float32)
    all_scaled_radii = np.zeros(total_particles, dtype = np.float32)
    r200m_per_part = np.zeros(total_particles, dtype = np.float32)
    all_orbit_assn = np.zeros((total_particles,2), dtype = np.int64)

    start = 0
    t_start = time.time()
    for i in range(num_halos):
        # find the indices of the particles within the expected r200 radius multiplied by times_r200 
        indices = particle_tree.query_ball_point(halo_positions[i,:], r = search_radius * halo_r200m[i])
        
        #Only take the particle positions that where found with the tree
        current_particles_pos = particles_pos[indices,:]
        current_particles_vel = particles_vel[indices,:]
        current_particles_pid = particles_pid[indices]
        current_particles_pid = current_particles_pid.astype(np.int64)
        # current_orbit_assn_sparta: halo_idx, PID, orb/inf
        current_orbit_assn_sparta = np.zeros((current_particles_pid.size, 2), dtype = np.int64)
        current_halos_pos = halo_positions[i,:]        

        curr_halo_idx = halo_idxs[i]
        current_orbit_assn_sparta[:,0] = ne.evaluate("0.5 * (current_particles_pid + curr_halo_idx) * (current_particles_pid + curr_halo_idx + 1) + curr_halo_idx")

        # how many new particles being added
        num_new_particles = len(indices)
        halo_indices[i,0] = halo_start_idx
        halo_indices[i,1] = num_new_particles
        halo_start_idx = halo_start_idx + num_new_particles

        #for how many new particles create an array of how much mass there should be within that particle radius
        use_mass = np.arange(1, num_new_particles + 1, 1) * mass       
        all_part_vel[start:start+num_new_particles] = current_particles_vel
            
        #calculate the radii of each particle based on the distance formula
        unsorted_particle_radii, unsorted_coord_dist = calculate_distance(current_halos_pos[0], current_halos_pos[1], current_halos_pos[2], current_particles_pos[:,0],
                                    current_particles_pos[:,1], current_particles_pos[:,2], num_new_particles, box_size)         
              
        if comp_snap == False:
            sparta_output = sparta.load(filename = sparta_file_path, halo_ids = curr_halo_id[i], log_level = 0)['tcr_ptl']['res_oct'] # last_pericenter_snap, n_is_lower_limit, n_pericenter, tracer_id
            sparta_last_pericenter_snap = sparta_output['last_pericenter_snap']
            sparta_tracer_ids = sparta_output['tracer_id']
            sparta_n_is_lower_limit = sparta_output['n_is_lower_limit']
            sparta_n_pericenter = sparta_output['n_pericenter']
            
            orbit_assn_sparta = np.zeros((sparta_tracer_ids.size, 2), dtype = np.int64)
            orbit_assn_sparta[:,0] = sparta_tracer_ids
            
            # only want pericenters that have occurred at or before this snapshot
            sparta_n_pericenter[np.where(sparta_last_pericenter_snap > p_snap)[0]] = 0

            # if there is more than one pericenter count as orbiting (1) and if it isn't it is infalling (0)
            orbit_assn_sparta[np.where(sparta_n_pericenter > 0)[0],1] = 1

            # However particle can also be orbiting if n_is_lower_limit is 1
            orbit_assn_sparta[np.where(sparta_n_is_lower_limit == 1)[0],1] = 1

            poss_pids = np.intersect1d(current_particles_pid, orbit_assn_sparta[:,0], return_indices = True) # only check pids that are within the tracers for this halo (otherwise infall)
            poss_pid_match = np.intersect1d(current_particles_pid[poss_pids[1]], orbit_assn_sparta[:,0], return_indices = True) # get the corresponding indices for the pids and their infall/orbit assn
            # create a mask to then set any particle that is not identified as orbiting to be infalling
            current_orbit_assn_sparta[poss_pids[1],1] = orbit_assn_sparta[poss_pid_match[2],1]
            mask = np.ones(current_particles_pid.size, dtype = bool) 
            mask[poss_pids[1]] = False
            current_orbit_assn_sparta[mask,1] = 0 # set every pid that didn't have a match to infalling  

        #sort the radii, positions, velocities, coord separations to allow for creation of plots and to correctly assign how much mass there is
        arrsortrad = unsorted_particle_radii.argsort()
        particle_radii = unsorted_particle_radii[arrsortrad]
        current_particles_pos = current_particles_pos[arrsortrad]
        current_particles_vel = current_particles_vel[arrsortrad]
        #current_orbit_assn = current_orbit_assn[arrsortrad]
        current_orbit_assn_sparta = current_orbit_assn_sparta[arrsortrad]
        coord_dist = unsorted_coord_dist[arrsortrad]
        
        #calculate the density at each particle
        calculated_densities = np.zeros(num_new_particles)
        calculated_densities = calculate_density(use_mass, particle_radii)
        
        #determine indices of particles where the expected r200 value is 
        indices_r200_met = check_where_r200(calculated_densities, rho_m)
        
        #if only one index is less than 200 * rho_c then that is the r200 radius
        if indices_r200_met[0].size == 1:
            calculated_r200m[i] = particle_radii[indices_r200_met[0][0]]
        #if there are none then the radius is 0
        elif indices_r200_met[0].size == 0:
            calculated_r200m[i] = 0
        #if multiple indices choose the first two and average them
        else:
            calculated_r200m[i] = (particle_radii[indices_r200_met[0][0]] + particle_radii[indices_r200_met[0][1]])/2
        
        
        # calculate peculiar, radial, and tangential velocity
        peculiar_velocity = calc_pec_vel(current_particles_vel, halos_vel[i])
        calculated_radial_velocities[start:start+num_new_particles], calculated_radial_velocities_comp[start:start+num_new_particles], curr_v200m, physical_vel, rhat = calc_rad_vel(peculiar_velocity, particle_radii, coord_dist, halo_r200m[i], red_shift, little_h, hubble_constant)
        calculated_tangential_velocities_comp[start:start+num_new_particles] = calc_tang_vel(calculated_radial_velocities[start:start+num_new_particles], physical_vel, rhat)/curr_v200m
        calculated_tangential_velocities[start:start+num_new_particles] = np.linalg.norm(calculated_tangential_velocities_comp[start:start+num_new_particles], axis = 1)

        # scale radial velocities and their components by V200m
        # scale radii by R200m
        # assign all values to portion of entire array for this halo mass bin
        calculated_radial_velocities[start:start+num_new_particles] = calculated_radial_velocities[start:start+num_new_particles]/curr_v200m
        calculated_radial_velocities_comp[start:start+num_new_particles] = calculated_radial_velocities_comp[start:start+num_new_particles]/curr_v200m
        all_radii[start:start+num_new_particles] = particle_radii
        all_scaled_radii[start:start+num_new_particles] = particle_radii/halo_r200m[i]
        r200m_per_part[start:start+num_new_particles] = halo_r200m[i]
        all_orbit_assn[start:start+num_new_particles] = current_orbit_assn_sparta
            
        if i == 50 or i == 100 or i == 113:
            compare_density_prf(particle_radii/halo_r200m[i], dens_prf_all[i], dens_prf_1halo[i], mass, current_orbit_assn_sparta[:,1], i, start_nu, end_nu)

        start += num_new_particles

        if i % 250 == 0 and i != 0:
            t_lap = time.time()
            tot_time = t_lap - t_start
            t_remain = (num_halos - i)/250 * tot_time
            print("Halos:", (i-250), "to", i, "time taken:", np.round(tot_time,2), "seconds" , "time remaining:", np.round(t_remain/60,2), "minutes,",  np.round((t_remain),2), "seconds")
            t_start = t_lap
            
    return all_orbit_assn, calculated_radial_velocities, all_scaled_radii, calculated_tangential_velocities
    
def split_into_bins(num_bins, radial_vel, scaled_radii, particle_radii, halo_r200_per_part, red_shift, hubble_constant):
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
    
def split_halo_by_mass(num_bins, num_ptl_params, num_halo_params, start_nu, num_iter, nu_step, times_r200m, p_halo_r200m, c_halo_r200m, sparta_file_path, sparta_file_name, snapshot_list, new_file, all_halo_idxs):
    p_snap = snapshot_list[0] 
    c_snap = snapshot_list[1]
    # with h5py.File((save_location + "all_halo_properties" + sparta_file_name + ".hdf5"), 'a') as all_halo_properties:
    #     if new_file:
    #         for key in all_halo_properties.keys():
    #             del all_halo_properties[key]
    #         new_file_halo = True
    #     else:
    #         new_file_halo = False

    # with h5py.File((save_location + "all_particle_properties" + sparta_file_name + ".hdf5"), 'a') as all_particle_properties:
    #     # Determine if we are creating a completely new file (file doesn't have any keys) or if we are accessing a different number of particles
    #     if new_file:
    #         for key in all_particle_properties.keys():
    #             del all_particle_properties[key]
    #         new_file_ptl = True
    #     else:
    #         new_file_ptl = False

    color = iter(cm.rainbow(np.linspace(0, 1, num_iter)))
    
    print("\nstart initial search")    
    # get the halo_masses and number of particles
    p_num_particles_per_halo, p_halo_masses = initial_search(p_halos_pos, times_r200m, p_halo_r200m, p_particle_tree)
    c_num_particles_per_halo, c_halo_masses = initial_search(c_halos_pos, times_r200m, c_halo_r200m, c_particle_tree)
    print("finish initial search")
    p_total_num_particles = np.sum(p_num_particles_per_halo)
    # convert masses to peaks
    p_scaled_halo_mass = p_halo_masses/little_h # units M⊙/h
    peak_heights = peaks.peakHeight(p_scaled_halo_mass, p_red_shift)

    file_counter = 0

    # For how many mass bin splits
    for j in range(num_iter):
        t_start = time.time()
        c = next(color)
        end_nu = start_nu + nu_step
        print("\nStart split:", start_nu, "to", end_nu)
        
        # Get the indices of the halos that are within the desired peaks as well as ones that are 
        halos_within_range = np.where((peak_heights >= start_nu) & (peak_heights < end_nu))[0]
        print("Num halos: ", halos_within_range.shape[0])
        
        # only get the parameters we want for this specific halo bin
        if halos_within_range.shape[0] > 0:
            use_halo_idxs = all_halo_idxs[halos_within_range]
            p_use_halo_pos = p_halos_pos[halos_within_range]
            p_use_halo_r200m = p_halo_r200m[halos_within_range]
            p_use_density_prf_all = p_density_prf_all[halos_within_range]
            p_use_density_prf_1halo = p_density_prf_1halo[halos_within_range]
            p_use_num_particles = p_num_particles_per_halo[halos_within_range]
            p_use_halo_id = p_halos_id[halos_within_range]

            c_use_halo_pos = c_halos_pos[halos_within_range]
            c_use_halo_r200m = c_halo_r200m[halos_within_range]
            c_use_density_prf_all = c_density_prf_all[halos_within_range]
            c_use_density_prf_1halo = c_density_prf_1halo[halos_within_range]
            c_use_num_particles = c_num_particles_per_halo[halos_within_range]
            c_use_halo_id = c_halos_id[halos_within_range]

            p_total_num_use_particles = np.sum(p_use_num_particles)
            c_total_num_use_particles = np.sum(c_use_num_particles)
            
            print("Num particles primary: ", p_total_num_use_particles)
            print("Num particles compare: ", c_total_num_use_particles)

            # calculate all the information
            p_orbital_assign, p_rad_vel, p_scaled_radii, p_tang_vel = search_halos(p_particle_tree, use_halo_idxs, p_use_halo_pos, p_use_halo_r200m, times_r200m, p_total_num_use_particles, p_use_density_prf_all, p_use_density_prf_1halo, start_nu, end_nu, p_use_halo_id, sparta_file_path, snapshot_list, False, p_box_size, p_particles_pos, p_particles_vel, p_particles_pid, p_rho_m, p_halos_vel, p_red_shift, p_hubble_constant)             
            c_orbital_assign, c_rad_vel, c_scaled_radii, c_tang_vel = search_halos(c_particle_tree, use_halo_idxs, c_use_halo_pos, c_use_halo_r200m, times_r200m, c_total_num_use_particles, c_use_density_prf_all, c_use_density_prf_1halo, start_nu, end_nu, c_use_halo_id, sparta_file_path, snapshot_list, True, c_box_size, c_particles_pos, c_particles_vel, c_particles_pid, c_rho_m, c_halos_vel, c_red_shift, c_hubble_constant)
            
            match_pidh_idx = np.intersect1d(p_orbital_assign[:,0], c_orbital_assign[:,0], return_indices=True)

            p_orbital_assign = p_orbital_assign[match_pidh_idx[1]]
            p_scaled_radii = p_scaled_radii[match_pidh_idx[1]]
            p_rad_vel = p_rad_vel[match_pidh_idx[1]]
            p_tang_vel = p_tang_vel[match_pidh_idx[1]]

            c_scaled_radii = c_scaled_radii[match_pidh_idx[2]]
            c_rad_vel = c_rad_vel[match_pidh_idx[2]]
            c_tang_vel = c_tang_vel[match_pidh_idx[2]]

            all_scaled_radii = np.column_stack((p_scaled_radii, c_scaled_radii))
            all_rad_vel = np.column_stack((p_rad_vel, c_rad_vel))
            all_tang_vel = np.column_stack((p_tang_vel, c_tang_vel))

            with h5py.File((save_location + "all_particle_properties" + sparta_file_name + ".hdf5"), 'a') as all_particle_properties:

                save_to_hdf5(new_file, all_particle_properties, "HPIDS_" + str(p_snap), dataset = p_orbital_assign[:,0], chunk = True, max_shape = (p_total_num_particles,), curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Orbit_Infall_" + str(p_snap), dataset = p_orbital_assign[:,1], chunk = True, max_shape = (p_total_num_particles,), curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Scaled_radii_" + str(p_snap), dataset = all_scaled_radii, chunk = True, max_shape = (p_total_num_particles,2), curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Radial_vel_magn_" + str(p_snap), dataset = all_rad_vel, chunk = True, max_shape = (p_total_num_particles,2), curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Tangential_vel_magn_" + str(p_snap), dataset = all_tang_vel, chunk = True, max_shape = (p_total_num_particles,2), curr_idx = file_counter, max_num_keys = num_ptl_params)
            
            # with h5py.File((save_location + "all_halo_properties" + sparta_file_name + ".hdf5"), 'a') as all_halo_properties:

            #     save_to_hdf5(new_file, all_halo_properties, "Halo_start_ind_" + str(p_snap), dataset = halo_idxs[:,0], chunk = True, max_shape = (total_num_halos,), curr_idx = file_counter, max_num_keys = num_halo_params, num_snap = snap_done)
            #     save_to_hdf5(new_file, all_halo_properties, "Halo_num_ptl_" + str(p_snap), dataset = halo_idxs[:,1], chunk = True, max_shape = (total_num_halos,), curr_idx = file_counter, max_num_keys = num_halo_params, num_snap = snap_done)
            #     save_to_hdf5(new_file, all_halo_properties, "Halo_id_" + str(p_snap), dataset = use_halo_id, chunk = True, max_shape = (total_num_halos,), curr_idx = file_counter, max_num_keys = num_halo_params, num_snap = snap_done)
            
            file_counter = file_counter + p_scaled_radii.shape[0]

        t_end = time.time()
        print("finished bin:", start_nu, "to", end_nu, "in", np.round((t_end- t_start)/60,2), "minutes,", np.round((t_end- t_start),2), "seconds")
        start_nu = end_nu
        
num_bins = 50        
start_nu = 0
nu_step = 0.5
num_iter = 7
num_save_ptl_params = 5
num_save_halo_params = 3
times_r200 = 6

t_dyn = 2.448854618582507 # calculated by (2 * R200m)/V200m not sure how to do this each time... but hard coded for now from running this code with set snapshots
global halo_start_idx 
curr_sparta_file = "sparta_cbol_l0063_n0256"

# snapshot list should go from high to low (the first value will be what is searched around and generally you want that to be the more recent snap)
snapshot_list = [190]

p_snap = snapshot_list[0]
hdf5_file_path = "/home/zvladimi/MLOIS/SPARTA_data/" + curr_sparta_file + ".hdf5"


t1 = time.time()
print("start particle assign")


t3 = time.time()

halo_start_idx = 0
cosmol = cosmology.setCosmology("bolshoi")
little_h = cosmol.h 
    
snapshot_path = "/home/zvladimi/MLOIS/particle_data/snapdir_" + "{:04d}".format(snapshot_list[0]) + "/snapshot_" + "{:04d}".format(snapshot_list[0])

# get constants from pygadgetreader
p_red_shift = readheader(snapshot_path, 'redshift')
p_scale_factor = 1/(1+p_red_shift)
p_rho_m = cosmol.rho_m(p_red_shift)
p_hubble_constant = cosmol.Hz(p_red_shift) * 0.001 # convert to units km/s/kpc
p_box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
p_box_size = p_box_size * 10**3 * p_scale_factor * little_h #convert to Kpc physical

t4 = time.time()
print("\n finish loading data: ", (t4- t3), " seconds")
#construct a search tree with all of the particle positions

curr_time = cosmol.age(p_red_shift)
past_time = curr_time - t_dyn
snapshot_list.append(find_closest_snap(cosmol, past_time))
print(snapshot_list)

c_snap = snapshot_list[1]
snapshot_path = "/home/zvladimi/MLOIS/particle_data/snapdir_" + "{:04d}".format(c_snap) + "/snapshot_" + "{:04d}".format(c_snap)
save_location =  "/home/zvladimi/MLOIS/calculated_info/" + "calc_from_" + curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[1]) + "/"

# get constants from pygadgetreader
c_red_shift = readheader(snapshot_path, 'redshift')
c_scale_factor = 1/(1+c_red_shift)
c_rho_m = cosmol.rho_m(c_red_shift)
c_hubble_constant = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
c_box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
c_box_size = c_box_size * 10**3 * c_scale_factor * little_h #convert to Kpc physical

p_particles_pid, p_particles_vel, p_particles_pos, mass = load_or_pickle_ptl_data(save_location, str(p_snap), snapshot_path, p_scale_factor, little_h)
p_halos_pos, p_halos_vel, p_halos_r200m, p_halos_id, p_density_prf_all, p_density_prf_1halo, p_halos_status, p_halos_last_snap = load_or_pickle_SPARTA_data(save_location, curr_sparta_file, hdf5_file_path, p_scale_factor, little_h, p_snap)

c_particles_pid, c_particles_vel, c_particles_pos, mass = load_or_pickle_ptl_data(save_location, str(c_snap), snapshot_path, c_scale_factor, little_h)
c_halos_pos, c_halos_vel, c_halos_r200m, c_halos_id, c_density_prf_all, c_density_prf_1halo, c_halos_status, c_halos_last_snap = load_or_pickle_SPARTA_data(save_location, curr_sparta_file, hdf5_file_path, c_scale_factor, little_h, c_snap)

match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= 190) & (c_halos_status > 0) & (c_halos_last_snap >= c_snap))[0]
total_num_halos = match_halo_idxs.shape[0]

p_halos_pos = p_halos_pos[match_halo_idxs]
p_halos_vel = p_halos_vel[match_halo_idxs]
p_halos_r200m = p_halos_r200m[match_halo_idxs]
p_halos_id = p_halos_id[match_halo_idxs]
p_density_prf_all = p_density_prf_all[match_halo_idxs]
p_density_prf_1halo = p_density_prf_1halo[match_halo_idxs]

c_halos_pos = c_halos_pos[match_halo_idxs]
c_halos_vel = c_halos_vel[match_halo_idxs]
c_halos_r200m = c_halos_r200m[match_halo_idxs]
c_halos_id = c_halos_id[match_halo_idxs]
c_density_prf_all = c_density_prf_all[match_halo_idxs]
c_density_prf_1halo = c_density_prf_1halo[match_halo_idxs]


p_particle_tree = cKDTree(data = p_particles_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size)
c_particle_tree = cKDTree(data = c_particles_pos, leafsize = 3, balanced_tree = False, boxsize = c_box_size)

split_halo_by_mass(num_bins, num_save_ptl_params, num_save_halo_params, start_nu, num_iter, nu_step, times_r200, p_halos_r200m, c_halos_r200m, hdf5_file_path, curr_sparta_file, snapshot_list, True, match_halo_idxs)    

t5 = time.time()
print("finish calculations: ", np.round((t5- t3)/60,2), "minutes,", (t5- t3), " seconds" + "\n")

t2 = time.time()
print("Total time: ", np.round((t2- t1)/60,2), "minutes,", (t2- t1), " seconds")
#plt.savefig("/home/zvladimi/MLOIS/Random_figures/avg_rad_vel_vs_pos.png")

