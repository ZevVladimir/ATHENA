import numexpr as ne
import numpy as np
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import peaks
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time
import h5py
import pickle
import os
##################################################################################################################
# General params
curr_sparta_file = "sparta_cbol_l0063_n0256"
path_to_MLOIS = "/home/zvladimi/MLOIS/"
path_to_snapshot = "/home/zvladimi/MLOIS/particle_data/"
path_to_hdf5 = "/home/zvladimi/MLOIS/SPARTA_data/"
path_to_pickle = "/home/zvladimi/MLOIS/pickle_data/"
prim_only = False
t_dyn_step = .5
snapshot_list = [190]
times_r200m = 6
total_num_snaps = 193
#TODO have these params set automatically
num_save_ptl_params = 5
num_save_halo_params = 3
test_halos_ratio = .2
p_snap = snapshot_list[0]

snap_format = "{:04d}" # how are the snapshots formatted with 0s
hdf5_file_path = path_to_hdf5 + curr_sparta_file + ".hdf5"
snapshot_path = path_to_snapshot + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

# Params for splitting up the search
num_halo_per_split = 1000
num_print_per_split = 5
cosmol = cosmology.setCosmology("bolshoi")
little_h = cosmol.h 
##################################################################################################################
# import pygadgetreader and sparta
import sys
sys.path.insert(0, path_to_MLOIS + "pygadgetreader")
sys.path.insert(0, path_to_MLOIS + "sparta/analysis")
from pygadgetreader import readsnap, readheader
from sparta import sparta
from data_and_loading_functions import load_or_pickle_SPARTA_data, load_or_pickle_ptl_data, save_to_hdf5, find_closest_snap, conv_halo_id_spid
from visualization_functions import compare_density_prf
from calculation_functions import *
##################################################################################################################

def initial_search(halo_positions, search_radius, halo_r200m, tree, red_shift):
    t_dyn_flag = False
    num_halos = halo_positions.shape[0]
    particles_per_halo = np.zeros(num_halos, dtype = np.int32)
    all_halo_mass = np.zeros(num_halos, dtype = np.float32)
    
    for i in range(num_halos):
        if halo_r200m[i] > 0:
            #find how many particles we are finding
            indices = tree.query_ball_point(halo_positions[i,:], r = search_radius * halo_r200m[i])

            # how many new particles being added and correspondingly how massive the halo is
            num_new_particles = len(indices)
            all_halo_mass[i] = num_new_particles * mass
            particles_per_halo[i] = num_new_particles
            
            if t_dyn_flag == False:
                corresponding_hubble_m200m = mass_so.R_to_M(halo_r200m[i], red_shift, "200c") * little_h
                curr_v200m = calc_v200m(corresponding_hubble_m200m, halo_r200m[np.where(halo_r200m>0)[0][0]])
                t_dyn = (2*halo_r200m[i])/curr_v200m
                print("t_dyn:", t_dyn)
                t_dyn_flag = True
                
    return particles_per_halo, all_halo_mass, t_dyn

def search_halos(particle_tree, sparta_output, halo_idxs, search_radius, total_particles, curr_snap, box_size, particles_pos, particles_vel, particles_pid, rho_m, red_shift, scale_factor, hubble_constant, comp_snap):
    num_halos = halo_idxs.shape[0]
    halos_pos = sparta_output['halos']['position'][:,curr_snap,:] * 10**3 * scale_factor * little_h
    halos_vel = sparta_output['halos']['velocity'][:,curr_snap,:]
    halos_r200m = sparta_output['halos']['R200m'][:,curr_snap] * little_h
    dens_prf_all = sparta_output['anl_prf']['M_all'][:,curr_snap,:]
    dens_prf_1halo = sparta_output['anl_prf']['M_1halo'][:,curr_snap,:]

    all_part_vel = np.zeros((total_particles,3), dtype = np.float32)
    calculated_r200m = np.zeros(halos_r200m.size)
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
        current_halo_pos = halos_pos[i]

        # find the indices of the particles within the expected r200 radius multiplied by times_r200 
        indices = particle_tree.query_ball_point(current_halo_pos, r = search_radius * halos_r200m[i])
        
        #Only take the particle positions that where found with the tree
        current_particles_pos = particles_pos[indices,:]
        current_particles_vel = particles_vel[indices,:]
        current_particles_pid = particles_pid[indices]
        current_particles_pid = current_particles_pid.astype(np.int64)
        num_new_particles = len(indices)
        
        # current_orbit_assn_sparta: halo_idx, PID, orb/inf
        current_orbit_assn_sparta = np.zeros((current_particles_pid.size, 2), dtype = np.uint64)

        # generate unique index for each particle according to it's id and the halo it belongs to
        curr_halo_idx = halo_idxs[i]
        current_orbit_assn_sparta[:,0] = ne.evaluate("0.5 * (current_particles_pid + curr_halo_idx) * (current_particles_pid + curr_halo_idx + 1) + curr_halo_idx")

        #for how many new particles create an array of how much mass there should be within that particle radius
        use_mass = np.arange(1, num_new_particles + 1, 1) * mass       
        all_part_vel[start:start+num_new_particles] = current_particles_vel
            
        #calculate the radii of each particle based on the distance formula
        unsorted_particle_radii, unsorted_coord_dist = calculate_distance(current_halo_pos[0], current_halo_pos[1], current_halo_pos[2], current_particles_pos[:,0],current_particles_pos[:,1], current_particles_pos[:,2], num_new_particles, box_size)         
        
        # Assign particles to infalling or orbiting based off SPARTA output parameters for tracers
        if comp_snap == False:
            curr_halo_first = sparta_output['halos']['ptl_oct_first'][i]
            curr_halo_n = sparta_output['halos']['ptl_oct_n'][i]
            
            sparta_last_pericenter_snap = sparta_output['tcr_ptl']['res_oct']['last_pericenter_snap'][curr_halo_first:curr_halo_first+curr_halo_n]
            sparta_n_pericenter = sparta_output['tcr_ptl']['res_oct']['n_pericenter'][curr_halo_first:curr_halo_first+curr_halo_n]
            sparta_tracer_ids = sparta_output['tcr_ptl']['res_oct']['tracer_id'][curr_halo_first:curr_halo_first+curr_halo_n]
            sparta_n_is_lower_limit = sparta_output['tcr_ptl']['res_oct']['n_is_lower_limit'][curr_halo_first:curr_halo_first+curr_halo_n]

            orbit_assn_sparta = np.zeros((sparta_tracer_ids.size, 2), dtype = np.int64)
            orbit_assn_sparta[:,0] = sparta_tracer_ids

            sparta_n_pericenter[np.where(sparta_last_pericenter_snap > curr_snap)[0]] = 0 # only want pericenters that have occurred at or before this snapshot
            orbit_assn_sparta[np.where(sparta_n_pericenter > 0)[0],1] = 1 # if there is more than one pericenter count as orbiting (1) and if it isn't it is infalling (0)
            orbit_assn_sparta[np.where(sparta_n_is_lower_limit == 1)[0],1] = 1 # However particle can also be orbiting if n_is_lower_limit is 1

            poss_pids = np.intersect1d(current_particles_pid, orbit_assn_sparta[:,0], return_indices = True) # only check pids that are within the tracers for this halo (otherwise infall)
            #poss_pid_match = np.intersect1d(current_particles_pid[poss_pids[1]], orbit_assn_sparta[:,0], return_indices = True) # get the corresponding indices for the pids and their infall/orbit assn
            current_orbit_assn_sparta[poss_pids[1],1] = orbit_assn_sparta[poss_pids[2],1] # create a mask to then set any particle that is not identified as orbiting to be infalling
        

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
        calculated_radial_velocities[start:start+num_new_particles], calculated_radial_velocities_comp[start:start+num_new_particles], curr_v200m, physical_vel, rhat = calc_rad_vel(peculiar_velocity, particle_radii, coord_dist, halos_r200m[i], red_shift, little_h, hubble_constant)
        calculated_tangential_velocities_comp[start:start+num_new_particles] = calc_tang_vel(calculated_radial_velocities[start:start+num_new_particles], physical_vel, rhat)/curr_v200m
        calculated_tangential_velocities[start:start+num_new_particles] = np.linalg.norm(calculated_tangential_velocities_comp[start:start+num_new_particles], axis = 1)

        # scale radial velocities and their components by V200m
        # scale radii by R200m
        # assign all values to portion of entire array for this halo mass bin
        calculated_radial_velocities[start:start+num_new_particles] = calculated_radial_velocities[start:start+num_new_particles]/curr_v200m
        calculated_radial_velocities_comp[start:start+num_new_particles] = calculated_radial_velocities_comp[start:start+num_new_particles]/curr_v200m
        all_radii[start:start+num_new_particles] = particle_radii
        all_scaled_radii[start:start+num_new_particles] = particle_radii/halos_r200m[i]
        r200m_per_part[start:start+num_new_particles] = halos_r200m[i]
        all_orbit_assn[start:start+num_new_particles] = current_orbit_assn_sparta

        start += num_new_particles

        if i % 250 == 0 and comp_snap == False:
            compare_density_prf(radii = particle_radii/halos_r200m[i], actual_prf_all=dens_prf_all[i], actual_prf_1halo=dens_prf_1halo[i], mass=mass, orbit_assn=current_orbit_assn_sparta, title = str(halo_idxs[i]), save_location="/home/zvladimi/MLOIS/Random_figures/", show_graph=False, save_graph=True)
        
        print_at = int(np.ceil((num_halo_per_split/num_print_per_split)))
        if i % print_at == 0 and i != 0:
            t_lap = time.time()
            tot_time = t_lap - t_start
            t_remain = (num_halos - i)/print_at * tot_time
            print("Halos:", (i-print_at), "to", i, "time taken:", np.round(tot_time,2), "seconds" , "time remaining:", np.round(t_remain/60,2), "minutes,",  np.round((t_remain),2), "seconds")
            t_start = t_lap
            
    return all_orbit_assn, calculated_radial_velocities, all_scaled_radii, calculated_tangential_velocities   
    
def get_comp_snap(t_dyn, t_dyn_step):
    # calculate one dynamical time ago and set that as the comparison snap
    curr_time = cosmol.age(p_red_shift)
    past_time = curr_time - (t_dyn_step * t_dyn)
    c_snap = find_closest_snap(cosmol, past_time, num_snaps = total_num_snaps, path_to_snap=path_to_snapshot, snap_format = snap_format)
    snapshot_list.append(c_snap)

    # switch to comparison snap
    c_snap = snapshot_list[1]
    snapshot_path = path_to_snapshot + "/snapdir_" + snap_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap)
    save_location =  path_to_MLOIS + "calculated_info/" + curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[1]) + "_" + str(times_r200m) + "r200msearch/"
    
    if os.path.exists(save_location) != True:
        os.makedirs(save_location)
        
    # get constants from pygadgetreader
    c_red_shift = readheader(snapshot_path, 'redshift')
    c_scale_factor = 1/(1+c_red_shift)
    c_rho_m = cosmol.rho_m(c_red_shift)
    c_hubble_constant = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
    c_box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
    c_box_size = c_box_size * 10**3 * c_scale_factor * little_h #convert to Kpc physical
    
    # load particle data and SPARTA data for the comparison snap
    c_particles_pid, c_particles_vel, c_particles_pos, mass = load_or_pickle_ptl_data(str(c_snap), snapshot_path, c_scale_factor, little_h, path_to_pickle)
    c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap = load_or_pickle_SPARTA_data(curr_sparta_file, hdf5_file_path, c_scale_factor, little_h, c_snap, path_to_pickle)

    return save_location, c_snap, c_box_size, c_rho_m, c_red_shift, c_hubble_constant, c_particles_pid, c_particles_vel, c_particles_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap 

def split_halo_by_mass(num_ptl_params, times_r200m, p_num_particles_per_halo, p_halos_id, sparta_file_name, p_snap, new_file, train, dataset_idxs, c_halos_pos = None, c_halos_r200m = None, c_halos_id = None, c_red_shift = None):
    #TODO have it so you don't have to delete the .hdf5 file each time
    prim_only = False
    if c_halos_pos is None or c_halos_r200m is None or c_halos_id is None:
        print("Can't do comparison, only doing primary snapshot")
        prim_only = True
        
    total_num_halos = dataset_idxs.shape[0]
           
    # choose all of these halos information
    p_num_particles_per_halo = p_num_particles_per_halo[dataset_idxs]
    p_total_num_particles = np.sum(p_num_particles_per_halo)

    if prim_only == False:
        c_num_particles_per_halo, c_halo_masses, c_t_dyn = initial_search(c_halos_pos, times_r200m, c_halos_r200m, c_particle_tree, c_red_shift)
        
    file_counter = 0
    num_iter = int(np.ceil(total_num_halos / num_halo_per_split))
    print("Total num bins:",num_iter)
        
    # For how many mass bin splits
    for i in range(num_iter):
        t_start = time.time()

        if i < num_iter - 1:
            halos_within_range = np.arange((i * num_halo_per_split), ((i+1) * num_halo_per_split))
        else:
            halos_within_range = np.arange((i * num_halo_per_split), total_num_halos)

        # only get the parameters we want for this specific halo bin
        use_halo_idxs = dataset_idxs[halos_within_range]
        p_use_num_particles = p_num_particles_per_halo[halos_within_range]
        p_total_num_use_particles = np.sum(p_use_num_particles)
        print("Num particles primary: ", p_total_num_use_particles)

        p_use_halo_id = p_halos_id[use_halo_idxs]
   
        sparta_output = sparta.load(filename=hdf5_file_path, halo_ids=p_use_halo_id, load_halo_data=True, results = ['oct'], log_level=0)
        new_idxs = conv_halo_id_spid(p_use_halo_id, sparta_output, p_snap)
        use_halo_idxs = use_halo_idxs[new_idxs]
                                    
        p_orbital_assign, p_rad_vel, p_scaled_radii, p_tang_vel = search_halos(p_particle_tree, sparta_output, use_halo_idxs, times_r200m, p_total_num_use_particles, p_snap, p_box_size, p_particles_pos, p_particles_vel, p_particles_pid, p_rho_m, p_red_shift, p_scale_factor, p_hubble_constant, comp_snap = False)

        if prim_only == False:
            all_scaled_radii = np.zeros((p_total_num_use_particles,2))
            all_rad_vel = np.zeros((p_total_num_use_particles,2))
            all_tang_vel = np.zeros((p_total_num_use_particles,2))

            c_use_num_particles = c_num_particles_per_halo[use_halo_idxs]
            c_total_num_use_particles = np.sum(c_use_num_particles)
            print("Num particles compare: ", c_total_num_use_particles)
                        
            c_orbital_assign, c_rad_vel, c_scaled_radii, c_tang_vel = search_halos(c_particle_tree, sparta_output, use_halo_idxs, times_r200m, c_total_num_use_particles, c_snap, c_box_size, c_particles_pos, c_particles_vel, c_particles_pid, c_rho_m, c_red_shift, c_scale_factor, c_hubble_constant, comp_snap = True)

            match_pidh_idx = np.intersect1d(p_orbital_assign[:,0], c_orbital_assign[:,0], return_indices=True)
            all_scaled_radii[:,0] = p_scaled_radii
            all_scaled_radii[match_pidh_idx[1],1] = c_scaled_radii[match_pidh_idx[2]]
            all_rad_vel[:,0] = p_rad_vel
            all_rad_vel[match_pidh_idx[1],1] = c_rad_vel[match_pidh_idx[2]]
            all_tang_vel[:,0] = p_tang_vel
            all_tang_vel[match_pidh_idx[1],1] = c_tang_vel[match_pidh_idx[2]]
            
            use_max_shape = (p_total_num_particles,2)
        
        if prim_only == True:
            all_scaled_radii = p_scaled_radii
            all_rad_vel = p_rad_vel
            all_tang_vel = p_tang_vel
            use_max_shape = (p_total_num_particles,)
    
        if train:
            with h5py.File((save_location + "train_all_particle_properties_" + sparta_file_name + ".hdf5"), 'a') as all_particle_properties:
                save_location + "train_all_particle_properties_" + sparta_file_name + ".hdf5"
                save_to_hdf5(new_file, all_particle_properties, "HPIDS", dataset = p_orbital_assign[:,0], chunk = True, max_shape = (p_total_num_particles,), curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Orbit_Infall", dataset = p_orbital_assign[:,1], chunk = True, max_shape = (p_total_num_particles,), curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Scaled_radii_", dataset = all_scaled_radii, chunk = True, max_shape = use_max_shape, curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Radial_vel_", dataset = all_rad_vel, chunk = True, max_shape = use_max_shape, curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Tangential_vel_", dataset = all_tang_vel, chunk = True, max_shape = use_max_shape, curr_idx = file_counter, max_num_keys = num_ptl_params)
        else:
            with h5py.File((save_location + "test_all_particle_properties_" + sparta_file_name + ".hdf5"), 'a') as all_particle_properties:
                save_to_hdf5(new_file, all_particle_properties, "HPIDS", dataset = p_orbital_assign[:,0], chunk = True, max_shape = (p_total_num_particles,), curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Orbit_Infall", dataset = p_orbital_assign[:,1], chunk = True, max_shape = (p_total_num_particles,), curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Scaled_radii_", dataset = all_scaled_radii, chunk = True, max_shape = use_max_shape, curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Radial_vel_", dataset = all_rad_vel, chunk = True, max_shape = use_max_shape, curr_idx = file_counter, max_num_keys = num_ptl_params)
                save_to_hdf5(new_file, all_particle_properties, "Tangential_vel_", dataset = all_tang_vel, chunk = True, max_shape = use_max_shape, curr_idx = file_counter, max_num_keys = num_ptl_params)

        file_counter = file_counter + all_scaled_radii.shape[0]


        t_end = time.time()
        print("finished bin:", (i+1), "in", np.round((t_end- t_start)/60,2), "minutes,", np.round((t_end- t_start),2), "seconds \n")
        
t1 = time.time()
t3 = time.time()

# get constants for primary snap
p_red_shift = readheader(snapshot_path, 'redshift')
p_scale_factor = 1/(1+p_red_shift)
p_rho_m = cosmol.rho_m(p_red_shift)
p_hubble_constant = cosmol.Hz(p_red_shift) * 0.001 # convert to units km/s/kpc
p_box_size = readheader(snapshot_path, 'boxsize') #units Mpc/h comoving
p_box_size = p_box_size * 10**3 * p_scale_factor * little_h #convert to Kpc physical

if prim_only == True:
    save_location =  path_to_MLOIS +  "calculated_info/" + curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(times_r200m) + "r200msearch/"
    if os.path.exists(save_location) != True:
        os.makedirs(save_location)
        
p_particles_pid, p_particles_vel, p_particles_pos, mass = load_or_pickle_ptl_data(str(p_snap), snapshot_path, p_scale_factor, little_h, path_to_pickle)
p_halos_pos, p_halos_r200m, p_halos_id, p_halos_status, p_halos_last_snap = load_or_pickle_SPARTA_data(curr_sparta_file, hdf5_file_path, p_scale_factor, little_h, p_snap, path_to_pickle)
p_particle_tree = cKDTree(data = p_particles_pos, leafsize = 3, balanced_tree = False, boxsize = p_box_size) # construct search trees for each snapshot

if prim_only == False:   
    # get the halo_masses and number of particles for primary and comparison snap
    p_num_particles_per_halo, p_halo_masses, t_dyn = initial_search(p_halos_pos, times_r200m, p_halos_r200m, p_particle_tree, p_red_shift)
    save_location, c_snap, c_box_size, c_rho_m, c_red_shift, c_hubble_constant, c_particles_pid, c_particles_vel, c_particles_pos, c_halos_pos, c_halos_r200m, c_halos_id, c_halos_status, c_halos_last_snap = get_comp_snap(t_dyn=t_dyn,t_dyn_step=t_dyn_step)
c_scale_factor = 1/(1+c_red_shift)

# only take halos that are hosts in primary snap and exist past the p_snap and exist in some form at the comparison snap
if prim_only == False:
    match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_snap) & (c_halos_status > 0) & (c_halos_last_snap >= c_snap))[0]
    
if prim_only == True:
    match_halo_idxs = np.where((p_halos_status == 10) & (p_halos_last_snap >= p_snap))[0]

total_num_halos = match_halo_idxs.shape[0]

rng = np.random.default_rng(11)
rng.shuffle(match_halo_idxs)
# split all indices into train and test groups
train_indices, test_indices = np.split(match_halo_idxs, [int((1-test_halos_ratio) * total_num_halos)])
# need to sort indices otherwise sparta.load breaks...
train_indices = np.sort(train_indices)
test_indices = np.sort(test_indices)

print("Total num halos:", total_num_halos)
print("Num train halos:", train_indices.shape[0])
print("Num test halos:", test_indices.shape[0])
print("Total num ptls:", np.sum(p_num_particles_per_halo))

with open(save_location + "test_indices.pickle", "wb") as pickle_file:
    pickle.dump(test_indices, pickle_file)
with open(save_location + "train_indices.pickle", "wb") as pickle_file:
    pickle.dump(train_indices, pickle_file)

t4 = time.time()
print("\nFinish setup:", (t4- t3), "seconds using snapshots:",snapshot_list)

if prim_only == False:
    c_particle_tree = cKDTree(data = c_particles_pos, leafsize = 3, balanced_tree = False, boxsize = c_box_size) 
    print("Start train set")                   
    split_halo_by_mass(num_save_ptl_params, times_r200m, p_num_particles_per_halo, p_halos_id, sparta_file_name = curr_sparta_file, p_snap = p_snap, new_file = True, train = True, dataset_idxs = train_indices, c_halos_pos = c_halos_pos, c_halos_r200m = c_halos_r200m, c_halos_id = c_halos_id, c_red_shift=c_red_shift)    
    print("Start test set")
    split_halo_by_mass(num_save_ptl_params, times_r200m, p_num_particles_per_halo, p_halos_id, sparta_file_name = curr_sparta_file, p_snap = p_snap, new_file = True, train = False, dataset_idxs = test_indices, c_halos_pos = c_halos_pos, c_halos_r200m = c_halos_r200m, c_halos_id = c_halos_id, c_red_shift=c_red_shift)    
if prim_only == True:
    print("Start train set")
    split_halo_by_mass(num_save_ptl_params, times_r200m, p_num_particles_per_halo, p_halos_id, sparta_file_name = curr_sparta_file, p_snap = p_snap, new_file = True, train = True, dataset_idxs = train_indices)    
    print("Start test set")
    split_halo_by_mass(num_save_ptl_params, times_r200m, p_num_particles_per_halo, p_halos_id, sparta_file_name = curr_sparta_file, p_snap = p_snap, new_file = True, train = False, dataset_idxs = test_indices)    

t5 = time.time()
print("finish calculations:", np.round((t5- t4)/60,2), "minutes,", (t5- t3), "seconds")

t2 = time.time()
print("Total time:", np.round((t2- t1)/60,2), "minutes,", (t2- t1), "seconds")