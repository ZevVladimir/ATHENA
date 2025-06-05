import numpy as np
from colossus.halo import mass_so
from colossus.utils import constants
import numexpr as ne

G = constants.G # kpc km^2 / M_⊙ / s^2

# How much memory a halo takes up. Needs to be adjusted if outputted parameters are changed
def calc_halo_mem(n_ptl):
    # rad, rad_vel, tang_vel, physical vel each 4bytes
    # HIPIDS is 8 bytes 
    # orbit/infall is one byte
    # Add some headroom for pandas/hdf5 = 32 bytes per particle
    n_bytes = 32 * n_ptl
    
    return n_bytes

# Calculate radii of particles relative to a halo and the difference in each coordinate for particles and halos for use in calculating rhat
def calc_radius(halo_x, halo_y, halo_z, particle_x, particle_y, particle_z, box_size):
    n_ptls = particle_x.shape[0]
    
    x_dist = particle_x - halo_x
    y_dist = particle_y - halo_y
    z_dist = particle_z - halo_z
    
    coord_diff = np.zeros((n_ptls, 3))
    half_box_size = box_size/2
    
    # Handles periodic boundary conditions by checking if you were to add or subtract a boxsize would it then be within half a box size of the halo
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

    # Calculate radii with standard distance formula
    distance = np.zeros((n_ptls,1))
    distance = np.sqrt(np.square((coord_diff[:,0])) + np.square((coord_diff[:,1])) + np.square((coord_diff[:,2])))
    
    return distance, coord_diff #kpc/h

# Calculates density within sphere of given radius with given mass and calculating volume at each particle's radius
# Also can scale by rho_m if supplied
def calc_rho(masses, bins, r200m, sim_splits, rho_m = None,print_test=False):
    rho = np.zeros_like(masses)
    
    if r200m.size == 1:
        V = (bins * r200m)**3 * 4.0 * np.pi / 3.0
        dV = V[1:] - V[:-1]
        dM = masses[1:] - masses[:-1]
        # if print_test:
        #     print(dV)
        #     print(dM)
        rho[0] = masses[0] / V[0]
        rho[1:] = dM / dV
    else:
        V = (bins[None, :] * r200m[:, None])**3 * 4.0 * np.pi / 3.0
        dV = V[:, 1:] - V[:, :-1]
        dM = masses[:, 1:] - masses[:, :-1]
        rho[:, 0] = masses[:, 0] / V[:, 0]
        rho[:, 1:] = dM / dV

    if rho_m != None:
        if len(sim_splits) == 1:
            rho = rho / rho_m
        else:
            for i in range(len(sim_splits)):
                if i == 0:
                    rho[:sim_splits[i]] = rho[:sim_splits[i]] / rho_m[i]
                elif i == len(sim_splits) - 1:
                    rho[sim_splits[i]:] = rho[sim_splits[i]:] / rho_m[i]
                else:
                    rho[sim_splits[i]:sim_splits[i+1]] = rho[sim_splits[i]:sim_splits[i+1]] / rho_m[i]

    return rho

# Calculate the peculiar velocity which is the particle velocity minus the corresponding halo velocity
def calc_pec_vel(particle_vel, halo_vel):   
    peculiar_velocities = particle_vel - halo_vel   

    return peculiar_velocities # km/s

# Calculate the direction of the radii
def calc_rhat(x_comp, y_comp, z_comp):
    rhat = np.zeros((x_comp.size, 3), dtype = np.float32)
    # Get the magnitude for each particle
    magnitude = np.sqrt(np.square(x_comp) + np.square(y_comp) + np.square(z_comp))
    
    # Scale the components by the magnitude to get a unit vector
    rhat[:,0] = x_comp/magnitude
    rhat[:,1] = y_comp/magnitude
    rhat[:,2] = z_comp/magnitude
    
    return rhat

# Calculate V200m, although this is generalizable to any mass/radius definition
def calc_v200m(mass, radius):
    # calculate the v200m for a halo based on its mass and radius
    return np.sqrt((G * mass)/radius) #km/s

# Calculate the radial velocities of particles
def calc_rad_vel(peculiar_vel, particle_dist, coord_sep, halo_r200m, red_shift, hubble_constant, little_h):
    
    # Get the corresponding components, distances, and halo v200m for every particle
    v_hubble = np.zeros(particle_dist.size, dtype = np.float32)
    corr_m200m = mass_so.R_to_M(halo_r200m, red_shift, "200m") 
    curr_v200m = calc_v200m(corr_m200m, halo_r200m)
        
    # calculate the unit vector of the halo to the particle  
    rhat = calc_rhat(coord_sep[:,0], coord_sep[:,1], coord_sep[:,2])
    
    # Hubble velocity is the hubble constant times the distance the particle is from the halo
    v_hubble = hubble_constant * particle_dist * little_h   #km/s/kpc * kpc/h * h= km/s
    v_hubble = rhat * v_hubble[:, np.newaxis] 
    
    phys_vel_comp = peculiar_vel + v_hubble    

    # dot phys_vel with rhat
    radial_vel_comp = phys_vel_comp * rhat
    radial_vel = np.sum(radial_vel_comp, axis = 1)
    phys_vel = np.linalg.norm(phys_vel_comp, axis = 1)
    
    # Dot the velocity with rhat to get the radial component
    #radial_component_vel = np.sum(np.multiply(peculiar_vel, rhat), axis = 1)
    
    # Add radial component and v_hubble since both are now in radial direction
    #radial_vel = radial_component_vel + v_hubble

    # scale all the radial velocities by v200m of the halo
    return radial_vel, curr_v200m, phys_vel, phys_vel_comp, rhat

# Calculate the tangential velocity of particles (does require radial velocity and physical velocity)
def calc_tang_vel(rv, phys_v_comp, rhat):
    rv_comp = rhat * rv[:, np.newaxis] 
    tv_comp = phys_v_comp - rv_comp
    tv = np.linalg.norm(tv_comp, axis=1)
    return tv

# Calculate the dynamical time based on 200m 
def calc_t_dyn(halo_r200m, red_shift):
    halo_m200m = mass_so.R_to_M(halo_r200m, red_shift, "200m")
    curr_v200m = calc_v200m(halo_m200m, halo_r200m)
    t_dyn = (2*halo_r200m)/curr_v200m
 
    return np.average(t_dyn)

def calc_mass_acc_rate(curr_r200m, past_r200m, curr_z, past_z):
    curr_m200m = mass_so.R_to_M(curr_r200m, curr_z, "200m")
    past_m200m = mass_so.R_to_M(past_r200m, past_z, "200m")
    
    curr_a = 1/(1+curr_z)
    past_a = 1/(1+past_z)
    
    return (np.log(curr_m200m) - np.log(past_m200m)) / (np.log(curr_a) - np.log(past_a))

# For a halo calculate the radius, radial velocity, tangential velocity for each particle, determine the particle's classification and assigne each particle a unique particle id-halo index id (HIPID)
def calc_halo_params(comp_snap, snap_dict, curr_halo_idx, curr_ptl_pids, curr_ptl_pos, curr_ptl_vel, 
                 halo_pos, halo_vel, halo_r200m, sparta_last_pericenter_snap=None, sparta_n_pericenter=None, sparta_tracer_ids=None,
                 sparta_n_is_lower_limit=None):
    snap = snap_dict["snap"]
    red_shift = snap_dict["red_shift"]
    scale_factor = snap_dict["scale_factor"]
    hubble_const = snap_dict["hubble_const"]
    box_size = snap_dict["box_size"] 
    little_h = snap_dict["h"]   
    
    halo_pos = halo_pos * 10**3 * scale_factor # Convert to kpc/h
    
    num_new_ptls = curr_ptl_pids.shape[0]

    # Generate unique ids for each particle id and halo idx combination. This is used to match particles within halos across snapshots
    curr_ptl_pids = curr_ptl_pids.astype(np.int64) # otherwise ne.evaluate doesn't work
    fnd_HIPIDs = ne.evaluate("0.5 * (curr_ptl_pids + curr_halo_idx) * (curr_ptl_pids + curr_halo_idx + 1) + curr_halo_idx")
    
    # Calculate the radii of each particle based on the distance formula
    ptl_rad, coord_dist = calc_radius(halo_pos[0], halo_pos[1], halo_pos[2], curr_ptl_pos[:,0], curr_ptl_pos[:,1], curr_ptl_pos[:,2], box_size)         
    
    # Only find orbiting(1)/infalling(0) classification for the primary snapshot
    if comp_snap == False:         
        compare_sparta_assn = np.zeros((sparta_tracer_ids.shape[0]))
        curr_orb_assn = np.zeros((num_new_ptls))
        
        # Anywhere sparta_last_pericenter is greater than the current snap then that is in the future so set to 0
        future_peri = np.where(sparta_last_pericenter_snap > snap)[0]
        adj_sparta_n_pericenter = sparta_n_pericenter
        adj_sparta_n_pericenter[future_peri] = 0
        adj_sparta_n_is_lower_limit = sparta_n_is_lower_limit
        adj_sparta_n_is_lower_limit[future_peri] = 0
        
        # If a particle has a pericenter or if the lower limit is 1 then it is orbiting
        compare_sparta_assn[np.where((adj_sparta_n_pericenter >= 1) | (adj_sparta_n_is_lower_limit == 1))[0]] = 1
        
        # Compare the ids between SPARTA and the found particle ids and match the SPARTA results
        matched_ids = np.intersect1d(curr_ptl_pids, sparta_tracer_ids, return_indices = True)
        curr_orb_assn[matched_ids[1]] = compare_sparta_assn[matched_ids[2]]

    # calculate peculiar, radial, and tangential velocity
    pec_vel = calc_pec_vel(curr_ptl_vel, halo_vel)
    fnd_rad_vel, curr_v200m, phys_vel, phys_vel_comp, rhat = calc_rad_vel(pec_vel, ptl_rad, coord_dist, halo_r200m, red_shift, hubble_const, little_h)
    fnd_tang_vel = calc_tang_vel(fnd_rad_vel, phys_vel_comp, rhat)
    
    # Scale radius by R200m, and velocities by V200m
    scaled_radii = ptl_rad / halo_r200m
    scaled_rad_vel = fnd_rad_vel / curr_v200m
    scaled_tang_vel = fnd_tang_vel / curr_v200m
    scaled_phys_vel = phys_vel / curr_v200m
    
    # Sort the radii so they go from smallest to largest, sort the other parameters in the same way
    scaled_radii_inds = scaled_radii.argsort()
    scaled_radii = scaled_radii[scaled_radii_inds]
    fnd_HIPIDs = fnd_HIPIDs[scaled_radii_inds]
    scaled_rad_vel = scaled_rad_vel[scaled_radii_inds]
    scaled_tang_vel = scaled_tang_vel[scaled_radii_inds]
    scaled_phys_vel = scaled_phys_vel[scaled_radii_inds]

    if comp_snap == False:
        curr_orb_assn = curr_orb_assn[scaled_radii_inds]

    if comp_snap == False:
        return fnd_HIPIDs, curr_orb_assn, scaled_rad_vel, scaled_tang_vel, scaled_radii, scaled_phys_vel
    else:
        return fnd_HIPIDs, scaled_rad_vel, scaled_tang_vel, scaled_radii

# Print the number of infalling particles within a multiple of R200m and the number of orbiting outside and the fraction of the total pop.
# r_scale is the radius scaled by R200m, nr200m is the multiple of r200m to consider
def nptl_inc_placement_r200m(r_scale, mult_r200m, orb_assn):
    # Get the number of infalling particles within R200m and orbiting particles outside
    n_inf_inside = np.where((r_scale < mult_r200m) & (orb_assn == 0))[0].shape[0]
    n_orb_inside = np.where((r_scale > mult_r200m) & (orb_assn == 1))[0].shape[0]
    
    n_inf = np.where(orb_assn==0)[0].shape[0]
    n_orb = np.where(orb_assn==1)[0].shape[0]

    print("Number of infalling particles within R200m:",n_inf_inside,"Fraction of total infalling population:",n_inf_inside / n_inf)
    print("Number of orbiting particles outside of R200m:",n_orb_inside, "Fraction of total orbiting population:",n_orb_inside / n_orb)
        
# Calculates the scaled position weight for a dataset. Which is used to weight the model towards the population with less particles (should be the orbiting population)
def calc_scal_pos_weight(df):
    count_negatives = (df['Orbit_infall'] == 0).sum()
    count_positives = (df['Orbit_infall'] == 1).sum()

    scale_pos_weight = count_negatives / count_positives
    return scale_pos_weight
        