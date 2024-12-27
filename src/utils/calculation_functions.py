import numpy as np
from colossus.halo import mass_so
from colossus.utils import constants
import multiprocessing as mp
from itertools import repeat
import numexpr as ne
from colossus.lss.peaks import peakHeight
from colossus.halo.mass_so import M_to_R

G = constants.G # kpc km^2 / M_⊙ / s^2

# Calculate radii of particles relative to a halo and the difference in each coordinate for particles and halos for use in calculating rhat
def calc_radius(halo_x, halo_y, halo_z, particle_x, particle_y, particle_z, new_particles, box_size):
    x_dist = particle_x - halo_x
    y_dist = particle_y - halo_y
    z_dist = particle_z - halo_z
    
    coord_diff = np.zeros((new_particles, 3))
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
    distance = np.zeros((new_particles,1))
    distance = np.sqrt(np.square((halo_x - particle_x)) + np.square((halo_y - particle_y)) + np.square((halo_z - particle_z)))
    
    return distance, coord_diff #kpc/h

# Calculates density within sphere of given radius with given mass and calculating volume at each particle's radius
# Also can scale by rho_m if supplied
def calculate_density(masses, bins, r200m, sim_splits, rho_m = None):
    rho = np.zeros_like(masses)
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
    corr_m200m = mass_so.R_to_M(halo_r200m, red_shift, "200c") 
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

    return t_dyn

def update_density_prf(calc_prf, radii, idx, start_bin, end_bin, mass):
    radii_within_range = np.where((radii >= start_bin) & (radii < end_bin))[0]
    
    # If there are particles in this bin and its not the first bin
    # Then add the mass of prior bin to the mass of this bin
    if radii_within_range.size != 0 and idx != 0:
        calc_prf[idx] = calc_prf[idx - 1] + radii_within_range.size * mass
    # If there are particles in this bin and its  the first bin
    # Then simply the mass of this bin
    elif radii_within_range.size != 0 and idx == 0:
        calc_prf[idx] = radii_within_range.size * mass
    # If there are  no particles in this bin and its not the first bin
    # Then use the mass of prior bin 
    elif radii_within_range.size == 0 and idx != 0:
        calc_prf[idx] = calc_prf[idx-1]
    # If there are  no particles in this bin and its the first bin
    # Then use the mass of this bin 
    else:
        calc_prf[idx] = calc_prf[idx-1]
    
    return calc_prf

def diff_n_prf(diff_n_ptl, radii, idx, start_bin, end_bin, mass, act_prf):
    radii_within_range = np.where((radii >= start_bin) & (radii < end_bin))[0]
    
    if radii_within_range.size != 0 and idx != 0:
        diff_n_ptl[idx] = np.round((act_prf[idx] - act_prf[idx-1])/mass) - radii_within_range.size
    elif radii_within_range.size != 0 and idx == 0:
        diff_n_ptl[idx] = np.round(act_prf[idx]/mass) - radii_within_range.size
    elif radii_within_range.size == 0 and idx != 0:
        diff_n_ptl[idx] = np.round((act_prf[idx] - act_prf[idx-1])/mass) - radii_within_range.size
    else:
        diff_n_ptl[idx] = np.round((act_prf[idx])/mass)
        
    return diff_n_ptl

def create_mass_prf(radii, orbit_assn, prf_bins, mass):  
    # Create bins for the density profile calculation
    num_prf_bins = prf_bins.shape[0] - 1

    calc_mass_prf_orb = np.zeros(num_prf_bins)
    calc_mass_prf_inf = np.zeros(num_prf_bins)
    calc_mass_prf_all = np.zeros(num_prf_bins)

    # Can adjust this to cut out halos that don't have enough particles within R200m
    min_ptl = 200
    ptl_in_r200m = np.where(radii <= 1)[0].size
    if ptl_in_r200m < min_ptl:
        calc_mass_prf_orb[:]=np.nan
        calc_mass_prf_inf[:]=np.nan
        calc_mass_prf_all[:]=np.nan
        m200m = np.nan
    else:
        # determine which radii correspond to orbiting and which to infalling
        orbit_radii = radii[np.where(orbit_assn == 1)[0]]
        infall_radii = radii[np.where(orbit_assn == 0)[0]]

        # loop through each bin's radii range and get the mass of each type of particle
        for i in range(num_prf_bins):
            start_bin = prf_bins[i]
            end_bin = prf_bins[i+1]          

            calc_mass_prf_all  = update_density_prf(calc_mass_prf_all, radii, i, start_bin, end_bin, mass)    
            calc_mass_prf_orb = update_density_prf(calc_mass_prf_orb, orbit_radii, i, start_bin, end_bin, mass)      
            calc_mass_prf_inf = update_density_prf(calc_mass_prf_inf, infall_radii, i, start_bin, end_bin, mass)      
    
        m200m = ptl_in_r200m * mass
    
    return calc_mass_prf_all,calc_mass_prf_orb, calc_mass_prf_inf, m200m

def comb_prf(prf, num_halo, dtype):
    if num_halo > 1:
        prf = np.stack(prf, axis=0)
    else:
        prf = np.asarray(prf)
        prf = np.reshape(prf, (1,prf.size))
    
    prf = prf.astype(dtype)

    return prf

def filter_prf(calc_prf, act_prf, min_disp_halos, nu_fltr = None):
    # for each bin checking how many halos have particles there
    # if there are less than half the total number of halos then just treat that bin as having 0
    for i in range(calc_prf.shape[1]):
        if np.where(calc_prf[:,i]>0)[0].shape[0] < min_disp_halos:
            calc_prf[:,i] = np.nan
            act_prf[:,i] = np.nan
            
    if nu_fltr is not None:
        calc_prf = calc_prf[nu_fltr,:]
        act_prf = act_prf[nu_fltr,:]
        
    return calc_prf, act_prf        

def create_stack_mass_prf(splits, radii, halo_first, halo_n, mass, orbit_assn, prf_bins, use_mp = True, all_z = []):
    calc_mass_prf_orb_lst = []
    calc_mass_prf_inf_lst = []
    calc_mass_prf_all_lst = []
    calc_nu_lst = []
    calc_r200m_lst = []
    
    for i in range(splits.size):
        if i < splits.size - 1:
            curr_num_halos = halo_n[splits[i]:splits[i+1]].shape[0]
        else:
            curr_num_halos = halo_n[splits[i]:].shape[0]
        # for each halo get the corresponding mass and density profile that the model predicts for it
        # Can do this using either multiprocessing or a for loop
        if use_mp:
            num_processes = mp.cpu_count()
            with mp.Pool(processes=num_processes) as p:
                calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_m200m = zip(*p.starmap(create_mass_prf, 
                                            zip((radii[halo_first[n]:halo_first[n]+halo_n[n]] for n in range(splits[i],splits[i]+curr_num_halos)),
                                                (orbit_assn[halo_first[n]:halo_first[n]+halo_n[n]] for n in range(splits[i],splits[i]+curr_num_halos)),
                                                repeat(prf_bins),repeat(mass[i])),
                                            chunksize=100))
            p.close()
            p.join()       

        else:
            calc_mass_prf_orb = []
            calc_mass_prf_inf = []
            calc_mass_prf_all = []
            calc_m200m = []
            
            for j in range(curr_num_halos):
                halo_mass_prf_all, halo_mass_prf_orb, halo_mass_prf_inf, m200m = create_mass_prf(radii[halo_first[splits[i]+j]:halo_first[splits[i]+j]+halo_n[splits[i]+j]], orbit_assn[halo_first[splits[i]+j]:halo_first[splits[i]+j]+halo_n[splits[i]+j]], prf_bins,mass[i])
                calc_mass_prf_orb.append(np.array(halo_mass_prf_orb))
                calc_mass_prf_inf.append(np.array(halo_mass_prf_inf))
                calc_mass_prf_all.append(np.array(halo_mass_prf_all))
                m200m = np.array(m200m)
                calc_m200m.append(m200m)
                
        # For each profile combine all halos for each bin
        # calc_mass_prf_xxx has shape (num_halo, num_bins)

        calc_mass_prf_orb_lst.append(comb_prf(calc_mass_prf_orb, curr_num_halos, np.float32))
        calc_mass_prf_inf_lst.append(comb_prf(calc_mass_prf_inf, curr_num_halos, np.float32))
        calc_mass_prf_all_lst.append(comb_prf(calc_mass_prf_all, curr_num_halos, np.float32))

        calc_nu_lst.append(peakHeight(np.array(calc_m200m),all_z[i]))
        calc_r200m_lst.append(M_to_R(np.array(calc_m200m),all_z[i],"200m"))

    calc_mass_prf_orb = np.vstack(calc_mass_prf_orb_lst)
    calc_mass_prf_inf = np.vstack(calc_mass_prf_inf_lst)
    calc_mass_prf_all = np.vstack(calc_mass_prf_all_lst)
    calc_nus = np.concatenate(calc_nu_lst)
    calc_r200m = np.concatenate(calc_r200m_lst)    
    
    return calc_mass_prf_all, calc_mass_prf_orb, calc_mass_prf_inf, calc_nus, calc_r200m.flatten()

def calc_halo_mem(n_ptl):
    # rad, rad_vel, tang_vel each 4bytes and two snaps
    # HIPIDS is 8 bytes 
    # orbit/infall is one byte
    n_bytes = (6 * 4 + 1 + 8) * n_ptl
    
    return n_bytes

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
    ptl_rad, coord_dist = calc_radius(halo_pos[0], halo_pos[1], halo_pos[2], curr_ptl_pos[:,0], curr_ptl_pos[:,1], curr_ptl_pos[:,2], num_new_ptls, box_size)         
    
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
