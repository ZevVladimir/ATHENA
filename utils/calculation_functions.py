import numpy as np
from colossus.halo import mass_so
from colossus.utils import constants

G = constants.G

#calculate distance of particle from halo
def calculate_distance(halo_x, halo_y, halo_z, particle_x, particle_y, particle_z, new_particles, box_size):
    x_dist = particle_x - halo_x
    y_dist = particle_y - halo_y
    z_dist = particle_z - halo_z
    
    coord_diff = np.zeros((new_particles, 3))
    half_box_size = box_size/2
    
    #handles periodic boundary conditions by checking if you were to add or subtract a boxsize would it then be within half a box size of the halo
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
def check_where_r200(my_density, rho_m):
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

def calc_rad_vel(peculiar_vel, particle_dist, coord_sep, halo_r200m, red_shift, hubble_constant):
    
    # Get the corresponding components, distances, and halo v200m for every particle
    v_hubble = np.zeros(particle_dist.size, dtype = np.float32)
    corr_m200m = mass_so.R_to_M(halo_r200m, red_shift, "200c") 
    curr_v200m = calc_v200m(corr_m200m, halo_r200m)
        
    # calculate the unit vector of the halo to the particle  
    rhat = calc_rhat(coord_sep[:,0], coord_sep[:,1], coord_sep[:,2])
    
    # Hubble velocity is the hubble constant times the distance the particle is from the halo
    v_hubble = hubble_constant * particle_dist   
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

def calc_tang_vel(rv, phys_v_comp, rhat):
    rv_comp = rhat * rv[:, np.newaxis] 
    tv_comp = phys_v_comp - rv_comp
    tv = np.linalg.norm(tv_comp, axis=1)
    return tv

def calc_t_dyn(halo_r200m, red_shift):
    halo_m200m = mass_so.R_to_M(halo_r200m, red_shift, "200c")
    curr_v200m = calc_v200m(halo_m200m, halo_r200m)
    t_dyn = (2*halo_r200m)/curr_v200m

    return t_dyn

