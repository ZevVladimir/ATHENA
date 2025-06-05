import time
from contextlib import contextmanager
import os
import numpy as np
import re
from pygadgetreader import readheader
from colossus.cosmology import cosmology

@contextmanager
def timed(txt):
    print("Starting: " + txt)
    t0 = time.time()
    yield
    t1 = time.time()
    time_s = t1 - t0
    time_min = time_s / 60
    
    print("Finished: %s time: %.5fs, %.2f min\n" % (txt, time_s, time_min))
    
def create_directory(path):
    os.makedirs(path,exist_ok=True)

def clean_dir(path):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files at location:",path)
        
# Depairs the hipids into (pids, halo_idxs) We use np.vectorize because depair returns two values and we want that split in two
def depair_np(z):
    """
    Modified from https://github.com/perrygeo/pairing to use numpy functions
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    w = np.floor((np.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = (z - t).astype(int)
    x = (w - y).astype(int)
    # assert z != pair(x, y, safe=False):
    return x, y

# Obtains the highest number snapshot in the given folder path
# We can't just get the total number of folders as there might be snapshots missing
def get_num_snaps(path):
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    numbers = [int(re.search(r'\d+', d).group()) for d in folders if re.search(r'\d+', d)]
    max_number = max(numbers, default=None)
    return (max_number+1)

def find_closest_z_snap(value,snap_loc,snap_dir_format,snap_format):
    tot_num_snaps = get_num_snaps(snap_loc)
    all_z = np.ones(tot_num_snaps) * -1000
    for i in range(tot_num_snaps):
        # Sometimes not all snaps exist
        if os.path.isdir(snap_loc + "snapdir_" + snap_dir_format.format(i)):
            all_z[i] = readheader(snap_loc + "snapdir_" + snap_dir_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift')

    idx = (np.abs(all_z - value)).argmin()
    return idx, all_z[idx]

# Returns the path of the rockstar file that has the closest redshift to the inputted value
def find_closest_a_rstar(z,rockstar_loc):
    all_a = []
    for filename in os.listdir(rockstar_loc):
        match = re.search(r"hlist_(\d+\.\d+)\.list", filename)
        if match:
            a_val = float(match.group(1))
            all_a.append(a_val)

    idx = (np.abs(all_a - 1/(1+z))).argmin()
    print(rockstar_loc + "/hlist_" + str(all_a[idx]) + ".list")
    return rockstar_loc + "/hlist_" + str(all_a[idx]) + ".list"

def find_closest_snap(value, cosmol, snap_loc, snap_dir_format, snap_format):
    tot_num_snaps = get_num_snaps(snap_loc)
    all_times = np.ones(tot_num_snaps) * -1000
    for i in range(tot_num_snaps):
        # Sometimes not all snaps exist
        if os.path.isdir(snap_loc + "snapdir_" + snap_dir_format.format(i)):
            all_times[i] = cosmol.age(readheader(snap_loc + "snapdir_" + snap_dir_format.format(i) + "/snapshot_" + snap_format.format(i), 'redshift'))
    idx = (np.abs(all_times - value)).argmin()
    return idx
    
def conv_halo_id_spid(my_halo_ids, sdata, snapshot):
    sparta_idx = np.zeros(my_halo_ids.shape[0], dtype = np.int32)
    for i, my_id in enumerate(my_halo_ids):
        sparta_idx[i] = int(np.where(my_id == sdata['halos']['id'][:,snapshot])[0])
    return sparta_idx    

def parse_ranges(ranges_str):
    ranges = []
    for part in ranges_str.split(','):
        start, end = map(float, part.split('-'))
        ranges.append((start, end))
    return ranges

def create_nu_string(nu_list):
    return '_'.join('-'.join(map(str, tup)) for tup in nu_list)

# From the input simulation name extract the simulation name (ex: cbol_l0063_n0256) and the SPARTA hdf5 output name (ex: cbol_l0063_n0256_4r200m_1-5v200m)
def split_sparta_hdf5_name(sim):
    # Get just the sim name of the form cbol_ (or cpla_) then the size of the box lxxxx and the number of particles in it nxxxx
    sim_pat = r"cbol_l(\d+)_n(\d+)"
    match = re.search(sim_pat, sim)
    if not match:
        sim_pat = r"cpla_l(\d+)_n(\d+)"
        match = re.search(sim_pat,sim)
        
    if match:
        sim_name = match.group(0)
           
    # now get the full name that includes the search radius in R200m and the velocity limit in v200m
    sim_search_pat = sim_pat + r"_(\d+)r200m_(\d+)v200m"
    name_match = re.search(sim_search_pat, sim)
    
    # also check if there is a decimal for v200m
    if not name_match:
        sim_search_pat = sim_pat + r"_(\d+)r200m_(\d+)-(\d+)v200m"
        name_match = re.search(sim_search_pat, sim)
    
    if name_match:
        search_name = name_match.group(0)
        
    if not name_match and not match:
        print("Couldn't read sim name correctly:",sim)
        print(match)
    
    return sim_name, search_name

def set_cosmology(sim_cosmol):
    if sim_cosmol == "planck13-nbody":
        cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
    else:
        #TODO add a try except statement
        cosmol = cosmology.setCosmology(sim_cosmol)
    return cosmol
