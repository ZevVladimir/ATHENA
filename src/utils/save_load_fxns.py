import pickle
import joblib
import h5py 
import os
import numpy as np

import dask.dataframe as dd
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta
from functools import reduce
import ast
import configparser

from .misc_fxns import split_sparta_hdf5_name, create_directory, find_closest_a_rstar, find_closest_snap, timed
from .dset_fxns import reform_dsets_nested

def parse_value(value):
    """Convert value to appropriate type (list, int, float, bool, str)."""
    try:
        return ast.literal_eval(value)  # Safely evaluates lists, ints, floats, etc.
    except (ValueError, SyntaxError):
        return value  # Keep as string if eval fails

# Load the config file into one dictionary separated by the sections as indicated in the config file
def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {key: parse_value(value) for key, value in config[section].items()}
    
    return config_dict
##################################################################################################################
# LOAD CONFIG PARAMETERS
config_params = load_config(os.getcwd() + "/config.ini")

pickled_path = config_params["PATHS"]["pickled_path"]
ML_dset_path = config_params["PATHS"]["ml_dset_path"]

SPARTA_output_path = config_params["SPARTA_DATA"]["sparta_output_path"]

pickle_data = config_params["MISC"]["pickle_data"]
##################################################################################################################

def save_pickle(data, path):
    with open(path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)

def load_pickle(path):
    if os.path.isfile(path):
        with open(path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    else:
        raise FileNotFoundError
    
# joblib is better for large np arrays
def save_joblib(data, path):
    joblib.dump(data,path)
    
def load_joblib(path):
    if os.path.isfile(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError

def load_ptl_param(sparta_name, param_name, snap, snap_path):
    # save to folder containing pickled data to be accessed easily later
    file_path = pickled_path + str(snap) + "_" + str(sparta_name) + "/" + param_name + "_" + str(snap) + ".pickle" 
    create_directory(pickled_path + str(snap) +  "_" + str(sparta_name) + "/")
    
    # check if the file has already been pickled if so just load it
    try:
        ptl_param = load_pickle(file_path)
    except FileNotFoundError:
        ptl_param = readsnap(snap_path, param_name, 'dm')
        #TODO check if ptl_param is np array and if so use joblib instead to save
        save_pickle(ptl_param,file_path)

    return ptl_param

def load_SPARTA_data(sparta_HDF5_path, param_path_list, sparta_name, pickle_data=False):
    create_directory(pickled_path + str(sparta_name) + "/")
    
    reload_sparta = False
    param_dict = {}

    all_save_names = []
    
    for param_path in param_path_list:
        save_name = "_".join(map(str, param_path))
        all_save_names.append(save_name)
        try:
            param = load_joblib(pickled_path + str(sparta_name) + "/" + save_name + ".pickle")
        except FileNotFoundError:
            if not reload_sparta:
                sparta_output = sparta.load(filename=sparta_HDF5_path, log_level= 0)
                reload_sparta = True
        
            param = reduce(lambda dct, key: dct[key], param_path, sparta_output)
            if pickle_data:
                save_joblib(param,pickled_path + str(sparta_name) +  "/" + save_name + ".pickle")

        param_dict[save_name] = param

    return param_dict,all_save_names

def load_RSTAR_data(rockstar_loc, param_list, curr_z):
    rstar_file_loc = find_closest_a_rstar(curr_z, rockstar_loc)
    
    param_data = {param: [] for param in param_list}
    col_index_map = {}

    with open(rstar_file_loc, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Only parse the column names once
                if not col_index_map:
                    header = line[1:].strip().split()
                    for i, entry in enumerate(header):
                        name = entry.split('(')[0]
                        col_index_map[name] = i
                continue  # Skip all header lines


            # Split data line and collect requested values
            parts = line.split()
            for param in param_list:
                idx = col_index_map.get(param)
                if idx is not None and len(parts) > idx:
                    param_data[param].append(float(parts[idx]))

    for param in param_list:
        param_data[param] = np.array(param_data[param])

    return param_data 

def get_comp_snap_info(t_dyn, t_dyn_step, cosmol, p_red_shift, all_sparta_z, snap_dir_format, snap_format, snap_path):
    c_snap_dict = {}
    # calculate one dynamical time ago and set that as the comparison snap
    curr_time = cosmol.age(p_red_shift)
    past_time = curr_time - (t_dyn_step * t_dyn)
    c_snap = find_closest_snap(past_time, cosmol, snap_path, snap_dir_format, snap_format)
    
    c_snap_dict["ptl_snap"] = c_snap
    
    # switch to comparison snap
    c_snap_path = snap_path + "/snapdir_" + snap_dir_format.format(c_snap) + "/snapshot_" + snap_format.format(c_snap)
        
    # get constants from pygadgetreader
    c_red_shift = readheader(c_snap_path, 'redshift')
    c_sparta_snap = np.abs(all_sparta_z - c_red_shift).argmin()
    c_snap_dict["sparta_snap"] = c_sparta_snap
    
    print("Complementary snapshot:", c_snap, "Complementary redshift:", c_red_shift)
    print("Corresponding SPARTA loc:", c_sparta_snap, "SPARTA redshift:",all_sparta_z[c_sparta_snap])

    c_scale_factor = 1/(1+c_red_shift)
    c_rho_m = cosmol.rho_m(c_red_shift)
    c_hubble_const = cosmol.Hz(c_red_shift) * 0.001 # convert to units km/s/kpc
    
    c_snap_dict["red_shift"] = c_red_shift
    c_snap_dict["scale_factor"] = c_scale_factor
    c_snap_dict["rho_m"] = c_rho_m
    c_snap_dict["hubble_const"] = c_hubble_const

    return c_snap_dict

# Returns a simulation's mass used and the redshift of the primary snapshot
def load_sim_mass_pz(sim,config_params):
    sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
    
    curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" +  sparta_search_name + ".hdf5"    
    
    with h5py.File(curr_sparta_HDF5_path,"r") as f:
        dic_sim = {}
        grp_sim = f['simulation']
        for f in grp_sim.attrs:
            dic_sim[f] = grp_sim.attrs[f]
    
    p_red_shift = config_params["all_snap_info"]["prime_snap_info"]["red_shift"]
    
    param_paths = [["simulation","particle_mass"]]
            
    sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, pickle_data=pickle_data)
    ptl_mass = sparta_params[sparta_param_names[0]]
    
    return ptl_mass, p_red_shift

# Reconstruct SPARTA's mass profiles and stack them together for a list of sims
def load_sparta_mass_prf(sim_splits,all_idxs,use_sims,ret_r200m=False):                
    mass_prf_all_list = []
    mass_prf_1halo_list = []
    all_r200m_list = []
    all_masses = []
    
    for i,sim in enumerate(use_sims):
        # Get the halo indices corresponding to this simulation
        if i < len(use_sims) - 1:
            use_idxs = all_idxs[sim_splits[i]:sim_splits[i+1]]
        else:
            use_idxs = all_idxs[sim_splits[i]:]
        
        sparta_name, sparta_search_name = split_sparta_hdf5_name(sim)
        
        with open(ML_dset_path + sim + "/dset_params.pickle", "rb") as file:
            dset_params = pickle.load(file)
            
        p_sparta_snap = dset_params["all_snap_info"]["prime_snap_info"]["sparta_snap"]
        
        curr_sparta_HDF5_path = SPARTA_output_path + sparta_name + "/" + sparta_search_name + ".hdf5"      
        
        param_paths = [["simulation","particle_mass"],["anl_prf","M_all"],["anl_prf","M_1halo"],["halos","R200m"],["config","anl_prf","r_bins_lin"]]
        sparta_params, sparta_param_names = load_SPARTA_data(curr_sparta_HDF5_path, param_paths, sparta_search_name, pickle_data=pickle_data)
        ptl_mass = sparta_params[sparta_param_names[0]]

        mass_prf_all_list.append(sparta_params[sparta_param_names[1]][use_idxs,p_sparta_snap,:])
        mass_prf_1halo_list.append(sparta_params[sparta_param_names[2]][use_idxs,p_sparta_snap,:])
        all_r200m_list.append(sparta_params[sparta_param_names[3]][use_idxs,p_sparta_snap])

        all_masses.append(ptl_mass)

    mass_prf_all = np.vstack(mass_prf_all_list)
    mass_prf_1halo = np.vstack(mass_prf_1halo_list)
    all_r200m = np.concatenate(all_r200m_list)
    
    # We assume bins are the same from each sparta simulation
    bins = sparta_params[sparta_param_names[4]]
    bins = np.insert(bins, 0, 0)

    if ret_r200m:
        return mass_prf_all,mass_prf_1halo,all_masses,bins,all_r200m
    else:
        return mass_prf_all,mass_prf_1halo,all_masses,bins

# Loads all the data for the inputted list of simulations into one dataframe. Finds the scale position weight for the dataset and any adjusted weighting for it if desired
def load_ML_dsets(client, sims, dset_name, sim_cosmol, prime_snap, file_lim=0, filter_nu=False, nu_splits=None):
    dask_dfs = []
    all_scal_pos_weight = []
        
    for sim in sims:
        with open(ML_dset_path + sim + "/dset_params.pickle","rb") as f:
            dset_params = pickle.load(f)
        # Get mass and redshift for this simulation
        ptl_mass, use_z = load_sim_mass_pz(sim,dset_params)
        max_mem = int(np.floor(dset_params["HDF5 Mem Size"] / 2))
        
        if dset_name == "Full":
            datasets = ["Train", "Test"]
        else:
            datasets = [dset_name]

        for dataset in datasets:
            with timed(f"Reformed {dataset} Dataset: {sim}"): 
                dataset_path = f"{ML_dset_path}{sim}/{dataset}"
                ptl_ddf,sim_scal_pos_weight = reform_dsets_nested(client,ptl_mass,use_z,max_mem,sim_cosmol,dataset_path,prime_snap,file_lim=file_lim,filter_nu=filter_nu,nu_splits=nu_splits)  
                all_scal_pos_weight.append(sim_scal_pos_weight)
                dask_dfs.append(ptl_ddf)
                    
    all_scal_pos_weight = np.average(np.concatenate([np.array(sublist).flatten() for sublist in all_scal_pos_weight]))
    act_scale_pos_weight = np.average(all_scal_pos_weight)

    all_dask_dfs = dd.concat(dask_dfs)
    
    return all_dask_dfs,act_scale_pos_weight
