import pandas as pd
import numpy as np
import h5py
import dask.dataframe as dd
import os
from dask import delayed
from colossus.lss import peaks

from .calc_fxns import calc_scal_pos_weight
from .misc_fxns import set_cosmology

# Returns an inputted dataframe with only the halos that fit within the inputted ranges of nus (peak height)
def filter_df_with_nus(df,nus,halo_first,halo_n, nu_splits):    
    # First masks which halos are within the inputted nu ranges
    mask = pd.Series([False] * nus.shape[0])

    for start, end in nu_splits:
        mask[np.where((nus >= start) & (nus <= end))[0]] = True
    
    # Then get the indices of all the particles that belong to these halos and combine them into another mask which returns only the wanted particles    
    halo_n = halo_n[mask]
    halo_first = halo_first[mask]
    halo_last = halo_first + halo_n
 
    use_idxs = np.concatenate([np.arange(start, end) for start, end in zip(halo_first, halo_last)])

    return df.iloc[use_idxs], halo_n, halo_first

def split_orb_inf(data, labels):
    infall = data[np.where(labels == 0)[0]]
    orbit = data[np.where(labels == 1)[0]]
    return infall, orbit

def split_dset_by_mass(halo_first, halo_n, path_to_dataset, curr_dataset):
    with h5py.File((path_to_dataset), 'r') as all_ptl_properties:
        first_prop = True
        for key in all_ptl_properties.keys():
            # only want the data important for the training now in the training dataset
            # dataset now has form HIPIDS, Orbit_Infall, Scaled Radii x num snaps, Rad Vel x num snaps, Tang Vel x num snaps
            if key != "Halo_first" and key != "Halo_n":
                if all_ptl_properties[key].ndim > 1:
                    for row in range(all_ptl_properties[key].ndim):
                        if first_prop:
                            curr_dataset = np.array(all_ptl_properties[key][halo_first:halo_first+halo_n,row])
                            first_prop = False
                        else:
                            curr_dataset = np.column_stack((curr_dataset,all_ptl_properties[key][halo_first:halo_first+halo_n,row])) 
                else:
                    if first_prop:
                        curr_dataset = np.array(all_ptl_properties[key][halo_first:halo_first+halo_n])
                        first_prop = False
                    else:
                        curr_dataset = np.column_stack((curr_dataset,all_ptl_properties[key][halo_first:halo_first+halo_n]))
    return curr_dataset

def split_data_by_halo(client,frac, halo_props, ptl_data, return_halo=False):
    #TODO implement functionality for multiple sims
    halo_first = halo_props["Halo_first"]
    halo_n = halo_props["Halo_n"]

    num_halos = len(halo_first)
    
    split_halo = int(np.ceil(frac * num_halos))
    
    halo_1 = halo_props.loc[:split_halo]
    halo_2 = halo_props.loc[split_halo:]
    
    halo_2.loc[:,"Halo_first"] = halo_2["Halo_first"] - halo_2["Halo_first"].iloc[0]
    
    num_ptls = halo_n.loc[:split_halo].sum()
    
    ptl_1 = ptl_data.compute().iloc[:num_ptls,:]
    ptl_2 = ptl_data.compute().iloc[num_ptls:,:]

    
    scatter_ptl_1 = client.scatter(ptl_1)
    ptl_1 = dd.from_delayed(scatter_ptl_1)
    
    scatter_ptl_2 = client.scatter(ptl_2)
    ptl_2 = dd.from_delayed(scatter_ptl_2)
    
    if return_halo:
        return ptl_1, ptl_2, halo_1, halo_2
    else:
        return ptl_1, ptl_2

# Goes through a folder where a dataset's hdf5 files are stored and reforms them into one pandas dataframe (in order)
def reform_dset_dfs(folder_path):
    hdf5_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.h5'):
            hdf5_files.append(f)
    hdf5_files.sort()

    dfs = []
    for file in hdf5_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_hdf(file_path) 
        dfs.append(df) 
    return pd.concat(dfs, ignore_index=True)
    
# Split a dataframe so that each one is below an inputted maximum memory size
def split_dataframe(df, max_size):
    total_size = df.memory_usage(index=True).sum()
    num_splits = int(np.ceil(total_size / max_size))
    chunk_size = int(np.ceil(len(df) / num_splits))
    print("splitting Dataframe into:",num_splits,"dataframes")
    
    split_dfs = []
    for i in range(0, len(df), chunk_size):
        split_dfs.append(df.iloc[i:i + chunk_size])

    return split_dfs

# Function to process a file in a dataset's folder: combines them all, performs any desired filtering, calculates weights if desired, and calculates scaled position weight
# Also splits the dataframe into smaller dataframes based of inputted maximum memory size
def process_file(folder_path, file_index, ptl_mass, use_z, max_mem, sim_cosmol, filter_nu, nu_splits):
    @delayed
    def delayed_task():
        cosmol = set_cosmology(sim_cosmol)
        # Get all the snap folders being used
        all_snap_fldrs = []
        for snap_fldr in os.listdir(folder_path + "/ptl_info/"):
            if os.path.isdir(os.path.join(folder_path + "/ptl_info/", snap_fldr)):
                all_snap_fldrs.append(snap_fldr)
        
        # Since they are just numbers we can first sort them and then sort them in descending order (primary snaps should always be the largest value)
        all_snap_fldrs.sort()
        all_snap_fldrs.reverse()
        
        # Stack column wise all the snapshots for each particle file
        ptl_df_list = []
        for snap_fldr in all_snap_fldrs:
            ptl_path = f"{folder_path}/ptl_info/{snap_fldr}/ptl_{file_index}.h5"
            ptl_df_list.append(pd.read_hdf(ptl_path))
        ptl_df = pd.concat(ptl_df_list,axis=1)

        halo_path = f"{folder_path}/halo_info/halo_{file_index}.h5"
        halo_df = pd.read_hdf(halo_path)

        # reset indices for halo_first halo_n indexing
        halo_df["Halo_first"] = halo_df["Halo_first"] - halo_df["Halo_first"][0]
        
        # Calculate peak heights for each halo
        nus = np.array(peaks.peakHeight((halo_df["Halo_n"][:] * ptl_mass), use_z))
        
        # Filter by nu and/or by radius
        if filter_nu:
            ptl_df, upd_halo_n, upd_halo_first = filter_df_with_nus(ptl_df, nus, halo_df["Halo_first"], halo_df["Halo_n"], nu_splits)

        # Calculate scale position weight
        scal_pos_weight = calc_scal_pos_weight(ptl_df)

        # If the dataframe is too large split it up
        if ptl_df.memory_usage(index=True).sum() > max_mem:
            ptl_dfs = split_dataframe(ptl_df, max_mem)
        else:
            ptl_dfs = [ptl_df]
        
        return ptl_dfs,scal_pos_weight
    return delayed_task()

# Combines the results of the processing of each file in the folder into one dataframe for the data and list for the scale position weights and an array of weights if desired
def combine_results(results, client):
    # Unpack the results
    ddfs,scal_pos_weights = [], []
    
    for res in results:
        ddfs.extend(res[0])
        scal_pos_weights.append(res[1]) # We append since scale position weight is just a number
            
    all_ddfs = dd.concat([dd.from_delayed(client.scatter(df)) for df in ddfs])

    return all_ddfs, scal_pos_weights

# Combines all the files in a dataset's folder into one dask dataframe and a list for the scale position weights and an array of weights if desired 
def reform_dsets_nested(client, ptl_mass, use_z, max_mem, sim_cosmol, folder_path, prime_snap, file_lim=0, filter_nu=None, nu_splits=None):
    snap_n_files = len(os.listdir(folder_path + "/ptl_info/" + str(prime_snap)+"/"))
    n_files = snap_n_files

    if file_lim > 0:
        n_files = np.min([snap_n_files,file_lim]) 

    delayed_results = []
    for file_index in range(n_files):
        # Create delayed tasks for each file
        delayed_results.append(process_file(
                folder_path, file_index, ptl_mass, use_z,
                max_mem, sim_cosmol, filter_nu, nu_splits))
    
    # Compute the results in parallel
    results = client.compute(delayed_results, sync=True)

    return combine_results(results, client)

