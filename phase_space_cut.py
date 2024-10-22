import os
import dask
from colossus.cosmology import cosmology
from utils.ML_support import get_combined_name
from utils.data_and_loading_functions import *
from utils.update_vis_fxns import *
from utils.ML_support import *

config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")
test_sims = json.loads(config.get("XGBOOST","test_sims"))
eval_datasets = json.loads(config.get("XGBOOST","eval_datasets"))
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
sim_cosmol = config["MISC"]["sim_cosmol"]

def get_alpha(x,a,alpha_inf):
    return alpha_inf * (x / (a + x))

def get_rh(r_h_p,r_h_s,M_orb,M_p):
    return r_h_p * (M_orb / M_p)**r_h_s

@delayed
def get_alpha_inf(alpha_inf_p,alpha_inf_s,M_orb,M_p):
    return alpha_inf_p * (M_orb / M_p)**alpha_inf_s

@delayed
def create_orb_prf(x,A,a,alpha_inf,halo_first,halo_n):
    start = halo_first 
    end = halo_first + halo_n
    curr_x = x[start:end]
    alpha = get_alpha(curr_x,a,alpha_inf)
    return A * ((curr_x/a)**-alpha) * np.exp(-((curr_x**2)/2))

@delayed
def calc_morb(halo_first,halo_n,y,mass):
    start = halo_first 
    end = halo_first + halo_n
    return da.from_array(y["Orbit_infall"].loc[start:end].sum() * mass)

@delayed
def calc_r_rh(X,r_h,halo_first,halo_n):
    start = halo_first 
    end = halo_first + halo_n
    return da.from_array(X["p_Scaled_radii"].loc[start:end].values / r_h)

if __name__ == '__main__':
    if use_gpu:
        mp.set_start_method("spawn")

    if not use_gpu and on_zaratan:
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            print("SLURM_CPUS_PER_TASK is not defined.")
        if use_gpu:
            initialize(local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        else:
            initialize(nthreads = cpus_per_task, local_directory = "/home/zvladimi/scratch/MLOIS/dask_logs/")
        print("Initialized")
        client = Client()
        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        login_node_address = "zvladimi@login.zaratan.umd.edu" # Change this to the address/domain of your login node

        logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
    else:
        client = get_CUDA_cluster()
        
    with timed("Setup"):
        if sim_cosmol == "planck13-nbody":
            cosmol = cosmology.setCosmology('planck13-nbody',{'flat': True, 'H0': 67.0, 'Om0': 0.32, 'Ob0': 0.0491, 'sigma8': 0.834, 'ns': 0.9624, 'relspecies': False})
        else:
            cosmol = cosmology.setCosmology(sim_cosmol) 

        feature_columns = ["p_Scaled_radii","p_Radial_vel","p_Tangential_vel","c_Scaled_radii","c_Radial_vel","c_Tangential_vel"]
        target_column = ["Orbit_infall"]

        model_comb_name = get_combined_name(model_sims)
        model_dir = model_type + "_" + model_comb_name + "nu" + nu_string 

        model_name =  model_dir + model_comb_name
                
        model_save_loc = path_to_xgboost + model_comb_name + "/" + model_dir + "/"
        gen_plot_save_loc = model_save_loc + "plots/"

        try:
            bst = xgb.Booster()
            bst.load_model(model_save_loc + model_name + ".json")
            print("Loaded Model Trained on:",model_sims)
        except:
            print("Couldn't load Booster Located at: " + model_save_loc + model_name + ".json")

        for curr_test_sims in test_sims:
            test_comb_name = get_combined_name(curr_test_sims) 
            for dset_name in eval_datasets:
                plot_loc = model_save_loc + dset_name + "_" + test_comb_name + "/plots/"
                create_directory(plot_loc)
                
                halo_files = []
                halo_dfs = []
                if dset_name == "Full":    
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + "Train" + "/halo_info/"))
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + "Test" + "/halo_info/"))
                else:
                    for sim in curr_test_sims:
                        halo_dfs.append(reform_df(path_to_calc_info + sim + "/" + dset_name + "/halo_info/"))

                halo_df = pd.concat(halo_dfs)
                
                data,scale_pos_weight = load_data(client,curr_test_sims,dset_name,limit_files=False)
                X_df = data[feature_columns]
                y_df = data[target_column]
                
                all_masses = []
                halo_first = halo_df["Halo_first"]
                halo_n = halo_df["Halo_n"]
                # Know where each simulation's data starts in the stacked dataset based on when the indexing starts from 0 again
                sim_splits = np.where(halo_first == 0)[0]
                # if there are multiple simulations, to correctly index the dataset we need to update the starting values for the 
                # stacked simulations such that they correspond to the larger dataset and not one specific simulation
                if len(curr_test_sims) != 1:
                    for i in range(1,len(curr_test_sims)):
                        if i < len(curr_test_sims) - 1:
                            halo_first[sim_splits[i]:sim_splits[i+1]] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
                        else:
                            halo_first[sim_splits[i]:] += (halo_first[sim_splits[i]-1] + halo_n[sim_splits[i]-1])
                            
                for i,sim in enumerate(curr_test_sims):
                    sparta_name, sparta_search_name = split_calc_name(sim)
                    # find the snapshots for this simulation
                    snap_pat = r"(\d+)to(\d+)"
                    match = re.search(snap_pat, sim)
                    if match:
                        curr_snap_list = [match.group(1), match.group(2)] 
                    
                    with open(path_to_calc_info + sim + "/config.pickle", "rb") as file:
                        config_dict = pickle.load(file)
                        
                        curr_z = config_dict["p_snap_info"]["red_shift"][()]
                        curr_snap_dir_format = config_dict["snap_dir_format"]
                        curr_snap_format = config_dict["snap_format"]
                        new_p_snap, curr_z = find_closest_z(curr_z,path_to_snaps + sparta_name + "/",curr_snap_dir_format,curr_snap_format)
                        p_scale_factor = 1/(1+curr_z)
                        
                    with h5py.File(path_to_SPARTA_data + sparta_name + "/" + sparta_search_name + ".hdf5","r") as f:
                        dic_sim = {}
                        grp_sim = f['simulation']

                        for attr in grp_sim.attrs:
                            dic_sim[attr] = grp_sim.attrs[attr]
                    
                    all_red_shifts = dic_sim['snap_z']
                    p_sparta_snap = np.abs(all_red_shifts - curr_z).argmin()
                    
                    halos_pos, halos_r200m, halos_id, halos_status, halos_last_snap, parent_id, ptl_mass = load_or_pickle_SPARTA_data(sparta_search_name, p_scale_factor, curr_snap_list[0], p_sparta_snap)
                    
                    delay_curr_morb = [calc_morb(halo_first.iloc[i],halo_n.iloc[i],y_df,ptl_mass) for i in range(10)]     
                    curr_morb_res = client.compute(delay_curr_morb, sync=True)
                    curr_morb = da.stack(curr_morb_res) 

                    if i == 0:
                        M_orb = curr_morb
                    else:
                        M_orb = da.concatenate([M_orb,curr_morb])
                    
    a = 0.0337
    alpha_inf_p = 2.018
    alpha_inf_s = -0.05
    r_h_s = 0.226
    r_h_p = 840.3
    M_p = 10e14
    A = 1
    
    
    alpha_inf = get_alpha_inf(alpha_inf_p,alpha_inf_s,M_orb,M_p)
    r_h = get_rh(r_h_p,r_h_s,M_orb,M_p)
    
    delayed_results = [calc_r_rh(X_df,r_h[i],halo_first[i],halo_n[i]) for i in range(10)]
    results = client.compute(delayed_results, sync=True)    
    x = da.concatenate(results)
    
    scatter_x = client.scatter(x)
    
    delayed_results = [create_orb_prf(scatter_x,A,a,alpha_inf[i],halo_first[i],halo_n[i]) for i in range(10)]
    results = client.compute(delayed_results, sync=True)
    for i in range(10):
        curr_res = results[i].compute()
        print(curr_res.shape)
    
    client.close()