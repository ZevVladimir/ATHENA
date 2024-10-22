from utils.ML_support import *
from utils.update_vis_fxns import plot_halo_slice
import os
import json
import multiprocessing as mp
import configparser
from dask.distributed import Client


config = configparser.ConfigParser()
config.read(os.getcwd() + "/config.ini")
on_zaratan = config.getboolean("MISC","on_zaratan")
use_gpu = config.getboolean("MISC","use_gpu")
model_sims = json.loads(config.get("XGBOOST","model_sims"))
model_type = config["XGBOOST"]["model_type"]
nu_splits = config["XGBOOST"]["nu_splits"]

if use_gpu:
    
    from sklearn.metrics import make_scorer
    import dask_ml.model_selection as dcv
    import cudf
elif not use_gpu and on_zaratan:
    from dask_mpi import initialize
    from mpi4py import MPI
    from distributed.scheduler import logger
    import socket
    #from dask_jobqueue import SLURMCluster
elif not on_zaratan:
    from dask_cuda import LocalCUDACluster
    
def create_nu_string(nu_list):
    return '_'.join('-'.join(map(str, tup)) for tup in nu_list)

if __name__ == "__main__":
    
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
        
    path = "/home/zvladimi/MLOIS/calculated_info/cbol_l0063_n0256_4r200m_1-5v200m_190to166/Test/"

    halo_df = reform_datasets(path + "halo_info/")
    ptl_df = reform_datasets(path + "ptl_info/")
    
    halo_num = 287
    
    halo_first = halo_df["Halo_first"][halo_num]
    halo_n = halo_df["Halo_n"][halo_num]
    
    halo_pos = ptl_df["p_Scaled_radii"][halo_first:halo_first+halo_n]
    halo_labels = ptl_df["Orbit_infall"][halo_first:halo_first+halo_n]
    
    model_comb_name = get_combined_name(model_sims) 

    nu_string = create_nu_string(nu_splits)
    model_dir = model_type + "_" + model_comb_name + "nu" + nu_string 

    model_name =  model_dir + model_comb_name
    model_save_loc = path_to_xgboost + model_comb_name + "/" + model_dir + "/"
    gen_plot_save_loc = model_save_loc + "plots/"
    
    plot_halo_slice(halo_pos,halo_labels,gen_plot_save_loc)