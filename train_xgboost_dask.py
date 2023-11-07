import numpy as np
import xgboost as xgb
import dask.array as da
import dask.distributed
from dask_cuda import LocalCUDACluster
from dask_ml.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle 
import time  
from data_and_loading_functions import create_directory
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
path_to_hdf5_file = path_to_SPARTA_data + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
path_to_xgboost = config["PATHS"]["path_to_xgboost"]
create_directory(path_to_MLOIS)
create_directory(path_to_snaps)
create_directory(path_to_SPARTA_data)
create_directory(path_to_hdf5_file)
create_directory(path_to_pickle)
create_directory(path_to_calc_info)
create_directory(path_to_xgboost)
snap_format = config["MISC"]["snap_format"]
global prim_only
prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")
global p_snap
p_snap = config.getint("SEARCH","p_snap")
c_snap = config.getint("XGBOOST","c_snap")
model_name = config["XGBOOST"]["model_name"]
radii_splits = config.get("XGBOOST","rad_splits").split(',')
for split in radii_splits:
    model_name = model_name + "_" + str(split)

snapshot_list = [p_snap, c_snap]
global search_rad
search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")
global num_save_ptl_params
num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
##################################################################################################################
# set what the paths should be for saving and getting the data
def get_cluster():

    cluster = LocalCUDACluster()

    client = dask.distributed.Client(cluster)

    return client

if __name__ == "__main__":
    if len(snapshot_list) > 1:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "to" + str(snapshot_list[-1]) + "_" + str(search_rad) + "r200msearch/"
    else:
        specific_save = curr_sparta_file + "_" + str(snapshot_list[0]) + "_" + str(search_rad) + "r200msearch/"

    save_location = path_to_xgboost + specific_save
    
    model_save_location = save_location + "models/" + model_name + "/"
    create_directory(model_save_location)

    curr_model_location = model_save_location + "range_all_" + curr_sparta_file + ".json"
    
    client = get_cluster()
    
    t1 = time.time()
    with open(save_location + "datasets/" + "train_dataset.pickle", "rb") as file:
        dataset = pickle.load(file)

    X = dataset[:,2:]
    y_np = dataset[:,1]
    del dataset
    X = da.from_array(X, chunks=5000)
    y = da.from_array(y_np, chunks=5000)
    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    t2 = time.time()
    
    print("Data loaded:", np.round((t2-t1),2), "seconds", np.round(((t2-t1)/60),2), "min")

    output = xgb.dask.train(
            client,
            {"verbosity": 0, "tree_method": "hist", "objective": "reg:squarederror"},
            dtrain,
            num_boost_round=4,
            evals=[(dtrain, "train")],
        )
    booster = output["booster"]
    booster.save_model(curr_model_location)
    
    t3 = time.time()
    print("Model trained:", np.round((t3-t2),2), "seconds", np.round(((t3-t2)/60),2), "min")
    
    prediction = xgb.dask.predict(client, output, X)
    preds = np.round(prediction.compute())    

    print(classification_report(y_np, preds))



