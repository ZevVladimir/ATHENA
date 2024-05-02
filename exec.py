import subprocess
import configparser
import os

config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
calc_ptl_prop_enable=config.getboolean("EXEC","calc_ptl_prop_enable")
create_train_dataset_enable=config.getboolean("EXEC","create_train_dataset_enable")
train_xgboost_enable=config.getboolean("EXEC","train_xgboost_enable")
test_xgboost_enable=config.getboolean("EXEC","test_xgboost_enable")

if calc_ptl_prop_enable:   
    subprocess.run(["python3", "calc_ptl_props.py"],check=True)

if create_train_dataset_enable:
    subprocess.run(["python3", "create_train_dataset.py"],check=True)

if train_xgboost_enable:
    subprocess.run(["python3", "train_xgboost_dask.py"],check=True)

if test_xgboost_enable:
    subprocess.run(["python3", "test_xgboost_dask.py"],check=True)


