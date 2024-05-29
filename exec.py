import subprocess
import configparser
import os

config = configparser.ConfigParser()
config.read(os.environ.get('PWD') + "/config.ini")
calc_ptl_prop_enable=config.getboolean("EXEC","calc_ptl_prop_enable")
train_xgboost_enable=config.getboolean("EXEC","train_xgboost_enable")
test_xgboost_enable=config.getboolean("EXEC","test_xgboost_enable")

if calc_ptl_prop_enable:   
    subprocess.run(["python3", "calc_ptl_props.py"],check=True)

if train_xgboost_enable:
    subprocess.run(["python3", "train_xgboost.py"],check=True)

if test_xgboost_enable:
    subprocess.run(["python3", "test_xgboost.py"],check=True)

