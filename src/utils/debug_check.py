import os
from pathlib import Path


def check_string(value, var_name):
    if not isinstance(value, str):
        raise TypeError(f"Expected a string for {var_name}, but got {type(value).__name__}")
    return value 

def check_list(value, var_name):
    if not isinstance(value, list):
        raise TypeError(f"Expected a list for {var_name}, but got {type(value).__name__}")
    return value 

def check_or_create_directory(path):
    path = Path(path)  # Convert to a Path object if it's a string

    if not path.exists():
        print(f"Path does not exist: {path}")
        path.mkdir(parents=True, exist_ok=True) 
        print(f"Directory: {path}, created successfully.")