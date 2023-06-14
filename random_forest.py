import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
import seaborn as sns
import h5py

save_location = "/home/zvladimi/ML_orbit_infall_project/calculated_info/"
curr_snapshot = "190"

# with h5py.File((save_location + "all_particle_properties" + curr_snapshot + ".hdf5"), 'r') as all_particle_properties:
#     print(all_particle_properties.keys())
#     scaled_radii = np.array(all_particle_properties["Scaled_radii"][:])
#     radial_vel = np.array(all_particle_properties["Radial_vel"][:])
#     tang_vel = np.array(all_particle_properties["Tangential_vel"][:])
    
    
import pandas as pd
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
import time 

t1 = time.time()
X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=0)
classifier = XGBClassifier()
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)
classification = classification_report(y_test, predictions)
print(classification)
t2 = time.time()
print(t2 - t1, "seconds")

t3 = time.time()
X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=0)
classifier = XGBClassifier(tree_method='gpu_hist')
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)
classification = classification_report(y_test, predictions)
print(classification)
t4 = time.time()
print(t4 - t3, "seconds")