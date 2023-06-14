# ML_orbit_infall_project

This is a project with Dr. Diemer that will use data from the nbody simulations run by Dr. Diemer and then processed with SPARTA. The goal is to construct and compare several machine learning models in order to develop one that can accurately predict if a particle is either currently orbiting a dark matter halo or is infalling into one. This will be used to help define the radius of a dark matter halo.

## Current State
The current state of the code is almost at the point where the machine learning can begin.
In find_particle_properties_ML.py data is loaded in from SPARTA and the radial velocity, tangential velocity, and radius of the particles are calculated and then labeled with whether they are orbiting or infalling.
This file takes in a sparta .hdf5 output file as well as the particle data read from a specific snapshot with pygadgetreader. Current functionality allows for easy switching between .hdf5 files as new directories are created and pickle files are stored there. Future work can be done so more pickle files (for radius or radial velocity) can be created to skip those steps as well. 

A tree search is implemented around all the halos and then the particle's radius, radial velocity, and tangential velocity (scaled by the halo's R200m, V200m, V200m respectively) are calculated and then all of this is saved in a new hdf5 file which serves as the training dataset. The search is done by halo mass bin as there is too much data to do at once.

## Next Step
After find_particle_properties_ML.py is fully finished and our training dataset is outputted training will begin on several models. Current plans include Random Forest, XG-boosted forest, and neural networks.
