# ML_orbit_infall_project

This is a project with Dr. Diemer that will use data from the IllustrisTNG 300 dark matter simulation. The goal is to construct a machine learning model that can predict if a particle is either currently orbiting a dark matter halo or is infalling into one.

This repository will contain files to adjust the data outputted by SPARTA into two training arrays of radius and velocities of particles. These arrays will then be used to train and test a machine learning model.

The file load_save_data.py as it says loads and saves all data as numpy arrays. From the SPARTA hdf5 file it gets all the halo information needed. From pygadget reader it gets all the particle information.

The file adjust_data_for_ML.py takes the numpy arrays and then assigns particles to the halo they belong to and calculates their distance from the halo in addition to each particle's tangential and radial velocity.
