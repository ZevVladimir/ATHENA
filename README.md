# Current State as of 05//2024

## Data Requirements
You need the particle data from a GADGET simulation and the .hdf5 output file from SPARTA. We used the Erebos suite of simulations for our work.

## Python Dependencies
- I make use of a customized shap fork that is also publicly available at https://github.com/ZevVladimir/shap. This is done to just allow for ease of adjusting different fontsizes as well as adjusting certain features in the plots that I want hidden or not. This should only impact code run in make_shap_plots.py. 
- https://docs.rapids.ai/install/ 
- Use the requirements.txt
- pygadgetreader: https://github.com/jveitchmichaelis/pygadgetreader
- SPARTA: https://bdiemer.bitbucket.io/sparta/ 

## Creating the Datasets

### Snapshot Data Requirements \[SNAP_DATA\]

Currently only GADGET simulation data is usable by the code as we use Pygadget reader to load particle data

- GADGET simulation particle data with the path to the data indicated with the *snap_path* parameter
- The particle data is expected to be stored in the format "/snapdir_*snap_dir_format*/snapshot_*snap_format*." 
- The formats should be something like {:03d} and supplied in the config file
- **If** you already have run the code **and** have the particle information pickled you can instead just use the *known_snaps* parameter.

### SPARTA Data Requirements \[SPARTA_DATA\]

- The halo data is expected to be supplidd from the .hdf5 output file from SPARTA
- The path to this file should be indicated with the *SPARTA_output_path* parameter
- The file's name should be provided with the *curr_sparta_file* parameter
- SPARTA should be run with at least the provided parameters so that all information is present

### Initial \[SEARCH\] Config Parameters

Before running the code several parameters must be specified to determine what dataset will be generated. The generated dataset can consist currently of only two snapshots of data. The dataset contains the following information split into two storage types. The information is stored in at least one pandas dataframe saved as an .h5 file. The number of dataframes is determined by the *sub_dset_mem_size* parameter. Set this to be the maximum number of bytes a file should hold.
 
Currently the way to determine which snapshots are used is done in the following way:
- The primary snapshot is determined with the snapshot that has the redshift closest to the *p_red_shift* parameter
- The secondary snapshot is determined by going *t_dyn_step* dynamical time steps (see paper for definition) back in time from the primary snapshot

The search for particles looks within the *search_radius* (in multiples of R200m) of each halo's center. We choose a measure of R200m as this allows for generalizability to all sizes of halos.

### Running the code: gen_ML_dsets.py

After the \[SEARCH\], \[SNAP_DATA\], and \[SPARTA_DATA\] parameters are set (and potentially \[MISC\] parameters as well) you are ready to create the datasets. This is done by simply running the python code: `python3 ~/src/gen_ML_dsets.py`

### Saved Information

The code will generate several .h5 files of the saved information within the specified output location *ML_dset_path* in the subdirectory of ~/*ML_dset_path*/ + *curr_sparta_file*_\<primary snap number\>to\<secondary snap number\>/ and then there will be a Train/ and Test/ folder each of which contain folders for halo information and particle information.

1. Halo Information
    1. Halo_first (The starting indices for the halos)
    2. Halo_n (The number of particles in each halo)
    3. Halo_indices (The indices of the halos locations in SPARTA's arrays)
2. Particle Information
    1. HIPIDS (The unique ID created for each particle and halo combination)
    2. Orbit_infall (The classification of each particle as orbiting or infalling according to SPARTA)
    3. p_Scaled_radii (The radii of the particles at the primary snapshot)
    4. p_Radial_vel (The radial velocities of the particles at the primary snapshot)
    5. p_Tangential_vel (The tangential velocities of the particles at the primary snapshot)
    6. c_Scaled_radii (The radii of the particles at the secondary snapshot)
    7. c_Radial_vel (The radial velocities of the particles at the secondary snapshot)
    8. c_Tangential_vel (The tangential velocities of the particles at the secondary snapshot)
    9. p_phys_vel (The physical velocities of the particles at the primary snapshot)






## Training the Model
#### train_xgboost.py
This trains the XGBoost model based off config parameters and requires calc_ptl_props.py to have been run before to generate the datasets needed. This outputs the trained XGBoost model as well as a config file and a readable dictionary that contains information about the parameters and the misclassification rates


## Testing the Model
#### test_xgboost.py
This tests the model on the dataset you decide in the config by making the plots you choose in the config and outputs them under the model and then under the tested dataset 

## Citations

