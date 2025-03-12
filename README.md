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

### Particle Data Requirements

Currently only GADGET simulation data is usable by the code as we use Pygadget reader to load particle data

- GADGET simulation particle data with the path to the data indicated with the *snap_path* parameter
- The particle data is expected to be stored in the format "/snapdir_*snap_dir_format*/snapshot_*snap_format*." 
- The formats should be something like {:03d} and supplied in the config file
- **If** you already have run the code **and** have the particle information pickled you can instead just use the *known_snaps* parameter.

### SPARTA Data Requirements

- The halo data is expected to be supplidd from the .hdf5 output file from SPARTA
- The path to this file should be indicated with the *SPARTA_output_path* parameter
- The file's name should be provided with the *curr_sparta_file* parameter
- SPARTA should be run with at least the provided parameters so that all information is present

### Initial Config Params

Before running the code several parameters must be specified to determine what dataset will be generated. The generated dataset can consist currently of either one or two snapshots of data. The dataset contains the following information.

1. halo_first (The starting indices for the halos)
2. halo_n (The number of particles in each halo)
3. HPIDS (The unique ID created for each particle and halo combination)
4. Orbit/Infall (The classification of each particle as orbiting or infalling according to SPARTA)
5. Radius (The radii of the particles)
6. Rad Vel (The radial velocities of the particles)
7. Tang Vel (The tangential velocities of the particles)

Radius, radial velocity, tangential velocity

- 

#### gen_ML_dsets.py


## Training the Model
#### train_xgboost.py
This trains the XGBoost model based off config parameters and requires calc_ptl_props.py to have been run before to generate the datasets needed. This outputs the trained XGBoost model as well as a config file and a readable dictionary that contains information about the parameters and the misclassification rates


## Testing the Model
#### test_xgboost.py
This tests the model on the dataset you decide in the config by making the plots you choose in the config and outputs them under the model and then under the tested dataset 

## Citations

