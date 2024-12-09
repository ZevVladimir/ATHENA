# Current State as of 05//2024

## Data Requirements
You need the particle data from a GADGET simulation and the .hdf5 output file from SPARTA. We used the Erebos suite of simulations for our work.

## Python Dependencies
- I make use of a customized shap fork that is also publicly available. This is done to just allow for ease of adjusting different fontsizes as well as adjusting certain features in the plots that I want hidden or not. This should only impact code run in make_shap_plots.py. 
- https://docs.rapids.ai/install/ 
- Use the requirements.txt
- pygadgetreader: https://github.com/jveitchmichaelis/pygadgetreader
- SPARTA: https://bdiemer.bitbucket.io/sparta/ 

## Running the code

### The config file
This is the heart of the code and (most) parameters that you would want to adjust are present here.

#### Paths

- path_to_MLOIS: the path to this code
- path_to_pickle: the path to where pickled data is stored to enable faster runs after the inital one
- path_to_calc_info: where the outputs of calc_ptl_props.py go
- path_to_xgboost: where the outputs of the XGBoost model training and testing goes
- path_to_pygadgetreader: where you installed pygadgetreader
- path_to_sparta: where you installed sparta
- path_to_snaps: where the particle data is stored
- path_to_SPARTA_data: where the SPARTA output file is stored

#### MISC
- curr_sparta_file: The full name of the SPARTA output file that is being used. Format should be [simulation type used (ex cbol)]_l[size of sim]_n[number of particles]_[search radius multiple]R200m_[mutiple of V200m SPARTA cutoff]v200m
- snap_dir_format: what format the particle data snapshot directories are saved as. Ones I used are either {:04d} or {:03d}
- snap_format: Same as above but for the individial snapshot files
- random_seed: Seed used for random elements of the code
- on_zaratan: If running on zaratan runs things differently but this is very dependent on machine and subject to change
- use_gpu: Whether GPUs should be used for XGBoost stuff
- sim_cosmol: what cosmology should be used. This is used to set colossus cosmology and so anything default in there or planck13_nbody is accepted

#### SEARCH
- reset: SET LEVELS for calc_ptl_props.py (no impact on train_xgboost.py) **0**: no reset will just run from the beginning or continue from where the .last run left off. **1**: Removes the calculated information (ptl_info and halo_info) and redoes all the calculations. **2**: Same as 1 and removes the the particle trees and the number of particles per halo. **3**: Same as 2 and removes all pickled data about the halos and particles from SPARTA/the simulation
- prim_only: Whether to only use the primary snapshot or to also calculate the past snapshot
- t_dyn_step: How many dynamical time steps ago the past shot should be
- p_red_shift: Finds the closest snapshot to this redshift for the primary snapshot
- search_rad: How far from each halo (in multiples of R200m) to search for particles
- total_num_snaps: How many snapshots of simulation data are there
- num_save_ptl_params: How many things are being saved about the particles shouldn't be adjusted unless you've made changes to calc_ptl_props.py. If for calc_ptl_props 7: halo_first, halo_n, HPIDS, Orbit/Infall, Radius, Rad Vel, Tang Vel. If for morb_cat 6: Halo_ID, Halo_pos, Halo_vel, M_orb, M200m, R200m
- hdf5_mem_size: How big each output hdf5 file should be
- chunk_size: Chunk size for multiprocessing

#### XGBOOST
- retrain: RETRAIN LEVELS for train_xgboost.py. **0**: no retraining will just predict with current model (if there is one otherwise will train a new one) on test dataset. **1**: will retrain model but use old parameters (if they exist). **2**: will retrain model with new parameters
- feature_columns: Which features should be included for XGBoost
- target_column: Which features should be the target column
- file_lim: How many files per simulation should be used in training
- model_sims: Which simulations should be used in the model training
- test_halos_ratio: What percentage of halos should be used for testing as a decimal
- test_sims: Which simulations should be tested on 
- model_type: Name of model 
- eval_datasets: Whether to evaluate on the Training, Testing, or Full dataset

#### TRAIN DATASET ADJUSTMENT PARAMS
For each adjustment set any (or all) of the params to 0 for these adjustments to not be performed

The following two parameters adjust the number of particles used in the training. This is done by dividing the radii into log bins, setting a maximum particle
number based off the total number of particles within a radii and then limiting every following bin to that maximum number of particles.
 
- reduce_rad: takes any radius (>0) and will set the amount of particles within that radii as the maximum number of particles per following radius bin
- reduce_perc: takes a decimal and will scale the maximum amount 


You can also determine a radius after which orbiting particles will start to be weighted less on an exponential curve (less important the further out). Weighting is of form: weights = e^((ln(min_weight)/(max_rad-weight_rad)) * (rad - weight_rad))
- weight_rad: determines the radius at which this weighting starts (all particles with smaller radii have weights of 1)
- min_weight: determines the lowest weight at the furthest radius.
- weight_exp: Exponent value for weighting if needed in that equation


Perform hyperparameter tuning on the weighting of the dataset. Overwrites the weight_rad/min_weight parameter
- opt_wghts: Optimize weights
- opt_scale_rad: Optimize scaling down of dataset


- nu_splits: Splits in the form of [low level 1]-[high level 1], [low level N]-[high level N] for which mass ranges are used for training
- plt_nu_splits: Splits in the form of [low level 1]-[high level 1], [low level N]-[high level N] for which mass ranges are used for pltoting density profiles

- hpo: Perform hyperparameter optimization. NOTE: Mark as TRUE even if the hpo model has already been trained
- hpo_loss: Options are: all: accuracy on all particles, orb: accuracy on only orbiting particles, inf: accuracy on only infalling particles (not recommended all does basically the same), mprf_all: accuracy on infalling + orbiting mass profiles, mprf_orb: accuracy on only orbiting mass profile
- training_rad: the radius that the training dataset will be created up to. Note: the testing dataset will use all data
- rad_splits: not implemented now
- frac_train_data: What fraction of training data to use. Similar to file_lim and probably worse


- dens_prf_plt: Make density profile plots
- fulldist_plt: Make full particle distribution plots
- misclass_plt: Make misclassication plots
- io_frac_plt: Make infalling/orbiting fraction plot
- per_err_plt: Make percent error plot


- linthrsh: For the missclass and fulldist plots where both linear and log scales are used can set the threshold for linear (from -thrsh to thrsh) and then the number of linear bins and log bins
- lin_nbin: Number of linear bins (if any)
- log_nbin: Number of log bins (if any). NOTE: THIS SHOULD BE DIVISIBLE BY 2 IF THERE ARE NEG and POS LOG BINS

List the ticks that will be displayed. The location for imshow plots is automatically calculated. Radial velocity (rv) ticks are mirrored for negative values. Leave blank if using auto generated ticks
- lin_rvticks: linear radial velocity
- log_rvticks: log radial velocity
- lin_tvticks: linear tangential velocity
- log_tvticks: log tangential velocity
- lin_rticks: linear radius
- log_rticks: log radius

### The Code itself
#### calc_ptl_props.py
This is the code that takes the simulation particle data and the SPARTA HDF5 output and turns that into the dataset that will be used for training and testing of the XGBoost model.

#### train_xgboost.py
This trains the XGBoost model based off config parameters and requires calc_ptl_props.py to have been run before to generate the datasets needed. This outputs the trained XGBoost model as well as a config file and a readable dictionary that contains information about the parameters and the misclassification rates

#### test_xgboost.py
This tests the model on the dataset you decide in the config by making the plots you choose in the config and outputs them under the model and then under the tested dataset 

## Citations

