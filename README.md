# ATHENA: pArticle classificaTion with macHine lEarning N-body simulAtions

## Data Requirements
You need the particle data from a GADGET simulation and the .hdf5 output file from SPARTA. We used the Erebos suite of simulations for our work which is freely available at http://erebos.astro.umd.edu/erebos/. The snapshot files are found under snaps/ and the halo catalogues under rockstar_cats_200m_bnd. The naming scheme of of these simulations is what is expected for the code.

## Python Dependencies
- I make use of a customized shap fork that is also publicly available at https://github.com/ZevVladimir/shap. This is done to just allow for ease of adjusting different fontsizes as well as adjusting certain features in the plots that I want hidden or not. This should only impact code run in make_shap_plots.py. 
- We ue Python 3.10.10 and CUDA 12
- For most of the python packages use the requirements.txt file to install
- pygadgetreader needs to be cloned and installed from this repository: https://github.com/jveitchmichaelis/pygadgetreader
- SPARTA needs to be cloned and installed from this repository: https://bdiemer.bitbucket.io/sparta/ 
- For dask packages that make use of CUDA you can find alternative ways to install at: https://docs.rapids.ai/install/ 
    - Requires Python 3.10, 3.11, or 3.12

## Creating the Datasets

### Snapshot Data Requirements \[SNAP_DATA\]

Currently only GADGET simulation data is usable by the code as we use Pygadgetreader to load particle data

To see an example of snapshot data formatting look at a simulatio in the snaps/ folder at http://erebos.astro.umd.edu/erebos/ 

- GADGET simulation particle data with the path to the data indicated with the *snap_path* parameter
- The particle data is expected to be stored in the format "/snapdir_*snap_dir_format*/snapshot_*snap_format*." 
- The formats should be something like {:03d} and supplied in the config file
- **If** you already have run the code **and** have the particle information pickled you can instead just use the *known_snaps* parameter.

### SPARTA Data Requirements \[SPARTA_DATA\]

A SPARTA output file for the cbol_l0063_n0256 simulation is provided in this repo

- The halo data is expected to be supplidd from the .hdf5 output file from SPARTA
- The path to this file should be indicated with the *SPARTA_output_path* parameter
- The file's name should be provided with the *curr_sparta_file* parameter
- SPARTA should be run with at least the provided parameters so that all information is present

### Initial \[DSET_CREATE\] Config Parameters

Before running the code several parameters must be specified to determine what dataset will be generated. The generated dataset can consist currently of only two snapshots of data. The dataset contains the following information split into two storage types. The information is stored in at least one pandas dataframe saved as an .h5 file. The number of dataframes is determined by the *sub_dset_mem_size* parameter. Set this to be the maximum number of bytes a file should hold.
 
Currently the way to determine which snapshots are used is done in the following way:
- The primary snapshot is determined with the snapshot that has the redshift closest to the *p_red_shift* parameter
- The secondary snapshot is determined by going *t_dyn_step* dynamical time steps (see paper for definition) back in time from the primary snapshot

The search for particles looks within the *search_radius* (in multiples of R200m) of each halo's center. We choose a measure of R200m as this allows for generalizability to all sizes of halos.

The halos are split into training and testing dataset based off *test_dset_frac* parameter. This is done on a halo basis not a particle basis and very simply so datasets, especially smaller ones, will not be perfectly balanced by particle count or by halo size.

### Running the code: gen_ML_dsets.py

After the \[DSET_CREATE\], \[SNAP_DATA\], and \[SPARTA_DATA\] parameters are set (and potentially \[MISC\] parameters as well) you are ready to create the datasets. This is done by simply running the python code: `python3 ./src/gen_ML_dsets.py`

<details>
### Saved Dataset Information

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

Information from the config file and information generated about the snapshots used and the simulations is saved and is used during the training of a model with these data. This is done as different simulations during training might have different simulation parameters and would require different config.ini files.

1. sparta_file: *curr_sparta_file*
2. snap_dir_format: *snap_dir_format*
3. snap_format: *snap_format*
4. t_dyn_step: *t_dyn_step*
5. search_rad: *search_radius*
6. total_num_snaps: Found during the creation of the dataset by finding the highest nummber snap in the provided snapshot folder
7. test_halos_ratio: *test_dset_frac*
8. chunk_size: *mp_chunk_size*
9. HDF5 Mem Size: *sub_dset_mem_size*
10. p_snap_info:
    1. ptl_snap: Snapshot number for the particle data
    2. sparta_snap: Snapshot number within SPARTA
    3. red_shift: Redshift of the snapshot
    4. scale_factor: Scalefactor at this snapshot (calculated from redshift with colossus)
    5. hubble_const: Hubbleconstant in km/s/kpc at this snapshot (calculated from redshift with colossus)
    6. box_size: The boxsize of the simulation at this snapshot in physical kpc/h 
    7. h: The value of h for this simulation
    8. rho_m: The value of the mean density of the university at this snapshot (calculated from redshift with colossus)
11. c_snap_info: The same parameters as in p_snap_info but for the secondary (comparison) snapshot
    
</details>

## Training the Model

### Initial \[TRAIN_MODEL\] Config Parameters

Training the model takes in previously created training dataset(s) and the config parameters also created during the creation of the dataset.

The features to use for training are specified with *feature_columns* and the feature for classification is specified with *target column*. Both should be entered as a list of strings that correspond exactly to the saved names used. It is recommended to use all parameters generated but if desired less parameters can be used or the code can be edited to save more parameters and then be used for training.

When using simulations with different box sizes and different particle numbers the amount of particles and halos of different sizes can vary, sometimes significantly. This can potentially lead to the model focusing on one type of halo rather than a holistic approach. To combat this the *file_lim* parameter sets the maximum number of particle files to be loaded per simulation. For this to work the best the *sub_dset_mem_size* parameter should be the same for each simulaiton. To not use this simply set it to 0.

The *model_sims* parameter controls which simulations are used and takes a list of the names of the simulations as strings. The simulation strings in the list should be exactly the same as the dataset folder name generated by gen_ML_dsets.py

The name of the folder where each model is saved is slightly complicated in order to provide separation between models trained on the different simulations and models trained on the same simulations but with different parameters. It is constructed as followed:

/*path_path_to_models*/\<combined and shortened name of simulations\>/*model_type*/

If *tree_err* is enabled plots will be created that show the accuracy of the model depending on the number of trees used as well as how fast predictions are made with the number of trees.

<details>
### Optimization of Model (WIP)

#TODO make the name change optional and instead encourage manually changing the model's name
If you use some of the additional optimization methods the path can be changed as they include information about the optimization parameters in the name to automatically differentiate the models. 

There are some elements within the code intended to provide ways to tune the dataset to optimize the accuracy of the model. We found that these ended up being unecessary for our data to train an accurate model. These might be potentially fully fleshed out and integrated in the future.

#TODO finish this section

</details>

### Running the code: train_model.py

After the \[TRAIN_MODEL\] parameters are set (and potentially \[MISC\] and \[DASK_CLIENT\] parameters as well) you are ready to create the datasets. This is done by simply running the python code: `python3 ./src/train_model.py`

<details>
### Saved Model Information

As mentioned all saved information goes to the path for the specific model. The model's json file is saved as well as general information about the model is put in a model_info.pickle.

The model_info file includes the parameters for XGBoost which can be set within the train_model.py file. It also includes misclassified percentages described in the Section Evaluating the Model if that is turned on. We include a generic function that can be added to the ~/.bash_aliases file to easily read the contents of this file:

```
rdml(){
        python3 -c "import sys; sys.path.insert(1, '/path/to/ATHENA/src/');from utils.ML_support import print_model_prop;print_model_prop('$1')"
}
```

To use this function you simply do `rdml /path/to/model_info.pickle`

</details>

## Evaluating the Model

All plots from the evaluations of the model go within a subfolder of the model's folder labeled by the simulation's name.

### Initial \[EVAL_MODEL\] Config Parameters

To determine which simulations the model will be tested on set the *test_sims* parameter. This is a list of lists such that you can test on multiple simulations separately or all together or some mixture. For example: \[\[l0063_n1024,l1000_n1024\]\] is testing the model on two simulations combined while \[\[l0063_n1024\],\[l1000_n1024\]\] is testing the model on one simulation at a time. You can also test on the "Test" dataset, the "Train" dataset, or "Full" (both the Train and Test dataset together) for both using the *eval_datasets* parameter.

Generally to evaluate the model we look at plots which can be enabled or disabled using the following parameters:
- *dens_prf_plt*: Plots the density profiles scaled by rho_m predicted by the model versus SPARTA's calculated profiles and the ratios between them. Can split the profiles by nu values using the *dens_prf_nu_split* parameter with the splits determined by *plt_nu_splits* where the splits are written like: 0.5-1.0,1.0-1.5. Or by mass accretion rate using *plt_macc_splits*
- *fulldist_plt*: Plots the distribution of the particles in the phase space of the model (vr vs r, vt vs r, vr vs vt)
- *misclass_plt*: Plots the distribution of where the model misclassified particles in the same phase space as the fulldist_plt
- *io_frac_plt*: Plots the ratio between infalling and orbiting particles

Both of the distribution plots make use of plotting linearly up to a threshold and then logarithmically afterwards in order to better visualize the distribution. The threshold for the transition is set with *linthrsh* and the number of bins in the linear region with *lin_nbin* and the number of bins in the logarithmic region with *log_nbin*. The tick labels for each feature can be set with *lin_rvticks*, *log_rvticks*, *lin_tvticks*, *log_tvticks*, *lin_rticks*, and *log_rticks*.

### test_xgboost.py

This code will test the model on the supplied simulations using the chosen plots described above. To run simply do `python3 ./src/test_model.py`


### Example Evaluation Plots

All of the following evaluation plots are for a model trained solely on the cbol_l0063_n0256 simulation and tested solely on that simulation. For further plots with a more representative sample refer to our paper [1].

#### Density profile comparison

![dens_prf](https://github.com/ZevVladimir/ATHENA/blob/main/imgs/test_dens_med_prfl_rat.png)

#### Particle Distribution

![ptl_dist](https://github.com/ZevVladimir/ATHENA/blob/main/imgs/test_ptl_distr.png)

#### Scaled Misclassified Plot

![scale_miss](https://github.com/ZevVladimir/ATHENA/blob/main/imgs/test_scaled_miss_class.png)

### make_shap_plots.py

This creates the SHAP beeswarm plots which can be selected for by filling out the dictionary shown in the file to filter which particles are displayed. `python3 ./src/make_shap_plots.py`
These plots allow us to evaluate how the model classifies certain populations of particles. This is further explored in Section 4.1 of our paper [1].

This example plot compares the shap values for a random sample of particles in the entire dataset and particles that do not have a secondary snapshot.

![comb_shap](https://github.com/ZevVladimir/ATHENA/blob/main/imgs/test_comb_shap.png)

## Phase Space Cut Method

To compare our model to alternative methods of classification we provide two methods that make cuts in the phase space of ln(v^2/v_200m^2) and r/R200m both of which are outputted in the datasets generated by gen_ML_dsets.py. The first method is based off of Salazar 2025 (in preparation) [5] and we provide a brief description in  our paper but refer details to his paper. The second method is an iteration on that method that fits a simple horizontal line to each radial bin (logarithmic bins from 0 to the max radius as implemented by SPARTA) such that all particles below the line are orbiting and all those above the line are infalling. The lines are fitted by minimizing the difference in the misclassifications of infalling and orbiting particles.

### phase_space_cut.py

The method as described in [5] although we also generate density profiles in the style of our paper [1].

### opt_phase_space.py

The method that iterates upon the method described in [5] with further details in [1]


## Debugging

### halo_cut_plot.py

### one_halo_class.py


## Citations

[1] My paper
[2] SPARTA
[3] Colossus
[4] pygadgetreader
[5] Edgar's method
