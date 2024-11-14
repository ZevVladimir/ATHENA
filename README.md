# MLOIS or Machine Learning the Orbit Infall Split

This is a project with Dr. Benedikt Diemer that will use data from the nbody simulations run by Dr. Diemer and then processed with SPARTA [1][2][3]. The goal is to construct and compare several machine learning models in order to develop one that can accurately predict if a particle is either currently orbiting a dark matter halo or is infalling into one. This will be used to help better define the radius of a dark matter halo.

## Current State as of 05//2024
### Dataset Creation
This code can be run with only the config.init file and the exec.py file (this differs for on Zaratan with implementation upcoming since jobs are submitted separately with different requirements). Within the config.ini you choose which output from SPARTA you want to analyize and then various parameters related to this analysis. This is what calc_ptl_props.py and create_train_dataset.py is based off of. calc_ptl_props.py goes through all the host halos from SPARTA and searches around their center and finds all the particles within a certain radius. For each of these particles their radius, radial velocity, and tangential velocity relative to the halo are determined. For all particles they are then assigned a classification of infalling versus orbiting based on SPARTA's pericenter determination. This process is then repeated for a secondary snap and compiled into an .hdf5 file for all host halos. This output is then used by create_train_dataset.py to organize it into a form easily used by the ML model.

### XGBoost Training
We make use of XGBoost [5] as our ML model. Currently, not all parameters are available to be tuned in config.ini but for general purposes all capabilities are there. You train the model on the dataset created before and various plots are created for the analysis of the models performance and are outputted under that model's folder in xgboost_results/. You then can adjust the curr_sparta_file parameter to test the model on another dataset (that you have created).

## Dependencies
I make use of a customized shap fork that is also publicly available. This is done to just allow for ease of adjusting different fontsizes as well as adjusting certain features in the plots that I want hidden or not. This should only impact code run in make_shap_plots.py. 

## Citations
[1] Diemer, B. (2017). The splashback radius of halos from particle dynamics. i. the Sparta algorithm. The Astrophysical Journal Supplement Series, 231(1), 5. https://doi.org/10.3847/1538-4365/aa799c 
[2] Diemer, B. (2020). The splashback radius of halos from particle dynamics. III. halo catalogs, merger trees, and host–subhalo relations. The Astrophysical Journal Supplement Series, 251(2), 17. https://doi.org/10.3847/1538-4365/abbf51 
[3] Diemer, B. (2022). A dynamics-based density profile for dark haloes – I. Algorithm and basic results. Monthly Notices of the Royal Astronomical Society, 513(1), 573–594. https://doi.org/10.1093/mnras/stac878 
[4] Thompson R., 2014, pyGadgetReader: GADGET snapshot reader for python (ascl:1411.001)
[5] Chen, T., Guestrin, C. (2016). XGBoost. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939785 
