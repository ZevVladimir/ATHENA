import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pairing import depair
from scipy.spatial import cKDTree
from colossus.cosmology import cosmology

from data_and_loading_functions import *
from visualization_functions import *
##################################################################################################################
# LOAD CONFIG PARAMETERS
import configparser
config = configparser.ConfigParser()
config.read("/home/zvladimi/MLOIS/config.ini")
curr_sparta_file = config["MISC"]["curr_sparta_file"]
path_to_MLOIS = config["PATHS"]["path_to_MLOIS"]
path_to_snaps = config["PATHS"]["path_to_snaps"]
path_to_SPARTA_data = config["PATHS"]["path_to_SPARTA_data"]
path_to_hdf5_file = path_to_SPARTA_data + curr_sparta_file + ".hdf5"
path_to_pickle = config["PATHS"]["path_to_pickle"]
path_to_calc_info = config["PATHS"]["path_to_calc_info"]
path_to_pygadgetreader = config["PATHS"]["path_to_pygadgetreader"]
path_to_sparta = config["PATHS"]["path_to_sparta"]
path_to_xgboost = config["PATHS"]["path_to_xgboost"]
create_directory(path_to_MLOIS)
create_directory(path_to_snaps)
create_directory(path_to_SPARTA_data)
create_directory(path_to_hdf5_file)
create_directory(path_to_pickle)
create_directory(path_to_calc_info)
create_directory(path_to_xgboost)
snap_format = config["MISC"]["snap_format"]

prim_only = config.getboolean("SEARCH","prim_only")
t_dyn_step = config.getfloat("SEARCH","t_dyn_step")

p_snap = config.getint("XGBOOST","p_snap")
c_snap = config.getint("XGBOOST","c_snap")
p_red_shift = config.getfloat("SEARCH","p_red_shift")
snapshot_list = [p_snap, c_snap]

search_rad = config.getfloat("SEARCH","search_rad")
total_num_snaps = config.getint("SEARCH","total_num_snaps")
per_n_halo_per_split = config.getfloat("SEARCH","per_n_halo_per_split")
test_halos_ratio = config.getfloat("SEARCH","test_halos_ratio")
curr_chunk_size = config.getint("SEARCH","chunk_size")

num_save_ptl_params = config.getint("SEARCH","num_save_ptl_params")
model_name = config["XGBOOST"]["model_name"]

num_processes = mp.cpu_count()
##################################################################################################################
import sys
sys.path.insert(0, path_to_pygadgetreader)
sys.path.insert(0, path_to_sparta)
from pygadgetreader import readsnap, readheader
from sparta_tools import sparta
from contextlib import contextmanager

cosmol = cosmology.setCosmology("bolshoi") 

with open("/home/zvladimi/MLOIS/xgboost_datasets_plots/sparta_cbol_l0063_n0256_10r200m_190to166_10.0r200msearch/datasets/test_dataset_all_keys.pickle","rb") as file:
    print(pickle.load(file))

with h5py.File(("/home/zvladimi/MLOIS/SPARTA_data/tcr_3674654_sparta_cbol_l0063_n0256_10r200m.hdf5")) as file:
    tjy_pos = file['tcr_ptl']['res_tjy']['x'][:]
    tjy_vel = file['tcr_ptl']['res_tjy']['v'][:]
    tjy_id = file["tcr_ptl"]['res_tjy']['tracer_id'][:]
    
    p_snap, p_red_shift = find_closest_z(p_red_shift)
    print("Snapshot number found:", p_snap, "Closest redshift found:", p_red_shift)
    sparta_output = sparta.load(filename=path_to_hdf5_file, load_halo_data=False, log_level= 0)
    all_red_shifts = sparta_output["simulation"]["snap_z"][:]
    p_sparta_snap = np.abs(all_red_shifts - p_red_shift).argmin()
    print("corresponding SPARTA snap num:", p_sparta_snap)
    print("check sparta redshift:",all_red_shifts[p_sparta_snap])
    
    # Set constants
    p_snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(p_snap) + "/snapshot_" + snap_format.format(p_snap)

    p_scale_factor = 1/(1+p_red_shift)
    p_rho_m = cosmol.rho_m(p_red_shift)
    p_hubble_constant = cosmol.Hz(p_red_shift) * 0.001 # convert to units km/s/kpc
    sim_box_size = sparta_output["simulation"]["box_size"] #units Mpc/h comoving
    p_box_size = sim_box_size * 10**3 * p_scale_factor #convert to Kpc/h physical

    p_snap_dict = {
        "snap":p_snap,
        "red_shift":p_red_shift,
        "scale_factor": p_scale_factor,
        "hubble_const": p_hubble_constant,
        "box_size": p_box_size,
    }
    halos_pos = file['halos']['position'][:,:,:] * 10**3 * p_scale_factor
    halos_vel = file['halos']['velocity'][:,:,:]
    halos_r200m = file['halos']['R200m'][:]

    p_ptls_pid, p_ptls_vel, p_ptls_pos = load_or_pickle_ptl_data(curr_sparta_file, str(p_snap), p_snapshot_path, p_scale_factor)

with h5py.File(("/home/zvladimi/MLOIS/calculated_info/sparta_cbol_l0063_n0256_10r200m_190to166_10.0r200msearch/train_all_particle_properties_" + curr_sparta_file + ".hdf5"), 'a') as all_particle_properties:
    scal_sqr_phys_vel = all_particle_properties["scal_sqr_phys_vel"][:]
    scaled_rad_vel = all_particle_properties["Radial_vel_"][:,0]
    halo_first = all_particle_properties["Halo_first"][:]
    halo_n = all_particle_properties["Halo_first"][:]
    scal_rad = all_particle_properties["Scaled_radii_"][:,0]
    hipids = all_particle_properties["HIPIDS"][:]
    labels = all_particle_properties["Orbit_Infall"][:]

# conditions
low_mask = np.logical_and.reduce(((scal_rad>=0.9), (scal_rad<1.05), (labels==1)))
high_mask = np.logical_and.reduce(((scal_rad>=1.05), (scal_rad<1.2), (labels==1)))
# find where in original array the array with the conditions applied is max
low_id = depair(hipids[np.where(scaled_rad_vel==np.max(np.abs(scaled_rad_vel[low_mask])))])
high_id = depair(hipids[np.where(scaled_rad_vel==np.max(np.abs(scaled_rad_vel[high_mask])))])

low_tjy_loc = np.where(p_ptls_pid==low_id[0])[0]
high_tjy_loc = np.where(p_ptls_pid==high_id[0])[0]

num_halo_search = 5

tree = cKDTree(data = halos_pos[:,p_sparta_snap,:], leafsize = 3, balanced_tree = False, boxsize = p_box_size)
low_dist, low_idxs = tree.query(p_ptls_pos[low_tjy_loc,:], k=num_halo_search, workers=num_processes)
high_dist, high_idxs = tree.query(p_ptls_pos[high_tjy_loc,:], k=num_halo_search, workers=num_processes)
low_idxs = low_idxs[0]
high_idxs = high_idxs[0]

num_plt_snaps = 40

all_low_use_ptl_pos = np.zeros((num_plt_snaps,3))
all_low_use_ptl_vel = np.zeros((num_plt_snaps,3))
all_high_use_ptl_pos = np.zeros((num_plt_snaps,3))
all_high_use_ptl_vel = np.zeros((num_plt_snaps,3))
all_low_use_halo_pos = np.zeros((num_plt_snaps,num_halo_search,3))
all_low_use_halo_vel = np.zeros((num_plt_snaps,num_halo_search,3))
all_low_use_halo_r200m = np.zeros((num_plt_snaps,num_halo_search))
all_high_use_halo_pos = np.zeros((num_plt_snaps,num_halo_search,3))
all_high_use_halo_vel = np.zeros((num_plt_snaps,num_halo_search,3))
all_high_use_halo_r200m = np.zeros((num_plt_snaps,num_halo_search))

for i in range(num_plt_snaps):
    curr_snap = (p_snap-num_plt_snaps) + i + 1
    curr_red_shift = all_red_shifts[curr_snap]
    curr_scale_factor = 1/(1+curr_red_shift)
    
    snapshot_path = path_to_snaps + "snapdir_" + snap_format.format(curr_snap) + "/snapshot_" + snap_format.format(curr_snap)

    ptls_pos = readsnap(snapshot_path, 'pos', 'dm', suppress=1) * 10**3 * curr_scale_factor
    ptls_vel = readsnap(snapshot_path, 'vel', 'dm', suppress=1)
    
    all_low_use_ptl_pos[i] = ptls_pos[low_tjy_loc,:]
    all_low_use_ptl_vel[i] = ptls_vel[low_tjy_loc,:]
    all_high_use_ptl_pos[i] = ptls_pos[high_tjy_loc,:]
    all_high_use_ptl_vel[i] = ptls_vel[high_tjy_loc,:]
    
    all_low_use_halo_pos[i,:,:] = halos_pos[low_idxs,curr_snap]
    all_low_use_halo_vel[i,:,:] = halos_vel[low_idxs,curr_snap]
    all_low_use_halo_r200m[i,:] = halos_r200m[low_idxs,curr_snap]
    
    all_high_use_halo_pos[i,:,:] = halos_pos[high_idxs,curr_snap]
    all_high_use_halo_vel[i,:,:] = halos_vel[high_idxs,curr_snap]
    all_high_use_halo_r200m[i,:] = halos_r200m[high_idxs,curr_snap]


def update(curr_frame, ax, halo_pos, halo_vel, ptl_pos, ptl_vel, radius):
    for i in range(num_halo_search):
        if curr_frame == 1:
            ax.scatter(np.array([]),np.array([]),color=halo_clrs[i], label=("Halo " + str(i)))
        
        ax.quiver(halo_pos[curr_frame,i,0],halo_pos[curr_frame,i,1],halo_pos[curr_frame,i,2],halo_vel[curr_frame,i,0],halo_vel[curr_frame,i,1],halo_vel[curr_frame,i,2], alpha=alphas[curr_frame], color=halo_clrs[i])
        # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        # x = halo_pos[i,j,0] + radius[i,j] * np.cos(u)*np.sin(v)
        # y = halo_pos[i,j,1] + radius[i,j] * np.sin(u)*np.sin(v)
        # z = halo_pos[i,j,2] + radius[i,j] * np.cos(v)

        # ax.plot_wireframe(x, y, z, color=halo_clrs[j])
        # ax.set_box_aspect([1,1,1])
    
    if curr_frame == 1:
        ax.scatter(np.array([]),np.array([]), color="blue", label = "Ptl")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.quiver(ptl_pos[curr_frame,0],ptl_pos[curr_frame,1],ptl_pos[curr_frame,2],ptl_vel[curr_frame,0],ptl_vel[curr_frame,1],ptl_vel[curr_frame,2], color="blue", alpha=alphas[curr_frame])

    if curr_frame == 0:    
        min_x = np.amin([np.amin(halo_pos[:,:,0]),np.amin(ptl_pos[:,0])])
        max_x = np.amax([np.amax(halo_pos[:,:,0]),np.amin(ptl_pos[:,0])])
        min_y = np.amin([np.amin(halo_pos[:,:,1]),np.amin(ptl_pos[:,1])])
        max_y = np.amax([np.amax(halo_pos[:,:,1]),np.amin(ptl_pos[:,1])])
        min_z = np.amin([np.amin(halo_pos[:,:,2]),np.amin(ptl_pos[:,2])])
        max_z = np.amax([np.amax(halo_pos[:,:,2]),np.amin(ptl_pos[:,2])])
        
        # x_adj = 0.5*(max_x-min_x)
        # y_adj = 0.5*(max_y-min_y)
        # z_adj = 0.5*(max_z-min_z)
        x_adj=0
        y_adj=0
        z_adj=0

        ax.set_xlim(np.max([min_x-x_adj,0]), np.min([(max_x+x_adj),p_box_size]))
        ax.set_ylim(np.max([min_y-y_adj,0]), np.min([(max_y+y_adj),p_box_size]))
        ax.set_zlim(np.max([min_z-z_adj,0]), np.min([(max_z+z_adj),p_box_size]))

    ax.text2D(.01,.95, s=str("Snapshot: " + str(p_snap-num_plt_snaps+curr_frame+1)), ha="left", va="top", transform=ax.transAxes, fontsize="x-large", bbox={"facecolor":'white',"alpha":.9,})

    return q,


halo_clrs = plt.cm.viridis(np.linspace(0, 1, num_halo_search))
alphas = np.logspace(np.log10(0.1),np.log10(1),num_plt_snaps)

low_fig = plt.figure()
low_ax = low_fig.add_subplot(projection='3d')
q = low_ax.quiver([], [], [], [], [], [], color='r')

high_fig = plt.figure()
high_ax = high_fig.add_subplot(projection='3d')
q = high_ax.quiver([], [], [], [], [], [], color='r')

fps = 3
    
ani = FuncAnimation(low_fig, update, frames=num_plt_snaps, fargs=(low_ax, all_low_use_halo_pos, all_low_use_halo_vel, all_low_use_ptl_pos, all_low_use_ptl_vel, all_low_use_halo_r200m), interval=200, blit=True)
ani.save("/home/zvladimi/MLOIS/Random_figures/low_ptl_track.mp4", writer='ffmpeg', fps=fps)

ani = FuncAnimation(high_fig, update, frames=num_plt_snaps, fargs=(high_ax, all_high_use_halo_pos, all_high_use_halo_vel, all_high_use_ptl_pos, all_high_use_ptl_vel, all_high_use_halo_r200m), interval=200, blit=True)
ani.save("/home/zvladimi/MLOIS/Random_figures/high_ptl_track.mp4", writer='ffmpeg', fps=fps)


# widths = [4,4,4,.25]
# heights = [4]

# fig = plt.figure(constrained_layout=True, figsize=(45,15))
# #scal_miss_class_fig.suptitle("Misclassified Particles/Num Targets " + title)
# num_bins = 250


# gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
# use_range= [[0,np.max(scal_rad)],[np.min(scal_sqr_phys_vel),np.max(scal_sqr_phys_vel)]]

# hist = np.histogram2d(scal_rad,scal_sqr_phys_vel, bins=num_bins, range=use_range)
# max_ptl = 5000
# magma_cmap = plt.get_cmap("magma_r")


# inf_loc = np.where(labels == 0)[0]
# orb_loc = np.where(labels == 1)[0]
# pos_vr_loc = np.where(scaled_rad_vel > 0)
# neg_vr_loc = np.where(scaled_rad_vel < 0)

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

# mpl.use('agg')
# mpl.rcParams.update({'font.size': 30})
# # phase_plot(fig.add_subplot(gs[0,0]),scal_rad[pos_vr_loc],scal_sqr_phys_vel[pos_vr_loc],min_ptl = 1, max_ptl=max_ptl,range=use_range,num_bins=num_bins,cmap=magma_cmap, norm="linear", x_label="$r/r_{200m}$", y_label="$ln(v^2/v_{200m}^2)$", xrange=[0,10],yrange=[-8,5],title="Radial Vel $> 0$")
# # phase_plot(fig.add_subplot(gs[0,1]),scal_rad[neg_vr_loc],scal_sqr_phys_vel[neg_vr_loc],min_ptl = 1, max_ptl=max_ptl,range=use_range,num_bins=num_bins,cmap=magma_cmap, norm="linear", x_label="$r/r_{200m}$", xrange=[0,10], yrange=[-8,5], title="Radial Vel $< 0$", hide_yticks=True)
# # phase_plot(fig.add_subplot(gs[0,2]),scal_rad,scal_sqr_phys_vel,min_ptl = 1, max_ptl=max_ptl,range=use_range,num_bins=num_bins,cmap=magma_cmap, norm="linear", x_label="$r/r_{200m}$", xrange=[0,10], yrange=[-8,5], title="All Particles", hide_yticks=True)
# # clr_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=1, vmax=max_ptl),cmap=magma_cmap), cax=plt.subplot(gs[0,-1]))
# # fig.savefig("/home/zvladimi/MLOIS/Random_figures/rv_ptl_distr_10r200m.png")

# fig = plt.figure(constrained_layout=True, figsize=(45,15))
# gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
# phase_plot(fig.add_subplot(gs[0,0]),scal_rad[orb_loc],scal_sqr_phys_vel[orb_loc],min_ptl = 0, max_ptl=max_ptl,range=use_range,num_bins=num_bins,cmap=magma_cmap, norm="linear", x_label="$r/r_{200m}$", y_label="$ln(v^2/v_{200m}^2)$", xrange=[0,3],yrange=[-8,5],title="Orbiting Particles")
# phase_plot(fig.add_subplot(gs[0,1]),scal_rad[inf_loc],scal_sqr_phys_vel[inf_loc],min_ptl = 0, max_ptl=max_ptl,range=use_range,num_bins=num_bins,cmap=magma_cmap, norm="linear", x_label="$r/r_{200m}$", xrange=[0,3], yrange=[-8,5], title="Infalling Particles", hide_yticks=True)
# phase_plot(fig.add_subplot(gs[0,2]),scal_rad,scal_sqr_phys_vel,min_ptl = 0, max_ptl=max_ptl,range=use_range,num_bins=num_bins,cmap=magma_cmap, norm="linear", x_label="$r/r_{200m}$", xrange=[0,3], yrange=[-8,5], title="All Particles", hide_yticks=True)

# props = dict(boxstyle='round', facecolor='white', alpha=0.5)

# m = -2.4
# b = 0.4

# textstr = '\n'.join((
#     r'$m=%.2f$' % (m, ),
#     r'$b=%.2f$' % (b, ),
#     ))

# fig.get_axes()[0].plot(scal_rad, (m*scal_rad+b),c='blue')
# fig.get_axes()[1].plot(scal_rad, (m*scal_rad+b),c='blue')
# fig.get_axes()[2].plot(scal_rad, (m*scal_rad+b),c='blue')
# fig.get_axes()[0].text(0.8, 0.95, textstr, transform=fig.get_axes()[0].transAxes, fontsize=36,
#         verticalalignment='top', bbox=props)
# clr_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=max_ptl),cmap=magma_cmap), cax=plt.subplot(gs[0,-1]))

# fig.savefig("/home/zvladimi/MLOIS/Random_figures/rv_ptl_distr_3r200m.png")
