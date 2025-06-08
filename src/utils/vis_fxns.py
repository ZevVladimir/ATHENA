import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import scipy.ndimage as ndimage

from .util_fxns import timed, split_orb_inf

plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# used to find the location of a number within bins 
def get_bin_loc(bin_edges,search_num):
    # if the search number is at or beyond the bins set it to be the maximum location
    if search_num >= bin_edges[-1]:
        search_loc = bin_edges.size - 1
    else:
        upper_index = np.searchsorted(bin_edges, search_num, side='right')
        lower_index = upper_index - 1

        lower_edge = bin_edges[lower_index]
        upper_edge = bin_edges[upper_index]

        # Interpolate the fractional position of 0 between the two edges
        fraction = (search_num - lower_edge) / (upper_edge - lower_edge)
        search_loc = lower_index + fraction
    
    return search_loc

def imshow_plot(ax, img, x_label="", y_label="", text="", title="", hide_xtick_labels=False, hide_ytick_labels=False,\
    xticks = None,yticks = None,xtick_color="white",ytick_color="white",xlinthrsh = None, ylinthrsh = None, xlim=None,ylim=None,\
        axisfontsize=28, number = None, return_img=False, kwargs=None):
    if kwargs is None:
        kwargs = {}
    
    ret_img=ax.imshow(img["hist"].T, interpolation="nearest", **kwargs)
    ax.tick_params(axis="both",which="major",length=8,width=3,direction="in")
    ax.tick_params(axis="both",which="minor",length=6,width=2,direction="in")
    ax.set_aspect('equal')
    xticks_loc = []
    yticks_loc = []
    
    if xticks is not None:
        for tick in xticks:
            xticks_loc.append(get_bin_loc(img["x_edge"],tick))
        ax.set_xticks(xticks_loc,xticks)
    if yticks is not None:
        for tick in yticks:
            yticks_loc.append(get_bin_loc(img["y_edge"],tick))
        ax.set_yticks(yticks_loc,yticks)
            
    if xlim is not None:
        min_xloc = get_bin_loc(img["x_edge"],xlim[0])
        max_xloc = get_bin_loc(img["x_edge"],xlim[1])
        ax.set_xlim(min_xloc,max_xloc)
        
    if ylim is not None:
        min_yloc = get_bin_loc(img["y_edge"],ylim[0])
        max_yloc = get_bin_loc(img["y_edge"],ylim[1])
        ax.set_ylim(min_yloc,max_yloc)
        
    if ylinthrsh is not None:
        ylinthrsh_loc = get_bin_loc(img["y_edge"],ylinthrsh)
        ax.axhline(y=ylinthrsh_loc, color='grey', linestyle='--', alpha=1)
        if np.any(np.array(yticks, dtype=np.float32) < 0):
            neg_ylinthrsh_loc = get_bin_loc(img["y_edge"],-ylinthrsh)
            y_zero_loc = get_bin_loc(img["y_edge"],0)
            ax.axhline(y=neg_ylinthrsh_loc, color='grey', linestyle='--', alpha=1)
            ax.axhline(y=y_zero_loc, color='#e6e6fa', linestyle='-.', alpha=1)
    
    if xlinthrsh is not None:
        xlinthrsh_loc = get_bin_loc(img["x_edge"],xlinthrsh)
        ax.axvline(x=xlinthrsh_loc, color='grey', linestyle='--', alpha=1)
        if np.any(np.array(xticks, dtype=np.float32) < 0):
            neg_xlinthrsh_loc = get_bin_loc(img["x_edge"],-xlinthrsh)
            x_zero_loc = get_bin_loc(img["x_edge"],0)
            ax.axvline(x=neg_xlinthrsh_loc, color='grey', linestyle='--', alpha=1)
            ax.axvline(x=x_zero_loc, color='#e6e6fa', linestyle='-.', alpha=1)

    if text != "":
        ax.text(0.01,0.03, text, ha="left", va="bottom", transform=ax.transAxes, fontsize=axisfontsize-2, bbox={"facecolor":'white',"alpha":0.9,})
    if number is not None:
        ax.text(0.02,0.90,number,ha="left",va="bottom",transform=ax.transAxes,fontsize=axisfontsize-4,bbox={"facecolor":'white',"alpha":0.9,})
    if title != "":
        ax.set_title(title,fontsize=axisfontsize+2)
    if x_label != "":
        ax.set_xlabel(x_label,fontsize=axisfontsize)
    if y_label != "":
        ax.set_ylabel(y_label,fontsize=axisfontsize)

    ax.tick_params(axis='x', which='major', labelsize=22,colors=xtick_color,labelbottom=not hide_xtick_labels,labelcolor="black",direction="in")
    ax.tick_params(axis='x', which='minor', labelsize=20,colors=xtick_color,labelbottom=not hide_xtick_labels,labelcolor="black",direction="in")
    ax.tick_params(axis='y', which='major', labelsize=22,colors=ytick_color,labelleft=not hide_ytick_labels,labelcolor="black",direction="in")
    ax.tick_params(axis='y', which='minor', labelsize=22,colors=ytick_color,labelleft=not hide_ytick_labels,labelcolor="black",direction="in")
           
    if return_img:
        return ret_img

# Uses np.histogram2d to create a histogram and the edges of the histogram in one dictionary
# Can also do a linear binning then a logarithmic binning (similar to symlog) but allows for 
# special case of only positive log and not negative log
def histogram(x,y,use_bins,hist_range,min_ptl,set_ptl,split_xscale_dict=None,split_yscale_dict=None):   
    if split_yscale_dict is not None:
        linthrsh = split_yscale_dict["linthrsh"]
        lin_nbin = split_yscale_dict["lin_nbin"]
        log_nbin = split_yscale_dict["log_nbin"]
        
        y_range = hist_range[1]
        # if the y axis goes to the negatives we split the number of log bins in two for pos and neg so there are the same amount of bins as if it was just positive
        if y_range[0] < 0:
            lin_bins = np.linspace(-linthrsh,linthrsh,lin_nbin,endpoint=False)
            neg_log_bins = -np.logspace(np.log10(-y_range[0]),np.log10(linthrsh),int(log_nbin/2),endpoint=False)
            pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(y_range[1]),int(log_nbin/2))
            y_bins = np.concatenate([neg_log_bins,lin_bins,pos_log_bins])
            
        else:
            lin_bins = np.linspace(y_range[0],linthrsh,lin_nbin,endpoint=False)
            pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(y_range[1]),log_nbin)
            y_bins = np.concatenate([lin_bins,pos_log_bins])

        if split_xscale_dict == None:
            use_bins[0] = y_bins.size 
        use_bins[1] = y_bins
        
    if split_xscale_dict is not None:
        linthrsh = split_xscale_dict["linthrsh"]
        lin_nbin = split_xscale_dict["lin_nbin"]
        log_nbin = split_xscale_dict["log_nbin"]
        
        x_range = hist_range[0]
        # if the y axis goes to the negatives
        if x_range[0] < 0:
            lin_bins = np.linspace(-linthrsh,linthrsh,lin_nbin,endpoint=False)
            neg_log_bins = -np.logspace(np.log10(-x_range[0]),np.log10(linthrsh),int(log_nbin/2),endpoint=False)
            pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(x_range[1]),int(log_nbin/2))
            x_bins = np.concatenate([neg_log_bins,lin_bins,pos_log_bins])    
        else:
            lin_bins = np.linspace(x_range[0],linthrsh,lin_nbin,endpoint=False)
            pos_log_bins = np.logspace(np.log10(linthrsh),np.log10(x_range[1]),log_nbin)
            x_bins = np.concatenate([lin_bins,pos_log_bins])

        use_bins[0] = x_bins
        if split_yscale_dict == None:
            use_bins[1] = x_bins.size 
        
    hist = np.histogram2d(x, y, bins=use_bins, range=hist_range)
    
    fin_hist = {
        "hist":hist[0],
        "x_edge":hist[1],
        "y_edge":hist[2]
    }
    
    fin_hist["hist"][fin_hist["hist"] < min_ptl] = set_ptl
    
    return fin_hist

# Scales hist_1 by hist_2. Adjustments can be made: if hist_1 doesn't have any particles in a box and hist_2 has more than act_min
# there set it to inc_min. And if hist_1 has particles in a box, the scaled hist is below inc_min, and hist_2 is larger than act_min
# again set that box to inc_min 
def scale_hists(hist_1, hist_2, make_adj = True, act_min=10, inc_min=1e-4):
    scaled_hist = {
        "x_edge":hist_2["x_edge"],
        "y_edge":hist_2["y_edge"]
    }
    scaled_hist["hist"] = np.divide(hist_1["hist"],hist_2["hist"],out=np.zeros_like(hist_1["hist"]), where=hist_2["hist"]!=0)
    
    if make_adj:
        scaled_hist["hist"] = np.where((hist_1["hist"] < 1) & (hist_2["hist"] >= act_min), inc_min, scaled_hist["hist"])
        # Where there are miss classified particles but they won't show up on the image, set them to the min
        scaled_hist["hist"] = np.where((hist_1["hist"] >= 1) & (scaled_hist["hist"] < inc_min) & (hist_2["hist"] >= act_min), inc_min, scaled_hist["hist"])
    
    return scaled_hist

# scale the number of particles so that there are no lines. Plot N / N_tot / dx / dy
def normalize_hists(hist,tot_nptl,min_ptl):
    scaled_hist = {
        "x_edge":hist["x_edge"],
        "y_edge":hist["y_edge"]
    }
    
    dx = np.diff(hist["x_edge"])
    dy = np.diff(hist["y_edge"])

    scaled_hist["hist"] = hist["hist"] / tot_nptl / dx[:,None] / dy[None,:]
    # scale all bins where lower than the min to the min

    scaled_hist["hist"] = np.where((scaled_hist["hist"] < min_ptl) & (scaled_hist["hist"] > 0), min_ptl, scaled_hist["hist"])

    return scaled_hist

def adjust_frac_hist(hist_data, inf_data, orb_data, max_value, min_value):
    hist = hist_data["hist"]
    inf_data = inf_data["hist"]
    orb_data = orb_data["hist"]

    # Create masks
    both_zero_mask = (inf_data == 0) & (orb_data == 0)
    only_infalling_mask = (inf_data > 0) & (orb_data == 0)
    only_orbiting_mask = (inf_data == 0) & (orb_data > 0)

    # Apply adjustments
    hist[both_zero_mask] = np.nan  # Set to NaN where both are zero
    hist[only_infalling_mask] = max_value  # Set to max for only infalling
    hist[only_orbiting_mask] = min_value  # Set to min for only orbiting

    hist_data["hist"] = hist
    return hist_data

def gen_ptl_dist_col(x_param, y_param, act_labels, split_scale_dict, ptl_lim_dict, split_x=False, split_y=False):
    # The default number of bins is assuming there is no log split and that all bins are linear (histogram will adjust the bins if log are required)
    lin_nbin = split_scale_dict["lin_nbin"]
    def_bins = [lin_nbin,lin_nbin]
    
    x_param_range = [np.nanmin(x_param),np.nanmax(x_param)]
    y_param_range = [np.nanmin(y_param),np.nanmax(y_param)]
    
    act_min_ptl = ptl_lim_dict["act_min_ptl"]
    set_ptl = ptl_lim_dict["set_ptl"]
    scale_min_ptl = ptl_lim_dict["scale_min_ptl"]

    inf_x_param, orb_x_param = split_orb_inf(x_param,act_labels)
    inf_y_param, orb_y_param = split_orb_inf(y_param,act_labels)
    
    if split_x:
        split_xscale_dict = split_scale_dict
    else:
        split_xscale_dict = None
    if split_y:
        split_yscale_dict = split_scale_dict
    else:
        split_yscale_dict = None
        
    # Use the binning from all particles for the orbiting and infalling plots and the secondary snap to keep it consistent
    all_fhist = histogram(x_param,y_param,use_bins=def_bins,hist_range=[x_param_range,y_param_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_xscale_dict=split_xscale_dict,split_yscale_dict=split_yscale_dict)
    inf_fhist = histogram(inf_x_param,inf_y_param,use_bins=[all_fhist["x_edge"],all_fhist["y_edge"]],hist_range=[x_param_range,y_param_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_xscale_dict=split_xscale_dict,split_yscale_dict=split_yscale_dict)
    orb_fhist = histogram(orb_x_param,orb_y_param,use_bins=[all_fhist["x_edge"],all_fhist["y_edge"]],hist_range=[x_param_range,y_param_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_xscale_dict=split_xscale_dict,split_yscale_dict=split_yscale_dict)

    hist_frac = scale_hists(inf_fhist, orb_fhist, make_adj=False)

    hist_frac["hist"][np.where(hist_frac["hist"] == 0)] = 1e-4
    hist_frac["hist"] = np.log10(hist_frac["hist"])

    max_frac_ptl = ptl_lim_dict["max_frac_ptl"]
    min_frac_ptl = ptl_lim_dict["min_frac_ptl"]
            
    hist_frac = adjust_frac_hist(hist_frac, inf_fhist, orb_fhist, max_frac_ptl, min_frac_ptl)

    tot_nptl = x_param.shape[0]
    
    # normalize the number of particles so that there are no lines.
    all_fhist = normalize_hists(all_fhist,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
    inf_fhist = normalize_hists(inf_fhist,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
    orb_fhist = normalize_hists(orb_fhist,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
    
    return all_fhist, inf_fhist, orb_fhist, hist_frac

# Pass all the data in a dictionary to allow for reuse and general number of plots.
# Then pass in plot_combos a list of dictionaries of the keys to the data_dict of what you want plotted (left to right order for plotting)
# and whether there should be splits in the scales (boolean value for "split_x" and "split_y") and any labels/titles
# Example entry in the list, pass empty strings if you don't wish for labels or titles to be shown. By default x labels are only shown on the bottom row and y labels for every one: 
# {"x": "p_r", "y": "p_rv", "split_x": False, "split_y": True, "x_label":"$r/R_{200m}$", "y_label":"$v_r/v_{200m}$","title":"Current Snapshot", "x_ticks"=[0,0.5,1,2,3,4], "y_ticks"=[-12,-6,-3,-2,-1,0,1,2,3,6,12]}
# For the ticks include both lin and log and positive and negative if that is what you wish to display
def gen_ptl_dist_plt(act_labels, split_scale_dict, save_loc, data_dict = {}, plot_combo_list = [], \
    ptl_lim_dict = {"act_min_ptl":10,"set_ptl":0,"scale_min_ptl":1e-4,"max_frac_ptl":3.5,"min_frac_ptl":-3.5}, save_title=""):
    with timed("Full Ptl Dist Plot"):
        linthrsh = split_scale_dict["linthrsh"]

        n_col = len(plot_combo_list)
        
        widths = [4] * n_col + [0.5]
        heights = [0.15,4,4,4,4] # have extra row up top so there is space for the title
        
        cividis_cmap = plt.get_cmap("cividis")
        cividis_cmap.set_under(color='black')
        cividis_cmap.set_bad(color='black') 
        
        rdbu_cmap = plt.get_cmap("RdBu")
        rdbu_cmap.set_under(color='black')
        rdbu_cmap.set_bad(color='black') 
        
        plot_kwargs = {
                "norm":"log",
                "origin":"lower",
                "aspect":"auto",
                "cmap":cividis_cmap,
        }
        
        frac_plot_kwargs = {
                "origin":"lower",
                "aspect":"auto",
                "cmap":rdbu_cmap,
        }
        
        fig = plt.figure(constrained_layout=True, figsize=(6.5*n_col,22))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
        
        all_max_ptls = []
        all_plt_list = []
        inf_plt_list = []
        orb_plt_list = []
        frac_plt_list = []
        for i,plot_combo in enumerate(plot_combo_list):
            x_param = data_dict[plot_combo["x"]]
            y_param = data_dict[plot_combo["y"]]
            
            all_fhist, inf_fhist, orb_fhist, hist_frac = gen_ptl_dist_col(x_param, y_param, act_labels, split_scale_dict, ptl_lim_dict, plot_combo["split_x"], plot_combo["split_y"])
            all_max_ptls.append(np.nanmax(all_fhist["hist"]))

            if i == 0:
                text_all = "All Particles"
                text_inf = "Infalling Particles"
                text_orb = "Orbiting Particles"
                text_frac = r"$N_{infalling} / N_{orbiting}$"
            else:
                text_all = ""
                text_inf = ""
                text_orb = ""
                text_frac = ""
            all_plt_list.append(imshow_plot(fig.add_subplot(gs[1,i]),all_fhist,y_label=plot_combo["y_label"],text=text_all,title=plot_combo["title"],hide_xtick_labels=True,xticks=plot_combo["x_ticks"],yticks=plot_combo["y_ticks"],ylinthrsh=linthrsh,number="D"+str(i+1),kwargs=plot_kwargs,return_img=True))
            inf_plt_list.append(imshow_plot(fig.add_subplot(gs[2,i]),inf_fhist,y_label=plot_combo["y_label"],text=text_inf,hide_xtick_labels=True,xticks=plot_combo["x_ticks"],yticks=plot_combo["y_ticks"],ylinthrsh=linthrsh,number="D"+str(i+5),kwargs=plot_kwargs,return_img=True)  )
            orb_plt_list.append(imshow_plot(fig.add_subplot(gs[3,i]),orb_fhist,y_label=plot_combo["y_label"],text=text_orb,hide_xtick_labels=True,yticks=plot_combo["y_ticks"],ylinthrsh=linthrsh,number="D"+str(i+9),kwargs=plot_kwargs,return_img=True))
            frac_plt_list.append(imshow_plot(fig.add_subplot(gs[4,i]),hist_frac,x_label=plot_combo["x_label"],y_label=plot_combo["y_label"],text=text_frac, xticks=plot_combo["x_ticks"],yticks=plot_combo["y_ticks"],ylinthrsh=linthrsh,number="D"+str(i+13),kwargs=frac_plot_kwargs,return_img=True))
        
        # Can just do the all particle arrays since inf/orb will have equal or less
        max_ptl = np.nanmax(np.array(all_max_ptls))
        
        for i,plot_combo in enumerate(plot_combo_list):
            all_plt_list[i].set_clim(ptl_lim_dict["scale_min_ptl"],max_ptl)
            inf_plt_list[i].set_clim(ptl_lim_dict["scale_min_ptl"],max_ptl)
            orb_plt_list[i].set_clim(ptl_lim_dict["scale_min_ptl"],max_ptl)
            frac_plt_list[i].set_clim(ptl_lim_dict["min_frac_ptl"],ptl_lim_dict["max_frac_ptl"])            

        color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=ptl_lim_dict["scale_min_ptl"], vmax=max_ptl),cmap=cividis_cmap), cax=plt.subplot(gs[1:-1,-1]))
        color_bar.set_label(r"$dN N^{-1} dx^{-1} dy^{-1}$",fontsize=22)
        color_bar.ax.tick_params(which="major",direction="in",labelsize=18,length=5,width=3)
        color_bar.ax.tick_params(which="minor",direction="in",labelsize=18,length=2.5,width=1.5)
        
        color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=ptl_lim_dict["min_frac_ptl"], vmax=ptl_lim_dict["max_frac_ptl"]),cmap=rdbu_cmap), cax=plt.subplot(gs[-1,-1]))
        color_bar.set_label(r"$\log_{10}{N_{inf}/N_{orb}}$",fontsize=22)
        color_bar.ax.tick_params(which="major",direction="in",labelsize=18,length=5,width=3)
        color_bar.ax.tick_params(which="minor",direction="in",labelsize=18,length=2.5,width=1.5)
            
        fig.savefig(save_loc + "ptl_distr" + save_title + ".pdf")
        plt.close()

def gen_missclass_col(x_param, y_param, inc_inf_fltr, inc_orb_fltr, act_labels, split_scale_dict, ptl_lim_dict, split_x=False, split_y=False):
    # The default number of bins is assuming there is no log split and that all bins are linear (histogram will adjust the bins if log are required)
    lin_nbin = split_scale_dict["lin_nbin"]
    def_bins = [lin_nbin,lin_nbin]
    
    inc_inf_x_param = x_param[inc_inf_fltr]
    inc_orb_x_param = x_param[inc_orb_fltr]
    inc_inf_y_param = y_param[inc_inf_fltr]
    inc_orb_y_param = y_param[inc_orb_fltr]
    

    act_inf_x_param, act_orb_x_param = split_orb_inf(x_param, act_labels)
    act_inf_y_param, act_orb_y_param = split_orb_inf(y_param, act_labels)
    
    x_param_range = [np.nanmin(x_param),np.nanmax(x_param)]
    y_param_range = [np.nanmin(y_param),np.nanmax(y_param)]
    
    if split_x:
        split_xscale_dict = split_scale_dict
    else:
        split_xscale_dict = None
    if split_y:
        split_yscale_dict = split_scale_dict
    else:
        split_yscale_dict = None
    
    # Create histograms for all particles and then for the incorrect particles
    # Use the binning from all particles for the orbiting and infalling plots and the secondary snap to keep it consistent
    act_all_fhist = histogram(x_param,y_param,use_bins=def_bins,hist_range=[x_param_range,y_param_range],min_ptl=ptl_lim_dict["act_min_ptl"],set_ptl=ptl_lim_dict["act_set_ptl"],split_xscale_dict=split_xscale_dict,split_yscale_dict=split_yscale_dict)
    act_inf_fhist = histogram(act_inf_x_param,act_inf_y_param,use_bins=[act_all_fhist["x_edge"],act_all_fhist["y_edge"]],hist_range=[x_param_range,y_param_range],min_ptl=ptl_lim_dict["act_min_ptl"],set_ptl=ptl_lim_dict["act_set_ptl"],split_xscale_dict=split_xscale_dict,split_yscale_dict=split_yscale_dict)
    act_orb_fhist = histogram(act_orb_x_param,act_orb_y_param,use_bins=[act_all_fhist["x_edge"],act_all_fhist["y_edge"]],hist_range=[x_param_range,y_param_range],min_ptl=ptl_lim_dict["act_min_ptl"],set_ptl=ptl_lim_dict["act_set_ptl"],split_xscale_dict=split_xscale_dict,split_yscale_dict=split_yscale_dict)
    inc_inf_fhist = histogram(inc_inf_x_param,inc_inf_y_param,use_bins=[act_all_fhist["x_edge"],act_all_fhist["y_edge"]],hist_range=[x_param_range,y_param_range],min_ptl=ptl_lim_dict["act_min_ptl"],set_ptl=ptl_lim_dict["act_set_ptl"],split_xscale_dict=split_xscale_dict,split_yscale_dict=split_yscale_dict)
    inc_orb_fhist = histogram(inc_orb_x_param,inc_orb_y_param,use_bins=[act_all_fhist["x_edge"],act_all_fhist["y_edge"]],hist_range=[x_param_range,y_param_range],min_ptl=ptl_lim_dict["act_min_ptl"],set_ptl=ptl_lim_dict["act_set_ptl"],split_xscale_dict=split_xscale_dict,split_yscale_dict=split_yscale_dict)

    inc_all_fhist = {
        "hist":inc_inf_fhist["hist"] + inc_orb_fhist["hist"],
        "x_edge":act_all_fhist["x_edge"],
        "y_edge":act_all_fhist["y_edge"]
    }

    scale_inc_all_fhist = scale_hists(inc_all_fhist,act_all_fhist,act_min=ptl_lim_dict["act_min_ptl"],inc_min=ptl_lim_dict["inc_min_ptl"])
    scale_inc_inf_fhist = scale_hists(inc_inf_fhist,act_inf_fhist,act_min=ptl_lim_dict["act_min_ptl"],inc_min=ptl_lim_dict["inc_min_ptl"])    
    scale_inc_orb_fhist = scale_hists(inc_orb_fhist,act_orb_fhist,act_min=ptl_lim_dict["act_min_ptl"],inc_min=ptl_lim_dict["inc_min_ptl"])

    return scale_inc_all_fhist, scale_inc_inf_fhist, scale_inc_orb_fhist
        
# Same as ptl_dist_plt except plotting the misclassificaiton information but info passed in same way
# Pass all the data in a dictionary to allow for reuse and general number of plots.
# Then pass in plot_combos a list of dictionaries of the keys to the data_dict of what you want plotted (left to right order for plotting)
# and whether there should be splits in the scales (boolean value for "split_x" and "split_y") and any labels/titles
# Example entry in the list, pass empty strings if you don't wish for labels or titles to be shown. By default x labels are only shown on the bottom row and y labels for every one: 
# {"x": "p_r", "y": "p_rv", "split_x": False, "split_y": True, "x_label":"$r/R_{200m}$", "y_label":"$v_r/v_{200m}$","title":"Current Snapshot", "x_ticks"=[0,0.5,1,2,3,4], "y_ticks"=[-12,-6,-3,-2,-1,0,1,2,3,6,12]}
# For the ticks include both lin and log and positive and negative if that is what you wish to display
def gen_missclass_dist_plt(act_labels, pred_labels, split_scale_dict, save_loc, model_info, dset_name, data_dict = {}, plot_combo_list = [], \
    ptl_lim_dict = {"inc_min_ptl": 1e-4,"act_min_ptl": 10,"act_set_ptl": 0,}, save_title=""):
    with timed("Missclass Dist Plot"):
        # inc_inf: particles that are actually infalling but labeled as orbiting
        # inc_orb: particles that are actually orbiting but labeled as infalling
        inc_inf_fltr = np.where((pred_labels == 1) & (act_labels == 0))[0]
        inc_orb_fltr = np.where((pred_labels == 0) & (act_labels == 1))[0]
        num_inf = np.where(act_labels == 0)[0].shape[0]
        num_orb = np.where(act_labels == 1)[0].shape[0]
        tot_num_inc = inc_orb_fltr.shape[0] + inc_inf_fltr.shape[0]
        tot_num_ptl = num_orb + num_inf

        missclass_dict = {
            "Total Num of Particles": tot_num_ptl,
            "Num Incorrect Infalling Particles": str(inc_inf_fltr.shape[0])+", "+str(np.round(((inc_inf_fltr.shape[0]/num_inf)*100),2))+"% of infalling ptls",
            "Num Incorrect Orbiting Particles": str(inc_orb_fltr.shape[0])+", "+str(np.round(((inc_orb_fltr.shape[0]/num_orb)*100),2))+"% of orbiting ptls",
            "Num Incorrect All Particles": str(tot_num_inc)+", "+str(np.round(((tot_num_inc/tot_num_ptl)*100),2))+"% of all ptls",
        }
        
        if "Results" not in model_info:
            model_info["Results"] = {}
        
        if dset_name not in model_info["Results"]:
            model_info["Results"][dset_name]={}
        model_info["Results"][dset_name]["Primary Snap"] = missclass_dict
        
        linthrsh = split_scale_dict["linthrsh"]

        n_col = len(plot_combo_list)
        
        widths = [4] * n_col + [0.5]
        heights = [0.15,4,4,4] # have extra row up top so there is space for the title
        
        magma_cmap = plt.get_cmap("magma")
        magma_cmap = LinearSegmentedColormap.from_list('magma_truncated', magma_cmap(np.linspace(0.15, 1, 256)))
        magma_cmap.set_under(color='black')
        magma_cmap.set_bad(color='black')
        
        scale_miss_class_args = {
                "vmin":ptl_lim_dict["inc_min_ptl"],
                "vmax":1,
                "norm":"log",
                "origin":"lower",
                "aspect":"auto",
                "cmap":magma_cmap,
        }
        
        fig = plt.figure(constrained_layout=True,figsize=(6*n_col,16))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
            
        for i,plot_combo in enumerate(plot_combo_list):
            x_param = data_dict[plot_combo["x"]]
            y_param = data_dict[plot_combo["y"]]
            
            scale_inc_all_fhist, scale_inc_inf_fhist, scale_inc_orb_fhist = gen_missclass_col(x_param, y_param, inc_inf_fltr, inc_orb_fltr, act_labels, split_scale_dict, ptl_lim_dict, plot_combo["split_x"], plot_combo["split_y"])

            if i == 0:
                text_all = "All Particles"
                text_inf = "Infalling Particles"
                text_orb = "Orbiting Particles"
                text_frac = r"$N_{infalling} / N_{orbiting}$"
            else:
                text_all = ""
                text_inf = ""
                text_orb = ""
                text_frac = ""
            imshow_plot(fig.add_subplot(gs[1,i]),scale_inc_all_fhist,y_label=plot_combo["y_label"],hide_xtick_labels=True,text="All Misclassified",xticks=plot_combo["x_ticks"],yticks=plot_combo["y_ticks"],ylinthrsh=linthrsh,number="S"+str(i+1),kwargs=scale_miss_class_args, title=plot_combo["title"])
            imshow_plot(fig.add_subplot(gs[2,i]),scale_inc_inf_fhist,hide_xtick_labels=True,y_label=plot_combo["y_label"],text="Label: Orbit\nReal: Infall",xticks=plot_combo["x_ticks"],yticks=plot_combo["y_ticks"],ylinthrsh=linthrsh,number="S"+str(i+5),kwargs=scale_miss_class_args)
            imshow_plot(fig.add_subplot(gs[3,i]),scale_inc_orb_fhist,x_label=plot_combo["x_label"],y_label=plot_combo["y_label"],text="Label: Infall\nReal: Orbit",xticks=plot_combo["x_ticks"],yticks=plot_combo["y_ticks"],ylinthrsh=linthrsh,number="S"+str(i+9),kwargs=scale_miss_class_args)
        
        color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=ptl_lim_dict["inc_min_ptl"], vmax=1),cmap=magma_cmap), cax=plt.subplot(gs[1:,-1]))
        color_bar.set_label(r"$N_{\mathrm{bin, inc}} / N_{\mathrm{bin, tot}}$",fontsize=22)
        color_bar.ax.tick_params(which="major",direction="in",labelsize=18,length=10,width=3)
        color_bar.ax.tick_params(which="minor",direction="in",labelsize=18,length=5,width=1.5)
        
        fig.savefig(save_loc + save_title + "scaled_miss_class.pdf")
        plt.close()

def plot_log_vel(log_phys_vel,radii,labels,save_loc,split_scale_dict,add_line=[None,None],show_v200m=False,v200m=1.5):
    if v200m == -1:
        title = "no_cut"
    else:
        title = str(v200m) + "v200m"
    
    # The default number of bins is assuming there is no log split and that all bins are linear
    lin_nbin = split_scale_dict["lin_nbin"]
    def_bins = [lin_nbin,lin_nbin]
    
    orb_loc = np.where(labels == 1)[0]
    inf_loc = np.where(labels == 0)[0]
    
    r_range = [0,np.max(radii)]
    pv_range = [np.min(log_phys_vel),np.max(log_phys_vel)]
    
    num_bins = 500
    min_ptl = 10
    set_ptl = 0
    scale_min_ptl = 1e-4
    
    all = histogram(radii,log_phys_vel,use_bins=def_bins,hist_range=[r_range,pv_range],min_ptl=min_ptl,set_ptl=set_ptl)
    inf = histogram(radii[inf_loc],log_phys_vel[inf_loc],use_bins=def_bins,hist_range=[r_range,pv_range],min_ptl=min_ptl,set_ptl=set_ptl)
    orb = histogram(radii[orb_loc],log_phys_vel[orb_loc],use_bins=def_bins,hist_range=[r_range,pv_range],min_ptl=min_ptl,set_ptl=set_ptl)
    
    tot_nptl = radii.shape[0]
    
    all = normalize_hists(all,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
    inf = normalize_hists(inf,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
    orb = normalize_hists(orb,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
    
    # Can just do the all particle arrays since inf/orb will have equal or less
    max_ptl = np.max(np.array([np.max(all["hist"]),np.max(inf["hist"]),np.max(orb["hist"])]))
    
    magma_cmap = plt.get_cmap("magma")
    magma_cmap = LinearSegmentedColormap.from_list('magma_truncated', magma_cmap(np.linspace(0.15, 1, 256)))
    magma_cmap.set_under(color='black')
    magma_cmap.set_bad(color='black')
    
    lin_plot_kwargs = {
            "vmin":scale_min_ptl,
            "vmax":max_ptl,
            "norm":"linear",
            "origin":"lower",
            "aspect":"auto",
            "cmap":magma_cmap,
    }
    
    rticks = [0,0.5,1,2,3]
    pv_ticks = [-2,-1,0,1,2]
    
    widths = [4,4,4,.5]
    heights = [4]
    
    fig = plt.figure(constrained_layout=True, figsize=(25,10))
    if show_v200m:
        fig.suptitle(title,fontsize=32)
    gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    
    imshow_plot(ax1,all,x_label="$r/R_{200}$",xticks=rticks,yticks=pv_ticks,y_label="$log_{10}(v_{phys}/v_{200m})$",ylim=[-2,2],title="All Particles",xtick_color="white",axisfontsize=28,kwargs=lin_plot_kwargs)
    imshow_plot(ax2,inf,x_label="$r/R_{200}$",xticks=rticks,hide_ytick_labels=True,ylim=[-2,2],title="Infalling Particles",xtick_color="white",axisfontsize=28,kwargs=lin_plot_kwargs)
    imshow_plot(ax3,orb,x_label="$r/R_{200}$",xticks=rticks,hide_ytick_labels=True,ylim=[-2,2],title="Orbiting Particles",xtick_color="white",axisfontsize=28,kwargs=lin_plot_kwargs)
    
    if v200m > 0 and show_v200m:
        ax1.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
        ax2.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
        ax3.hlines(np.log10(v200m),xmin=r_range[0],xmax=r_range[1],colors="black")
    
    
    line_xloc = []
    line_yloc = []
    if add_line[0] is not None:
        line_xloc.append(get_bin_loc(all["x_edge"],radii[0]))
        line_xloc.append(get_bin_loc(all["x_edge"],radii[-1]))
        line_yloc.append(get_bin_loc(all["y_edge"],add_line[0] * radii[0] + add_line[1]))
        line_yloc.append(get_bin_loc(all["y_edge"],add_line[0] * radii[-1] + add_line[1]))
        line_label = fr"$\log_{{10}}(v_{{\text{{phys}}}} / v_{{200m}}) = {add_line[0]:.2f} r/R_{{200}} + {add_line[1]:.2f}$"
        ax1.plot(line_xloc, line_yloc,color="white",label = line_label)    
        ax2.plot(line_xloc, line_yloc,color="white")    
        ax3.plot(line_xloc, line_yloc,color="white")    
        
        ax1.legend(fontsize=20)
        

    lin_color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=scale_min_ptl, vmax=max_ptl),cmap=magma_cmap), cax=plt.subplot(gs[0,-1]))
    lin_color_bar.set_label(r"$dN / N dx dy$",fontsize=22)
    lin_color_bar.ax.tick_params(which="major",direction="in",labelsize=18,length=10,width=3)
    lin_color_bar.ax.tick_params(which="minor",direction="in",labelsize=18,length=5,width=1.5)
    
    fig.savefig(save_loc + "log_phys_vel_" + title + ".pdf")
    

    line_y = add_line[0] * radii + add_line[1]
    line_preds = np.zeros(radii.size) 
    line_preds[log_phys_vel <= line_y] = 1
    
    num_inc_inf = np.where((line_preds == 1) & (labels == 0))[0].shape[0]
    num_inc_orb = np.where((line_preds == 0) & (labels == 1))[0].shape[0]
    num_inf = np.where(labels == 0)[0].shape[0]
    num_orb = np.where(labels == 1)[0].shape[0]
    tot_num_inc = num_inc_orb + num_inc_inf
    tot_num_ptl = num_orb + num_inf


    print("Total Num of Particles", tot_num_ptl)
    print("Num Incorrect Infalling Particles", str(num_inc_inf)+", "+str(np.round(((num_inc_inf/num_inf)*100),2))+"% of infalling ptls")
    print("Num Incorrect Orbiting Particles", str(num_inc_orb)+", "+str(np.round(((num_inc_orb/num_orb)*100),2))+"% of orbiting ptls")
    print("Num Incorrect All Particles", str(tot_num_inc)+", "+str(np.round(((tot_num_inc/tot_num_ptl)*100),2))+"% of all ptls")
        
def plot_halo_slice_class(ptl_pos,preds,labels,halo_pos,halo_r200m,save_loc,search_rad=0,title=""):    
    ptl_pos[:,0] = ptl_pos[:,0] - halo_pos[0]
    ptl_pos[:,1] = ptl_pos[:,1] - halo_pos[1]
    
    inc_inf = np.where((preds == 1) & (labels == 0))[0]
    inc_orb = np.where((preds == 0) & (labels == 1))[0]
    inc_all = np.concatenate([inc_inf, inc_orb])
    
    corr_inf = np.where((preds == 0) & (labels == 0))[0]
    corr_orb = np.where((preds == 1) & (labels == 1))[0]
    corr_all = np.where(preds == labels)[0]
    
    if search_rad > 0:
        search_circle_0 = Circle((0,0),radius=search_rad*halo_r200m,edgecolor="yellow",facecolor='none',linestyle="--",fill=False,label="Search radius: 4R200m")
        search_circle_1 = Circle((0,0),radius=search_rad*halo_r200m,edgecolor="yellow",facecolor='none',linestyle="--",fill=False,label="Search radius: 4R200m")
        search_circle_2 = Circle((0,0),radius=search_rad*halo_r200m,edgecolor="yellow",facecolor='none',linestyle="--",fill=False,label="Search radius: 4R200m")
    
    r200m_circle_0 = Circle((0,0),radius=halo_r200m,edgecolor="black",facecolor='none',linestyle="--",linewidth=1,fill=False,label="R200m")
    r200m_circle_1 = Circle((0,0),radius=halo_r200m,edgecolor="black",facecolor='none',linestyle="--",linewidth=1,fill=False,label="R200m")
    r200m_circle_2 = Circle((0,0),radius=halo_r200m,edgecolor="black",facecolor='none',linestyle="--",linewidth=1,fill=False,label="R200m")
            
    axisfontsize = 10
    titlefontsize = 12
    legendfontsize = 8
    tickfontsize = 8
    
    widths = [4,4,4,.5]
    heights = [0.12]
    
    fig = plt.figure(constrained_layout=True,figsize=(9,3))
    gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
    all_ax = fig.add_subplot(gs[0])
    orb_ax = fig.add_subplot(gs[1])
    inf_ax = fig.add_subplot(gs[2])
    
    all_ax.scatter(ptl_pos[corr_all,0],ptl_pos[corr_all,1],color='green',label="Correctly Labeled",s=1)
    all_ax.scatter(ptl_pos[inc_all,0],ptl_pos[inc_all,1],color='red',label="Incorrectly Labeled",s=1)
    all_ax.add_patch(r200m_circle_0)
    all_ax.set_xlabel(r"$x [h^{-1}kpc]$",fontsize=axisfontsize)
    all_ax.set_ylabel(r"$y [h^{-1}kpc]$",fontsize=axisfontsize)
    all_ax.set_title("All Particles",fontsize=titlefontsize)
    all_ax.tick_params(axis='x', which='major', labelsize=tickfontsize, direction="in", colors="black",labelcolor="black",length=3,width=1.5)
    all_ax.tick_params(axis='y', which='major', labelsize=tickfontsize, direction="in", colors="black",labelcolor="black",length=3,width=1.5)
    all_ax.set_aspect('equal')
    
    orb_ax.scatter(ptl_pos[corr_orb,0],ptl_pos[corr_orb,1],color='green',label="Correctly Labeled",s=1)
    orb_ax.scatter(ptl_pos[inc_orb,0],ptl_pos[inc_orb,1],color='red',label="Incorrectly Labeled",s=1)
    orb_ax.add_patch(r200m_circle_1)
    orb_ax.set_xlabel(r"$x [h^{-1}kpc]$",fontsize=axisfontsize)
    orb_ax.set_title("Orbiting Particles",fontsize=titlefontsize)
    orb_ax.tick_params(axis='x', which='major', labelsize=tickfontsize, direction="in", colors="black",labelcolor="black",length=3,width=1.5)
    orb_ax.tick_params(axis='y', which='both',left=False,labelleft=False, direction="in", colors="black",labelcolor="black",length=3,width=1.5)
    # ax[1].legend(fontsize=legendfontsize)
    orb_ax.set_aspect('equal')

    inf_ax.scatter(ptl_pos[corr_inf,0],ptl_pos[corr_inf,1],color='green',label="Correctly Labeled",s=1)
    inf_ax.scatter(ptl_pos[inc_inf,0],ptl_pos[inc_inf,1],color='red',label="Incorrectly Labeled",s=1)
    inf_ax.add_patch(r200m_circle_2)
    inf_ax.set_xlabel(r"$x [h^{-1}kpc]$",fontsize=axisfontsize)
    inf_ax.set_title("Infalling Particles",fontsize=titlefontsize)
    inf_ax.tick_params(axis='x', which='major', labelsize=tickfontsize, direction="in", colors="black",labelcolor="black",length=3,width=1.5)
    inf_ax.tick_params(axis='y', which='both',left=False,labelleft=False, direction="in", colors="black",labelcolor="black",length=3,width=1.5)
    # ax[2].legend(fontsize=legendfontsize)
    inf_ax.set_aspect('equal')
    
    if search_rad > 0:
        all_ax.add_patch(search_circle_0)
        orb_ax.add_patch(search_circle_1)
        inf_ax.add_patch(search_circle_2) 
    
    xlims_all = [ax.get_xlim() for ax in [all_ax, inf_ax, orb_ax]]
    ylims_all = [ax.get_ylim() for ax in [all_ax, inf_ax, orb_ax]]

    # Calculate the global min and max for x and y
    combined_xlim = (min(x[0] for x in xlims_all), max(x[1] for x in xlims_all))
    combined_ylim = (min(y[0] for y in ylims_all), max(y[1] for y in ylims_all))

    # Set the same limits for all axes
    for ax in [all_ax, inf_ax, orb_ax]:
        ax.set_xlim(combined_xlim)
        ax.set_ylim(combined_ylim)
    
    all_ax.legend(fontsize=legendfontsize)
    fig.savefig(save_loc+title+"classif_halo_dist.pdf",dpi=300)        

def plot_halo_slice(ptl_pos, labels, halo_pos, halo_r200m, save_loc, search_rad=0, title=""):
    cividis_cmap = plt.get_cmap("cividis")
    cividis_cmap.set_under(color='white')
    cividis_cmap.set_bad(color='white')

    # Shift particle positions to be relative to halo center
    ptl_pos[:, 0] -= halo_pos[0]
    ptl_pos[:, 1] -= halo_pos[1]

    # Determine plot limits
    if search_rad > 0:
        lim = search_rad * halo_r200m * 1.05
        xlim, ylim = lim, lim
    else:
        xlim = np.max(np.abs(ptl_pos[:, 0]))
        ylim = np.max(np.abs(ptl_pos[:, 1]))

    nbins = 250

    # Calculate 2D histograms
    all_hist, xedges, yedges = np.histogram2d(
        ptl_pos[:, 0], ptl_pos[:, 1], bins=nbins, range=[[-xlim, xlim], [-ylim, ylim]]
    )
    orb_hist, _, _ = np.histogram2d(
        ptl_pos[labels == 1, 0], ptl_pos[labels == 1, 1], bins=nbins, range=[[-xlim, xlim], [-ylim, ylim]]
    )
    inf_hist, _, _ = np.histogram2d(
        ptl_pos[labels == 0, 0], ptl_pos[labels == 0, 1], bins=nbins, range=[[-xlim, xlim], [-ylim, ylim]]
    )

    # Normalize histograms by bin area
    dx = np.diff(xedges)[0]
    dy = np.diff(yedges)[0]
    bin_area = dx * dy
    
    all_hist = all_hist / bin_area
    orb_hist = orb_hist / bin_area
    inf_hist = inf_hist / bin_area
    # Set up circles for visual aids
    search_circle_1 = Circle((0, 0), radius=search_rad * halo_r200m, edgecolor="green", facecolor='none', linestyle="--", fill=False)
    search_circle_2 = Circle((0, 0), radius=search_rad * halo_r200m, edgecolor="green", facecolor='none', linestyle="--", fill=False)
    search_circle_3 = Circle((0, 0), radius=search_rad * halo_r200m, edgecolor="green", facecolor='none', linestyle="--", fill=False)
    r200m_circle_1 = Circle((0, 0), radius=halo_r200m, edgecolor="black", facecolor='none', linestyle="--", linewidth=1, fill=False)
    r200m_circle_2 = Circle((0, 0), radius=halo_r200m, edgecolor="black", facecolor='none', linestyle="--", linewidth=1, fill=False)
    r200m_circle_3 = Circle((0, 0), radius=halo_r200m, edgecolor="black", facecolor='none', linestyle="--", linewidth=1, fill=False)

    # Set up plot parameters
    axisfontsize = 10
    titlefontsize = 12
    legendfontsize = 8
    tickfontsize = 8

    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    norm = mpl.colors.LogNorm(vmin=np.min(all_hist[all_hist > 0]), vmax=np.max(all_hist))

    # Plot all particles
    im_all = axs[0].imshow(
        all_hist.T, origin='lower', extent=[-xlim, xlim, -ylim, ylim], cmap=cividis_cmap, norm=norm
    )
    axs[0].add_patch(r200m_circle_1)
    if search_rad > 0:
        axs[0].add_patch(search_circle_1)
    axs[0].set_title("All Particles", fontsize=titlefontsize)
    axs[0].set_xlabel(r"$x [h^{-1}kpc]$", fontsize=axisfontsize)
    axs[0].set_ylabel(r"$y [h^{-1}kpc]$", fontsize=axisfontsize)

    # Plot orbiting particles
    im_orb = axs[1].imshow(
        orb_hist.T, origin='lower', extent=[-xlim, xlim, -ylim, ylim], cmap=cividis_cmap, norm=norm
    )
    axs[1].add_patch(r200m_circle_2)
    if search_rad > 0:
        axs[1].add_patch(search_circle_2)
    axs[1].set_title("Orbiting Particles", fontsize=titlefontsize)
    axs[1].set_xlabel(r"$x [h^{-1}kpc]$", fontsize=axisfontsize)

    # Plot infalling particles
    im_inf = axs[2].imshow(
        inf_hist.T, origin='lower', extent=[-xlim, xlim, -ylim, ylim], cmap=cividis_cmap, norm=norm
    )
    axs[2].add_patch(r200m_circle_3)
    if search_rad > 0:
        axs[2].add_patch(search_circle_3)
    axs[2].set_title("Infalling Particles", fontsize=titlefontsize)
    axs[2].set_xlabel(r"$x [h^{-1}kpc]$", fontsize=axisfontsize)

    # Adjust tick parameters and set aspect ratio
    for i,ax in enumerate(axs):
        if i == 0:
            use_labelleft = True
        else:
            use_labelleft = False
        ax.tick_params(axis='both', which='major', labelsize=tickfontsize, labelleft = use_labelleft, direction="in", length=3, width=1.5)
        ax.set_aspect('equal')

    # Add colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cividis_cmap), ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(r"$N_{ptl} / dx / dy$", fontsize=10)
    cbar.ax.tick_params(which="major", direction="in", labelsize=8, length=5, width=1.5)

    # Save the figure
    plt.savefig(f"{save_loc}{title}_halo_dist.pdf", dpi=500)
    plt.close(fig)

def plt_cust_ke_line(ax, b,bins,linewidth):
    for i in range(bins.shape[0]-1):
        x1 = bins[i]
        x2 = bins[i+1]
        y1 = b[i]
        y2 = b[i]
        if i == 0:
            ax.plot([x1,x2],[y1,y2],lw=linewidth, color="cyan", label="Bin-by-bin Fit Cut")
        else:
            ax.plot([x1,x2],[y1,y2],lw=linewidth, color="cyan")
            
def plt_SPARTA_KE_dist(feat_dict, fltr_combs, bins, r, lnv2, perc, width, r_cut, plot_loc, title, plot_lin_too = False, cust_line_dict = None):
    m_pos = feat_dict["m_pos"]
    b_pos = feat_dict["b_pos"]
    m_neg = feat_dict["m_neg"]
    b_neg = feat_dict["b_neg"]
    
    x = np.linspace(0, 3, 1000)
    y12 = m_pos * x + b_pos
    y22 = m_neg * x + b_neg

    nbins = 200   
    
    x_range = (0, 3)
    y_range = (-7, 5)

    hist1, xedges, yedges = np.histogram2d(r[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], bins=nbins, range=(x_range, y_range))
    hist2, _, _ = np.histogram2d(r[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], bins=nbins, range=(x_range, y_range))
    hist3, _, _ = np.histogram2d(r[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], bins=nbins, range=(x_range, y_range))
    hist4, _, _ = np.histogram2d(r[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], bins=nbins, range=(x_range, y_range))

    # Combine the histograms to determine the maximum density for consistent color scaling
    combined_hist = np.maximum.reduce([hist1, hist2, hist3, hist4])
    vmax=combined_hist.max()
    
    lin_vmin = 0
    log_vmin = 1

    legend_title_fntsize = 16
    legend_fntsize = 24
    axis_fntsize = 32
    txt_fntsize = 26
    tick_label_fntsize = 24
    cbar_label_fntsize = 26
    cbar_tick_fntsize = 24

    with timed("SPARTA KE Dist plot"):
        magma_cmap = plt.get_cmap("magma")
        magma_cmap.set_under(color='black')
        magma_cmap.set_bad(color='black') 
        
        line_width = 6.0
        
        widths = [4,4,4,4,.5]
        nrows = 2 if plot_lin_too else 1
        heights = [4] * nrows
        fig = plt.figure(constrained_layout=True, figsize=(36, 9 * nrows))
        gs = fig.add_gridspec(nrows, len(widths), width_ratios=widths, height_ratios=heights, hspace=0.05, wspace=0)


        axes = []
        for row in range(nrows):
            for col in range(4):  # Only first 4 columns used for hist2d
                ax = fig.add_subplot(gs[row, col])
                axes.append(ax)
        row_norms = [None,"log"] if plot_lin_too else ["log"]

        for row_idx, norm in enumerate(row_norms):
            offset = row_idx * 4
            ax1, ax2, ax3, ax4 = axes[offset:offset+4]

            for ax in [ax2, ax3, ax4]:
                ax.tick_params('y', labelleft=False, colors="white", direction="in")

            hist_args = dict(bins=nbins, vmin=(log_vmin if norm == "log" else lin_vmin), vmax=vmax,
                            cmap=magma_cmap, range=(x_range, y_range), norm=(norm if norm == "log" else None), rasterized=True)

            # Orb vr > 0
            h1 = ax1.hist2d(r[fltr_combs["orb_vr_pos"]], lnv2[fltr_combs["orb_vr_pos"]], **hist_args)
            ax1.plot(x, y12, lw=line_width, color="g", label="Fast Cut")
            ax1.vlines(x=r_cut, ymin=y_range[0], ymax=y_range[1], lw=line_width, label="Calibration Limit")
            if cust_line_dict is not None:
                plt_cust_ke_line(ax=ax1,b=cust_line_dict["orb_vr_pos"]["b"], bins=bins, linewidth=line_width)
            ax1.text(0.05, 3.6, r"Orbiting Particles $v_r>0$" + "\nAccording to SPARTA",
                    fontsize=txt_fntsize, weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax1.set_ylabel(r'$\ln(v^2/v_{200m}^2)$', fontsize=axis_fntsize)
            if row_idx == (nrows - 1):
                ax1.set_xlabel(r'$r/R_{200m}$', fontsize=axis_fntsize)
            ax1.set_xlim(0, 2)

            # Inf vr > 0
            ax2.hist2d(r[fltr_combs["inf_vr_pos"]], lnv2[fltr_combs["inf_vr_pos"]], **hist_args)
            ax2.plot(x, y12, lw=line_width, color="g", label="Fast Cut")
            ax2.vlines(x=r_cut, ymin=y_range[0], ymax=y_range[1], lw=line_width, label="Calibration Limit")
            if cust_line_dict is not None:
                plt_cust_ke_line(ax=ax2,b=cust_line_dict["inf_vr_pos"]["b"], bins=bins, linewidth=line_width)
            ax2.text(0.05, 3.6, r"Infalling Particles $v_r>0$" + "\nAccording to SPARTA",
                    fontsize=txt_fntsize, weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            if row_idx == (nrows - 1):
                ax2.set_xlabel(r'$r/R_{200m}$', fontsize=axis_fntsize)
            ax2.set_xlim(0, 2)

            # Orb vr < 0
            ax3.hist2d(r[fltr_combs["orb_vr_neg"]], lnv2[fltr_combs["orb_vr_neg"]], **hist_args)
            ax3.plot(x, y22, lw=line_width, color="g", label="Fast Cut")
            ax3.vlines(x=r_cut, ymin=y_range[0], ymax=y_range[1], lw=line_width, label="Calibration Limit")
            if cust_line_dict is not None:
                plt_cust_ke_line(ax=ax3,b=cust_line_dict["orb_vr_neg"]["b"], bins=bins, linewidth=line_width)
            ax3.text(0.05, 3.6, r"Orbiting Particles $v_r<0$" + "\nAccording to SPARTA",
                    fontsize=txt_fntsize, weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            if row_idx == (nrows - 1):
                ax3.set_xlabel(r'$r/R_{200m}$', fontsize=axis_fntsize)
            ax3.set_xlim(0, 2)

            # Inf vr < 0
            ax4.hist2d(r[fltr_combs["inf_vr_neg"]], lnv2[fltr_combs["inf_vr_neg"]], **hist_args)
            ax4.plot(x, y22, lw=line_width, color="g", label="Fast Cut")
            ax4.vlines(x=r_cut, ymin=y_range[0], ymax=y_range[1], lw=line_width, label="Calibration Limit")
            if cust_line_dict is not None:
                plt_cust_ke_line(ax=ax4,b=cust_line_dict["inf_vr_neg"]["b"], bins=bins, linewidth=line_width)
            if row_idx == 0:
                ax4.legend(loc="lower left", fontsize=legend_fntsize)
            ax4.text(0.05, 3.6, r"Infalling Particles $v_r<0$" + "\nAccording to SPARTA",
                    fontsize=txt_fntsize, weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            if row_idx == (nrows - 1):
                ax4.set_xlabel(r'$r/R_{200m}$', fontsize=axis_fntsize)
            ax4.set_xlim(0, 2)
            
            cbar = fig.colorbar(h1[3], cax=plt.subplot(gs[row_idx,-1]), orientation='vertical')
            cbar.ax.tick_params(labelsize=cbar_tick_fntsize)
            cbar.set_label(r'$N$ (Counts)', fontsize=cbar_label_fntsize)

            for i in range(4):
                ax = axes[offset + i]
                if i == 3 and row_idx == 0:
                    ax.text(0.1, -3.7, "Orbiting According\nto Kinetic Energy Cut", fontsize=txt_fntsize, color="r",
                            weight="bold", bbox=dict(facecolor='w', alpha=0.75))
                    ax.text(0.7, 2.1, "Infalling According\nto Kinetic Energy Cut", fontsize=txt_fntsize, color="b",
                            weight="bold", bbox=dict(facecolor='w', alpha=0.75))
                if row_idx == (nrows - 1):
                    ax.tick_params(axis='both', which='both', labelcolor="black", colors="white",
                                direction="in", labelsize=tick_label_fntsize, length=8, width=2)
                else:
                    ax.tick_params(axis='both', which='both', labelcolor="black", colors="white",
                                direction="in", labelsize=tick_label_fntsize, labelbottom=False, length=8, width=2)
                

        if plot_lin_too:
            fig.savefig(plot_loc + "lin_log_" + title + "sparta_KE_dist_cut.pdf",dpi=400)    
        else:
            fig.savefig(plot_loc + "log_" + title + "sparta_KE_dist_cut.pdf",bbox_inches='tight',dpi=400) 
        
def plt_KE_dist_grad(feat_dict, fltr_combs, r_r200m, vr, lnv2, nbins, x_range, y_range, r_cut_calib, plot_loc, sim_title="Simulation: Bolshoi 1000Mpc"):
    with timed("KE Dist plot"):
        title_fntsize = 22
        legend_fntsize = 18
        
        m_pos = feat_dict["m_pos"]
        b_pos = feat_dict["b_pos"]
        m_neg = feat_dict["m_neg"]
        b_neg = feat_dict["b_neg"]
        
        x = np.linspace(0, 3, 1000)
        y12 = m_pos * x + b_pos
        y22 = m_neg * x + b_neg
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        axes = axes.flatten()
        fig.suptitle(
            r"Kinetic energy distribution of particles around halos at $z=0$"+"\n" + sim_title,fontsize=16)

        for ax in axes:
            ax.set_xlabel(r'$r/R_{200m}$',fontsize=16)
            ax.set_ylabel(r'$\ln(v^2/v_{200m}^2)$',fontsize=16)
            ax.set_xlim(0, 2)
            ax.set_ylim(-2, 2.5)
            ax.text(0.25, -1.4, "Orbiting", fontsize=16, color="r",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.text(1.5, 0.7, "Infalling", fontsize=16, color="b",
                    weight="bold", bbox=dict(facecolor='w', alpha=0.75))
            ax.tick_params(axis='both',which='both',direction="in",labelsize=12,length=8,width=2)

        plt.sca(axes[0])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        plt.hist2d(r_r200m[fltr_combs["mask_vr_pos"]], lnv2[fltr_combs["mask_vr_pos"]], bins=nbins,
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="k",
                label="Fast Cut")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        

        plt.sca(axes[1])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        plt.hist2d(r_r200m[fltr_combs["mask_vr_neg"]], lnv2[fltr_combs["mask_vr_neg"]], bins=nbins,
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="k",
                label="Fast Cut")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        plt.sca(axes[2])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        h3 = plt.hist2d(r_r200m[fltr_combs["mask_vr_pos"]], lnv2[fltr_combs["mask_vr_pos"]], bins=nbins, norm="log",
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y12, lw=2.0, color="k",
                label="Fast Cut")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(h3[3], label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)

        plt.sca(axes[3])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        h4 = plt.hist2d(r_r200m[fltr_combs["mask_vr_neg"]], lnv2[fltr_combs["mask_vr_neg"]], bins=nbins, norm="log",
                    cmap="terrain", range=(x_range, y_range))
        plt.plot(x, y22, lw=2.0, color="k",
                label="Fast Cut")
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(h4[3], label=r'$N$ (Counts)')
        plt.legend(loc="upper right",fontsize=legend_fntsize)
        plt.xlim(0, 2)
        
        mask_vrn = (vr < 0)
        mask_vrp = ~mask_vrn

        # Compute density and gradient.
        # For vr > 0
        hist_zp, hist_xp, hist_yp = np.histogram2d(r_r200m[mask_vrp], lnv2[mask_vrp], 
                                                    bins=nbins, 
                                                    range=((0, 3.), (-2, 2.5)),
                                                    density=True)
        # Bin centres
        hist_xp = 0.5 * (hist_xp[:-1] + hist_xp[1:])
        hist_yp = 0.5 * (hist_yp[:-1] + hist_yp[1:])
        # Bin spacing
        dx = np.mean(np.diff(hist_xp))
        dy = np.mean(np.diff(hist_yp))
        # Generate a 2D grid corresponding to the histogram
        hist_xp, hist_yp = np.meshgrid(hist_xp, hist_yp)
        # Evaluate the gradient at each radial bin
        hist_z_grad = np.zeros_like(hist_zp)
        for i in range(hist_xp.shape[0]):
            hist_z_grad[i, :] = np.gradient(hist_zp[i, :], dy)
        # Apply a gaussian filter to smooth the gradient.
        hist_zp = ndimage.gaussian_filter(hist_z_grad, 2.0)

        # Same for vr < 0
        hist_zn, hist_xn, hist_yn = np.histogram2d(r_r200m[mask_vrn], lnv2[mask_vrn],
                                                    bins=nbins,
                                                    range=((0, 3.), (-2, 2.5)),
                                                    density=True)
        hist_xn = 0.5 * (hist_xn[:-1] + hist_xn[1:])
        hist_yn = 0.5 * (hist_yn[:-1] + hist_yn[1:])
        dy = np.mean(np.diff(hist_yn))
        hist_xn, hist_yn = np.meshgrid(hist_xn, hist_yn)
        hist_z_grad = np.zeros_like(hist_zn)
        for i in range(hist_xn.shape[0]):
            hist_z_grad[i, :] = np.gradient(hist_zn[i, :], dy)
        hist_zn = ndimage.gaussian_filter(hist_z_grad, 2.0)

        #Plot the smoothed gradient
        plt.sca(axes[4])
        plt.title(r'$v_r > 0$',fontsize=title_fntsize)
        plt.contourf(hist_xp, hist_yp, hist_zp.T, levels=80, cmap='terrain')
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label="Smoothed Gradient Magnitude")
        plt.xlim(0, 2)
        
        # Plot the smoothed gradient
        plt.sca(axes[5])
        plt.title(r'$v_r < 0$',fontsize=title_fntsize)
        plt.contourf(hist_xn, hist_yn, hist_zn.T, levels=80, cmap='terrain')
        plt.vlines(x=r_cut_calib,ymin=y_range[0],ymax=y_range[1],label="Radius cut")
        plt.colorbar(label="Smoothed Gradient Magnitude")
        plt.xlim(0, 2)
        
        

        plt.tight_layout();
        plt.savefig(plot_loc + "fast_KE_dist_grad.pdf")
