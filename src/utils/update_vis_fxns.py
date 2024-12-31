import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerTuple
import xgboost as xgb

from .data_and_loading_functions import split_orb_inf, timed

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

def gen_ticks(bin_edges,spacing=6):
    ticks = []
    tick_loc = []

    # Add every spacing bin edge
    ticks.extend(bin_edges[::spacing])

    # Ensure the first and last bin edges are included
    ticks.extend([bin_edges[0], bin_edges[-1]])

    tick_loc = np.arange(bin_edges.size)[::spacing].tolist()
    tick_loc.extend([0,bin_edges.size-1])

    zero_loc = get_bin_loc(bin_edges,0)

    # only add the tick if it is noticeably far away from 0
    if zero_loc > 0.05:
        tick_loc.append(zero_loc)
        ticks.append(0)

    # Remove ticks that will get rounded down to 0
    ticks = np.round(ticks,2).tolist()
    rmv_ticks = np.where(ticks == 0)[0]
    if rmv_ticks.size > 0:
        ticks = ticks.pop(rmv_ticks)
        tick_loc = tick_loc.pop(rmv_ticks)

    # Remove duplicates and sort the list
    ticks = sorted(set(ticks))
    tick_loc = sorted(set(tick_loc))
    
    return tick_loc, ticks

# TODO add a configuration dictionary that can be passed instead
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

def plot_full_ptl_dist(p_corr_labels, p_r, p_rv, p_tv, c_r, c_rv, split_scale_dict, num_bins, save_loc):
    with timed("Full Ptl Dist Plot"):
        
        linthrsh = split_scale_dict["linthrsh"]
        log_nbin = split_scale_dict["log_nbin"]
        
        p_r_range = [np.min(p_r),np.max(p_r)]
        p_rv_range = [np.min(p_rv),np.max(p_rv)]
        p_tv_range = [np.min(p_tv),np.max(p_tv)]
        
        act_min_ptl = 10
        set_ptl = 0
        scale_min_ptl = 1e-4
        
        inf_p_r, orb_p_r = split_orb_inf(p_r,p_corr_labels)
        inf_p_rv, orb_p_rv = split_orb_inf(p_rv,p_corr_labels)
        inf_p_tv, orb_p_tv = split_orb_inf(p_tv,p_corr_labels)
        inf_c_r, orb_c_r = split_orb_inf(c_r,p_corr_labels)
        inf_c_rv, orb_c_rv = split_orb_inf(c_rv,p_corr_labels)
        
        # Use the binning from all particles for the orbiting and infalling plots and the secondary snap to keep it consistent
        all_p_r_p_rv = histogram(p_r,p_rv,use_bins=[num_bins,num_bins],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        all_p_r_p_tv = histogram(p_r,p_tv,use_bins=[num_bins,num_bins],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        all_p_rv_p_tv = histogram(p_rv,p_tv,use_bins=[num_bins,num_bins],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
        all_c_r_c_rv = histogram(c_r,c_rv,use_bins=[all_p_r_p_rv["x_edge"],all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        
        inf_p_r_p_rv = histogram(inf_p_r,inf_p_rv,use_bins=[all_p_r_p_rv["x_edge"],all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        inf_p_r_p_tv = histogram(inf_p_r,inf_p_tv,use_bins=[all_p_r_p_tv["x_edge"],all_p_r_p_tv["y_edge"]],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        inf_p_rv_p_tv = histogram(inf_p_rv,inf_p_tv,use_bins=[all_p_rv_p_tv["x_edge"],all_p_rv_p_tv["y_edge"]],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
        inf_c_r_c_rv = histogram(inf_c_r,inf_c_rv,use_bins=[all_c_r_c_rv["x_edge"],all_c_r_c_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        
        orb_p_r_p_rv = histogram(orb_p_r,orb_p_rv,use_bins=[all_p_r_p_rv["x_edge"],all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        orb_p_r_p_tv = histogram(orb_p_r,orb_p_tv,use_bins=[all_p_r_p_tv["x_edge"],all_p_r_p_tv["y_edge"]],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        orb_p_rv_p_tv = histogram(orb_p_rv,orb_p_tv,use_bins=[all_p_rv_p_tv["x_edge"],all_p_rv_p_tv["y_edge"]],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
        orb_c_r_c_rv = histogram(orb_c_r,orb_c_rv,use_bins=[all_p_r_p_rv["x_edge"],all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
        
        hist_frac_p_r_p_rv = scale_hists(inf_p_r_p_rv, orb_p_r_p_rv, make_adj=False)
        hist_frac_p_r_p_tv = scale_hists(inf_p_r_p_tv, orb_p_r_p_tv, make_adj=False)
        hist_frac_p_rv_p_tv = scale_hists(inf_p_rv_p_tv, orb_p_rv_p_tv, make_adj=False)
        hist_frac_c_r_c_rv = scale_hists(inf_c_r_c_rv, orb_c_r_c_rv, make_adj=False)
        
        hist_frac_p_r_p_rv["hist"][np.where(hist_frac_p_r_p_rv["hist"] == 0)] = 1e-4
        hist_frac_p_r_p_tv["hist"][np.where(hist_frac_p_r_p_tv["hist"] == 0)] = 1e-4
        hist_frac_p_rv_p_tv["hist"][np.where(hist_frac_p_rv_p_tv["hist"] == 0)] = 1e-4
        hist_frac_c_r_c_rv["hist"][np.where(hist_frac_c_r_c_rv["hist"] == 0)] = 1e-4
        
        hist_frac_p_r_p_rv["hist"] = np.log10(hist_frac_p_r_p_rv["hist"])
        hist_frac_p_r_p_tv["hist"] = np.log10(hist_frac_p_r_p_tv["hist"])
        hist_frac_p_rv_p_tv["hist"] = np.log10(hist_frac_p_rv_p_tv["hist"])
        hist_frac_c_r_c_rv["hist"] = np.log10(hist_frac_c_r_c_rv["hist"])
        
        # max_frac_ptl = np.max(np.array([np.max(hist_frac_p_r_p_rv["hist"]),np.max(hist_frac_p_r_p_tv["hist"]),np.max(hist_frac_p_rv_p_tv["hist"]),np.max(hist_frac_c_r_c_rv["hist"])]))
        # min_frac_ptl = np.min(np.array([np.min(hist_frac_p_r_p_rv["hist"]),np.min(hist_frac_p_r_p_tv["hist"]),np.min(hist_frac_p_rv_p_tv["hist"]),np.min(hist_frac_c_r_c_rv["hist"])]))
        max_frac_ptl = 3.5
        min_frac_ptl = -3.5
        
        
        hist_frac_p_r_p_rv = adjust_frac_hist(hist_frac_p_r_p_rv, inf_p_r_p_rv, orb_p_r_p_rv, max_frac_ptl, min_frac_ptl)
        hist_frac_p_r_p_tv = adjust_frac_hist(hist_frac_p_r_p_tv, inf_p_r_p_tv, orb_p_r_p_tv, max_frac_ptl, min_frac_ptl)
        hist_frac_p_rv_p_tv = adjust_frac_hist(hist_frac_p_rv_p_tv, inf_p_rv_p_tv, orb_p_rv_p_tv, max_frac_ptl, min_frac_ptl)
        hist_frac_c_r_c_rv = adjust_frac_hist(hist_frac_c_r_c_rv, inf_c_r_c_rv, orb_c_r_c_rv, max_frac_ptl, min_frac_ptl)
        
        tot_nptl = p_r.shape[0]
        
        # normalize the number of particles so that there are no lines.
        all_p_r_p_rv = normalize_hists(all_p_r_p_rv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        all_p_r_p_tv = normalize_hists(all_p_r_p_tv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        all_p_rv_p_tv = normalize_hists(all_p_rv_p_tv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        all_c_r_c_rv = normalize_hists(all_c_r_c_rv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)

        inf_p_r_p_rv = normalize_hists(inf_p_r_p_rv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        inf_p_r_p_tv = normalize_hists(inf_p_r_p_tv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        inf_p_rv_p_tv = normalize_hists(inf_p_rv_p_tv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        inf_c_r_c_rv = normalize_hists(inf_c_r_c_rv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)

        orb_p_r_p_rv = normalize_hists(orb_p_r_p_rv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        orb_p_r_p_tv = normalize_hists(orb_p_r_p_tv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        orb_p_rv_p_tv = normalize_hists(orb_p_rv_p_tv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        orb_c_r_c_rv = normalize_hists(orb_c_r_c_rv,tot_nptl=tot_nptl,min_ptl=scale_min_ptl)
        
        # Can just do the all particle arrays since inf/orb will have equal or less
        max_ptl = np.max(np.array([np.max(all_p_r_p_rv["hist"]),np.max(all_p_r_p_tv["hist"]),np.max(all_p_rv_p_tv["hist"]),np.max(all_c_r_c_rv["hist"])]))
        
        cividis_cmap = plt.get_cmap("cividis")
        cividis_cmap.set_under(color='black')
        cividis_cmap.set_bad(color='black') 
        
        rdbu_cmap = plt.get_cmap("RdBu")
        rdbu_cmap.set_under(color='black')
        rdbu_cmap.set_bad(color='black') 
        
        plot_kwargs = {
                "vmin":scale_min_ptl,
                "vmax":max_ptl,
                "norm":"log",
                "origin":"lower",
                "aspect":"auto",
                "cmap":cividis_cmap,
        }
        
        frac_plot_kwargs = {
                "vmin":min_frac_ptl,
                "vmax":max_frac_ptl,
                "origin":"lower",
                "aspect":"auto",
                "cmap":rdbu_cmap,
        }
        
        r_ticks = split_scale_dict["lin_rticks"] + split_scale_dict["log_rticks"]
        
        rv_ticks = split_scale_dict["lin_rvticks"] + split_scale_dict["log_rvticks"]
        rv_ticks = rv_ticks + [-x for x in rv_ticks if x != 0]
        rv_ticks.sort()

        tv_ticks = split_scale_dict["lin_tvticks"] + split_scale_dict["log_tvticks"]       
        
        widths = [4,4,4,4,.5]
        heights = [0.15,4,4,4,4] # have extra row up top so there is space for the title
        
        fig = plt.figure(constrained_layout=True, figsize=(26,22))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
        
        imshow_plot(fig.add_subplot(gs[1,0]),all_p_r_p_rv,y_label="$v_r/v_{200m}$",text="All Particles",title="Current Snapshot",hide_xtick_labels=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D1",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[1,1]),all_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xtick_labels=True,xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="D2",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[1,2]),all_p_rv_p_tv,hide_xtick_labels=True,hide_ytick_labels=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="D3",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[1,3]),all_c_r_c_rv,y_label="$v_r/v_{200m}$",title="Past Snapshot",hide_xtick_labels=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D4",kwargs=plot_kwargs)
        
        imshow_plot(fig.add_subplot(gs[2,0]),inf_p_r_p_rv,y_label="$v_r/v_{200m}$",text="Infalling Particles",hide_xtick_labels=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D5",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[2,1]),inf_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xtick_labels=True,xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="D6",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[2,2]),inf_p_rv_p_tv,hide_xtick_labels=True,hide_ytick_labels=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="D7",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[2,3]),inf_c_r_c_rv,y_label="$v_r/v_{200m}$",hide_xtick_labels=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D8",kwargs=plot_kwargs)
                    
        imshow_plot(fig.add_subplot(gs[3,0]),orb_p_r_p_rv,y_label="$v_r/v_{200m}$",text="Orbiting Particles",hide_xtick_labels=True,yticks=rv_ticks,ylinthrsh=linthrsh,number="D9",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[3,1]),orb_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xtick_labels=True,yticks=tv_ticks,ylinthrsh=linthrsh,number="D10",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[3,2]),orb_p_rv_p_tv,hide_ytick_labels=True,hide_xtick_labels=True,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="D11",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[3,3]),orb_c_r_c_rv,y_label="$v_r/v_{200m}$",hide_xtick_labels=True,yticks=rv_ticks,ylinthrsh=linthrsh,number="D12",kwargs=plot_kwargs)

        imshow_plot(fig.add_subplot(gs[4,0]),hist_frac_p_r_p_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",text=r"$N_{infalling} / N_{orbiting}$", xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D13",kwargs=frac_plot_kwargs)
        imshow_plot(fig.add_subplot(gs[4,1]),hist_frac_p_r_p_tv,x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$",xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="D14",kwargs=frac_plot_kwargs)
        imshow_plot(fig.add_subplot(gs[4,2]),hist_frac_p_rv_p_tv,x_label="$v_r/V_{200m}$",hide_ytick_labels=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="D15",kwargs=frac_plot_kwargs)
        imshow_plot(fig.add_subplot(gs[4,3]),hist_frac_c_r_c_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D16",kwargs=frac_plot_kwargs)
    
        
        color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=scale_min_ptl, vmax=max_ptl),cmap=cividis_cmap), cax=plt.subplot(gs[1:-1,-1]))
        color_bar.set_label(r"$dN N^{-1} dx^{-1} dy^{-1}$",fontsize=22)
        color_bar.ax.tick_params(which="major",direction="in",labelsize=18,length=5,width=3)
        color_bar.ax.tick_params(which="minor",direction="in",labelsize=18,length=2.5,width=1.5)
        
        color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min_frac_ptl, vmax=max_frac_ptl),cmap=rdbu_cmap), cax=plt.subplot(gs[-1,-1]))
        color_bar.set_label(r"$\log_{10}{N_{inf}/N_{orb}}$",fontsize=22)
        color_bar.ax.tick_params(which="major",direction="in",labelsize=18,length=5,width=3)
        color_bar.ax.tick_params(which="minor",direction="in",labelsize=18,length=2.5,width=1.5)
            
        fig.savefig(save_loc + "ptl_distr.png")
        plt.close()

def plot_miss_class_dist(p_corr_labels, p_ml_labels, p_r, p_rv, p_tv, c_r, c_rv, split_scale_dict, num_bins, save_loc, model_info,dataset_name):
    with timed("Miss Class Dist Plot"):

        linthrsh = split_scale_dict["linthrsh"]
        log_nbin = split_scale_dict["log_nbin"]

        p_r_range = [np.min(p_r),np.max(p_r)]
        p_rv_range = [np.min(p_rv),np.max(p_rv)]
        p_tv_range = [np.min(p_tv),np.max(p_tv)]
        
        inc_min_ptl = 1e-4
        act_min_ptl = 10
        act_set_ptl = 0

        # inc_inf: particles that are actually infalling but labeled as orbiting
        # inc_orb: particles that are actually orbiting but labeled as infalling
        inc_inf = np.where((p_ml_labels == 1) & (p_corr_labels == 0))[0]
        inc_orb = np.where((p_ml_labels == 0) & (p_corr_labels == 1))[0]
        num_inf = np.where(p_corr_labels == 0)[0].shape[0]
        num_orb = np.where(p_corr_labels == 1)[0].shape[0]
        tot_num_inc = inc_orb.shape[0] + inc_inf.shape[0]
        tot_num_ptl = num_orb + num_inf

        missclass_dict = {
            "Total Num of Particles": tot_num_ptl,
            "Num Incorrect Infalling Particles": str(inc_inf.shape[0])+", "+str(np.round(((inc_inf.shape[0]/num_inf)*100),2))+"% of infalling ptls",
            "Num Incorrect Orbiting Particles": str(inc_orb.shape[0])+", "+str(np.round(((inc_orb.shape[0]/num_orb)*100),2))+"% of orbiting ptls",
            "Num Incorrect All Particles": str(tot_num_inc)+", "+str(np.round(((tot_num_inc/tot_num_ptl)*100),2))+"% of all ptls",
        }
        
        if "Results" not in model_info:
            model_info["Results"] = {}
        
        if dataset_name not in model_info["Results"]:
            model_info["Results"][dataset_name]={}
        model_info["Results"][dataset_name]["Primary Snap"] = missclass_dict
        
        inc_inf_p_r = p_r[inc_inf]
        inc_orb_p_r = p_r[inc_orb]
        inc_inf_p_rv = p_rv[inc_inf]
        inc_orb_p_rv = p_rv[inc_orb]
        inc_inf_p_tv = p_tv[inc_inf]
        inc_orb_p_tv = p_tv[inc_orb]
        inc_inf_c_r = c_r[inc_inf]
        inc_orb_c_r = c_r[inc_orb]
        inc_inf_c_rv = c_rv[inc_inf]
        inc_orb_c_rv = c_rv[inc_orb]

        act_inf_p_r, act_orb_p_r = split_orb_inf(p_r, p_corr_labels)
        act_inf_p_rv, act_orb_p_rv = split_orb_inf(p_rv, p_corr_labels)
        act_inf_p_tv, act_orb_p_tv = split_orb_inf(p_tv, p_corr_labels)
        act_inf_c_r, act_orb_c_r = split_orb_inf(c_r, p_corr_labels)
        act_inf_c_rv, act_orb_c_rv = split_orb_inf(c_rv, p_corr_labels)
        
        # Create histograms for all particles and then for the incorrect particles
        # Use the binning from all particles for the orbiting and infalling plots and the secondary snap to keep it consistent
        act_all_p_r_p_rv = histogram(p_r,p_rv,use_bins=[num_bins,num_bins],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        act_all_p_r_p_tv = histogram(p_r,p_tv,use_bins=[num_bins,num_bins],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        act_all_p_rv_p_tv = histogram(p_rv,p_tv,use_bins=[num_bins,num_bins],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
        act_all_c_r_c_rv = histogram(c_r,c_rv,use_bins=[act_all_p_r_p_rv["x_edge"],act_all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        
        act_inf_p_r_p_rv = histogram(act_inf_p_r,act_inf_p_rv,use_bins=[act_all_p_r_p_rv["x_edge"],act_all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        act_inf_p_r_p_tv = histogram(act_inf_p_r,act_inf_p_tv,use_bins=[act_all_p_r_p_tv["x_edge"],act_all_p_r_p_tv["y_edge"]],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        act_inf_p_rv_p_tv = histogram(act_inf_p_rv,act_inf_p_tv,use_bins=[act_all_p_rv_p_tv["x_edge"],act_all_p_rv_p_tv["y_edge"]],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
        act_inf_c_r_c_rv = histogram(act_inf_c_r,act_inf_c_rv,use_bins=[act_all_c_r_c_rv["x_edge"],act_all_c_r_c_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        
        act_orb_p_r_p_rv = histogram(act_orb_p_r,act_orb_p_rv,use_bins=[act_all_p_r_p_rv["x_edge"],act_all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        act_orb_p_r_p_tv = histogram(act_orb_p_r,act_orb_p_tv,use_bins=[act_all_p_r_p_tv["x_edge"],act_all_p_r_p_tv["y_edge"]],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        act_orb_p_rv_p_tv = histogram(act_orb_p_rv,act_orb_p_tv,use_bins=[act_all_p_rv_p_tv["x_edge"],act_all_p_rv_p_tv["y_edge"]],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
        act_orb_c_r_c_rv = histogram(act_orb_c_r,act_orb_c_rv,use_bins=[act_all_p_r_p_rv["x_edge"],act_all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
            
        inc_inf_p_r_p_rv = histogram(inc_inf_p_r,inc_inf_p_rv,use_bins=[act_all_p_r_p_rv["x_edge"],act_all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        inc_inf_p_r_p_tv = histogram(inc_inf_p_r,inc_inf_p_tv,use_bins=[act_all_p_r_p_tv["x_edge"],act_all_p_r_p_tv["y_edge"]],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        inc_inf_p_rv_p_tv = histogram(inc_inf_p_rv,inc_inf_p_tv,use_bins=[act_all_p_rv_p_tv["x_edge"],act_all_p_rv_p_tv["y_edge"]],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
        inc_inf_c_r_c_rv = histogram(inc_inf_c_r,inc_inf_c_rv,use_bins=[act_all_c_r_c_rv["x_edge"],act_all_c_r_c_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        
        inc_orb_p_r_p_rv = histogram(inc_orb_p_r,inc_orb_p_rv,use_bins=[act_all_p_r_p_rv["x_edge"],act_all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        inc_orb_p_r_p_tv = histogram(inc_orb_p_r,inc_orb_p_tv,use_bins=[act_all_p_r_p_tv["x_edge"],act_all_p_r_p_tv["y_edge"]],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)
        inc_orb_p_rv_p_tv = histogram(inc_orb_p_rv,inc_orb_p_tv,use_bins=[act_all_p_rv_p_tv["x_edge"],act_all_p_rv_p_tv["y_edge"]],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
        inc_orb_c_r_c_rv = histogram(inc_orb_c_r,inc_orb_c_rv,use_bins=[act_all_p_r_p_rv["x_edge"],act_all_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=act_set_ptl,split_yscale_dict=split_scale_dict)

        
        inc_all_p_r_p_rv = {
            "hist":inc_inf_p_r_p_rv["hist"] + inc_orb_p_r_p_rv["hist"],
            "x_edge":act_all_p_r_p_rv["x_edge"],
            "y_edge":act_all_p_r_p_rv["y_edge"]
        }
        inc_all_p_r_p_tv = {
            "hist":inc_inf_p_r_p_tv["hist"] + inc_orb_p_r_p_tv["hist"],
            "x_edge":act_all_p_r_p_tv["x_edge"],
            "y_edge":act_all_p_r_p_tv["y_edge"]
        }
        inc_all_p_rv_p_tv = {
            "hist":inc_inf_p_rv_p_tv["hist"] + inc_orb_p_rv_p_tv["hist"],
            "x_edge":act_all_p_rv_p_tv["x_edge"],
            "y_edge":act_all_p_rv_p_tv["y_edge"]
        }
        inc_all_c_r_c_rv = {
            "hist":inc_inf_c_r_c_rv["hist"] + inc_orb_c_r_c_rv["hist"],
            "x_edge":act_all_c_r_c_rv["x_edge"],
            "y_edge":act_all_c_r_c_rv["y_edge"]
        }

        scale_inc_all_p_r_p_rv = scale_hists(inc_all_p_r_p_rv,act_all_p_r_p_rv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_all_p_r_p_tv = scale_hists(inc_all_p_r_p_tv,act_all_p_r_p_tv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_all_p_rv_p_tv = scale_hists(inc_all_p_rv_p_tv,act_all_p_rv_p_tv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_all_c_r_c_rv = scale_hists(inc_all_c_r_c_rv,act_all_c_r_c_rv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        
        scale_inc_inf_p_r_p_rv = scale_hists(inc_inf_p_r_p_rv,act_inf_p_r_p_rv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_inf_p_r_p_tv = scale_hists(inc_inf_p_r_p_tv,act_inf_p_r_p_tv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_inf_p_rv_p_tv = scale_hists(inc_inf_p_rv_p_tv,act_inf_p_rv_p_tv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_inf_c_r_c_rv = scale_hists(inc_inf_c_r_c_rv,act_inf_c_r_c_rv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        
        scale_inc_orb_p_r_p_rv = scale_hists(inc_orb_p_r_p_rv,act_orb_p_r_p_rv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_orb_p_r_p_tv = scale_hists(inc_orb_p_r_p_tv,act_orb_p_r_p_tv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_orb_p_rv_p_tv = scale_hists(inc_orb_p_rv_p_tv,act_orb_p_rv_p_tv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        scale_inc_orb_c_r_c_rv = scale_hists(inc_orb_c_r_c_rv,act_orb_c_r_c_rv,act_min=act_min_ptl,inc_min=inc_min_ptl)
        
        magma_cmap = plt.get_cmap("magma")
        magma_cmap = LinearSegmentedColormap.from_list('magma_truncated', magma_cmap(np.linspace(0.15, 1, 256)))
        magma_cmap.set_under(color='black')
        magma_cmap.set_bad(color='black')
        
        
        scale_miss_class_args = {
                "vmin":inc_min_ptl,
                "vmax":1,
                "norm":"log",
                "origin":"lower",
                "aspect":"auto",
                "cmap":magma_cmap,
        }
        
        r_ticks = split_scale_dict["lin_rticks"] + split_scale_dict["log_rticks"]
        
        rv_ticks = split_scale_dict["lin_rvticks"] + split_scale_dict["log_rvticks"]
        rv_ticks = rv_ticks + [-x for x in rv_ticks if x != 0]
        rv_ticks.sort()

        tv_ticks = split_scale_dict["lin_tvticks"] + split_scale_dict["log_tvticks"]     
        
        widths = [4,4,4,4,.5]
        heights = [0.12,4,4,4]
        
        fig = plt.figure(constrained_layout=True,figsize=(24,16))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

        imshow_plot(fig.add_subplot(gs[1,0]),scale_inc_all_p_r_p_rv,y_label="$v_r/v_{200m}$",hide_xtick_labels=True,text="All Misclassified",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S1",kwargs=scale_miss_class_args, title="Current Snapshot")
        imshow_plot(fig.add_subplot(gs[1,1]),scale_inc_all_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xtick_labels=True,xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="S2",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[1,2]),scale_inc_all_p_rv_p_tv,hide_xtick_labels=True,hide_ytick_labels=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="S3",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[1,3]),scale_inc_all_c_r_c_rv,y_label="$v_r/v_{200m}$",hide_xtick_labels=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S4",kwargs=scale_miss_class_args, title="Past Snapshot")

        imshow_plot(fig.add_subplot(gs[2,0]),scale_inc_inf_p_r_p_rv,hide_xtick_labels=True,y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S5",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[2,1]),scale_inc_inf_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xtick_labels=True,xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="S6",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[2,2]),scale_inc_inf_p_rv_p_tv,hide_xtick_labels=True,hide_ytick_labels=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="S7",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[2,3]),scale_inc_inf_c_r_c_rv,y_label="$v_r/v_{200m}$",hide_xtick_labels=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S8",kwargs=scale_miss_class_args)
        
        imshow_plot(fig.add_subplot(gs[3,0]),scale_inc_orb_p_r_p_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S9",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[3,1]),scale_inc_orb_p_r_p_tv,x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$",xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="S10",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[3,2]),scale_inc_orb_p_rv_p_tv,x_label="$v_r/v_{200m}$",hide_ytick_labels=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="S11",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[3,3]),scale_inc_orb_c_r_c_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S12",kwargs=scale_miss_class_args)
        
        color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=inc_min_ptl, vmax=1),cmap=magma_cmap), cax=plt.subplot(gs[1:,-1]))
        color_bar.set_label(r"$N_{\mathrm{bin, inc}} / N_{\mathrm{bin, tot}}$",fontsize=22)
        color_bar.ax.tick_params(which="major",direction="in",labelsize=18,length=10,width=3)
        color_bar.ax.tick_params(which="minor",direction="in",labelsize=18,length=5,width=1.5)
        
        fig.savefig(save_loc + "scaled_miss_class.png")
        plt.close()

def plot_perr_err():
    return

def plot_log_vel(log_phys_vel,radii,labels,save_loc,add_line=[None,None],show_v200m=False,v200m=1.5):
    if v200m == -1:
        title = "no_cut"
    else:
        title = str(v200m) + "v200m"
    
    orb_loc = np.where(labels == 1)[0]
    inf_loc = np.where(labels == 0)[0]
    
    r_range = [0,np.max(radii)]
    pv_range = [np.min(log_phys_vel),np.max(log_phys_vel)]
    
    num_bins = 500
    min_ptl = 10
    set_ptl = 0
    scale_min_ptl = 1e-4
    
    all = histogram(radii,log_phys_vel,use_bins=[num_bins,num_bins],hist_range=[r_range,pv_range],min_ptl=min_ptl,set_ptl=set_ptl)
    inf = histogram(radii[inf_loc],log_phys_vel[inf_loc],use_bins=[num_bins,num_bins],hist_range=[r_range,pv_range],min_ptl=min_ptl,set_ptl=set_ptl)
    orb = histogram(radii[orb_loc],log_phys_vel[orb_loc],use_bins=[num_bins,num_bins],hist_range=[r_range,pv_range],min_ptl=min_ptl,set_ptl=set_ptl)
    
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
    
    fig.savefig(save_loc + "log_phys_vel_" + title + ".png")
    

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
    fig.savefig(save_loc+title+"classif_halo_dist.png",dpi=300)        

def plot_halo_3d_class(ptl_pos, preds, labels, halo_pos, halo_r200m, save_loc, search_rad=0, title=""):
    # Center particles around the halo position
    ptl_pos[:, 0] -= halo_pos[0]
    ptl_pos[:, 1] -= halo_pos[1]
    ptl_pos[:, 2] -= halo_pos[2]

    # Classify particles
    inc_inf = np.where((preds == 1) & (labels == 0))[0]
    inc_orb = np.where((preds == 0) & (labels == 1))[0]
    corr_inf = np.where((preds == 0) & (labels == 0))[0]
    corr_orb = np.where((preds == 1) & (labels == 1))[0]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for particle classifications
    ax.scatter(ptl_pos[corr_inf, 0], ptl_pos[corr_inf, 1], ptl_pos[corr_inf, 2], 
               color='green', label="Correctly Labeled Infalling", s=1)
    ax.scatter(ptl_pos[corr_orb, 0], ptl_pos[corr_orb, 1], ptl_pos[corr_orb, 2], 
               color='blue', label="Correctly Labeled Orbiting", s=1)
    ax.scatter(ptl_pos[inc_inf, 0], ptl_pos[inc_inf, 1], ptl_pos[inc_inf, 2], 
               color='red', label="Incorrectly Labeled Infalling", s=1)
    ax.scatter(ptl_pos[inc_orb, 0], ptl_pos[inc_orb, 1], ptl_pos[inc_orb, 2], 
               color='orange', label="Incorrectly Labeled Orbiting", s=1)

    # Add R200m sphere
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = halo_r200m * np.outer(np.cos(u), np.sin(v))
    # y = halo_r200m * np.outer(np.sin(u), np.sin(v))
    # z = halo_r200m * np.outer(np.ones_like(u), np.cos(v))
    # ax.plot_wireframe(x, y, z, color='black', linewidth=0.5, linestyle='--', label="R200m")

    # Add search radius sphere (if applicable)
    if search_rad > 0:
        x_search = search_rad * halo_r200m * np.outer(np.cos(u), np.sin(v))
        y_search = search_rad * halo_r200m * np.outer(np.sin(u), np.sin(v))
        z_search = search_rad * halo_r200m * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x_search, y_search, z_search, color='yellow', linewidth=0.5, linestyle='--', label=f"Search Radius: {search_rad}R200m")

    # Axis labels and title
    ax.set_xlabel(r"$x [h^{-1}kpc]$")
    ax.set_ylabel(r"$y [h^{-1}kpc]$")
    ax.set_zlabel(r"$z [h^{-1}kpc]$")
    ax.set_title(title)
    ax.legend()

    # Save the figure
    save_path = save_loc + title + "_3d_classif_halo_dist.png"
    fig.savefig(save_path,dpi=300)



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
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=tickfontsize, direction="in", length=3, width=1.5)
        ax.set_aspect('equal')

    # Add colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cividis_cmap), ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(r"$N_{ptl} / dx / dy$", fontsize=10)
    cbar.ax.tick_params(which="major", direction="in", labelsize=8, length=5, width=1.5)

    # Save the figure
    plt.savefig(f"{save_loc}{title}_halo_dist.png", dpi=500)
    plt.close(fig)

# Profiles should be a list [calc_prf,act_prf]
# You can either use the median plots with use_med=True or the average with use_med=False
def compare_prfs(all_prfs, orb_prfs, inf_prfs, bins, lin_rticks, save_location, title, use_med=True):    
    if use_med:
        prf_func = np.nanmedian
    else:
        prf_func = np.nanmean
    # Parameters to tune sizes of plots and fonts
    widths = [1]
    heights = [1,0.5]
    titlefntsize=18
    axisfntsize=12
    tickfntsize=10
    legendfntsize=10
    fill_alpha = 0.2
        
    fig = plt.figure(constrained_layout=True,figsize=(10,6))
    gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
    ax_0 = fig.add_subplot(gs[0])
    ax_1 = fig.add_subplot(gs[1],sharex=ax_0)
    
    invis_calc, = ax_0.plot([0], [0], color='black', linestyle='-')
    invis_act, = ax_0.plot([0], [0], color='black', linestyle='--')
    
    # Take the ratio of the calculated profiles and the actual profiles and center around 0
    ratio_all_prf = (all_prfs[0] / all_prfs[1]) - 1
    ratio_orb_prf = (orb_prfs[0] / orb_prfs[1]) - 1
    ratio_inf_prf = (inf_prfs[0] / inf_prfs[1]) - 1

    # Plot the calculated profiles
    all_lb, = ax_0.plot(bins, prf_func(all_prfs[0],axis=0), 'r-', label = "All")
    orb_lb, = ax_0.plot(bins, prf_func(orb_prfs[0],axis=0), 'b-', label = "Orbiting")
    inf_lb, = ax_0.plot(bins, prf_func(inf_prfs[0],axis=0), 'g-', label = "Infalling")
    
    # Plot the SPARTA (actual) profiles 
    ax_0.plot(bins, prf_func(all_prfs[1],axis=0), 'r--')
    ax_0.plot(bins, prf_func(orb_prfs[1],axis=0), 'b--')
    ax_0.plot(bins, prf_func(inf_prfs[1],axis=0), 'g--')

    
    fig.legend([(invis_calc, invis_act),all_lb,orb_lb,inf_lb], ['Predicted, Actual','All','Orbiting','Infalling'], numpoints=1,handlelength=3,handler_map={tuple: HandlerTuple(ndivide=None)},frameon=False,fontsize=legendfntsize)

    ax_1.plot(bins, prf_func(ratio_all_prf,axis=0), 'r')
    ax_1.plot(bins, prf_func(ratio_orb_prf,axis=0), 'b')
    ax_1.plot(bins, prf_func(ratio_inf_prf,axis=0), 'g')
    
    ax_1.fill_between(bins, np.nanpercentile(ratio_all_prf, q=15.9, axis=0),np.nanpercentile(ratio_all_prf, q=84.1, axis=0), color='r', alpha=fill_alpha)
    ax_1.fill_between(bins, np.nanpercentile(ratio_orb_prf, q=15.9, axis=0),np.nanpercentile(ratio_orb_prf, q=84.1, axis=0), color='b', alpha=fill_alpha)
    ax_1.fill_between(bins, np.nanpercentile(ratio_inf_prf, q=15.9, axis=0),np.nanpercentile(ratio_inf_prf, q=84.1, axis=0), color='g', alpha=fill_alpha) 
        
    ax_0.set_ylabel(r"$\rho / \rho_m$", fontsize=axisfntsize)
    ax_0.set_xscale("log")
    ax_0.set_yscale("log")
    ax_0.set_xlim(0.05,np.max(lin_rticks))
    ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
    ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
    
    fig.legend([(invis_calc, invis_act),all_lb,orb_lb,inf_lb], ['Predicted, Actual','All','Orbiting','Infalling'], numpoints=1,handlelength=3,handler_map={tuple: HandlerTuple(ndivide=None)},frameon=False,fontsize=legendfntsize)

    ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
    ax_1.set_ylabel(r"$\frac{\rho_{pred}}{\rho_{act}} - 1$", fontsize=axisfntsize)
    
    ax_1.set_xlim(0.05,np.max(lin_rticks))
    ax_1.set_ylim(bottom=-0.3,top=0.3)
    ax_1.set_xscale("log")
    tick_locs = lin_rticks.copy()
    if 0 in lin_rticks:
        tick_locs.remove(0)
    #TODO remove this and have it so ticks are specific to which plot is being made
    if 0.1 not in tick_locs:
        tick_locs.append(0.1)
        tick_locs = sorted(tick_locs)
    strng_ticks = list(map(str, tick_locs))

    ax_1.set_xticks(tick_locs,strng_ticks)  
    ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
    if use_med:
        fig.savefig(save_location + title + "med_prfl_rat.png",bbox_inches='tight')
    else:
        fig.savefig(save_location + title + "avg_prfl_rat.png",bbox_inches='tight')
    
# Profiles should be a list of lists where each list consists of [calc_prf,act_prf] for each nu split
# You can either use the median plots with use_med=True or the average with use_med=False
def compare_prfs_nu(plt_nu_splits, n_lines, all_prfs, orb_prfs, inf_prfs, bins, lin_rticks, save_location, title, use_med=True):    
    if use_med:
        prf_func = np.nanmedian
    else:
        prf_func = np.nanmean
    
    # Parameters to tune sizes of plots and fonts
    widths = [1,1,1]
    heights = [1,0.5]
    titlefntsize=18
    axisfntsize=12
    textfntsize = 10
    tickfntsize=10
    legendfntsize=8
    fill_alpha = 0.2
        
    fig = plt.figure(constrained_layout=True,figsize=(10,5))
    gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
    all_ax_0 = fig.add_subplot(gs[0,0])
    all_ax_1 = fig.add_subplot(gs[1,0],sharex=all_ax_0)
    orb_ax_0 = fig.add_subplot(gs[0,1])
    orb_ax_1 = fig.add_subplot(gs[1,1],sharex=orb_ax_0)
    inf_ax_0 = fig.add_subplot(gs[0,2])
    inf_ax_1 = fig.add_subplot(gs[1,2],sharex=inf_ax_0)
    
    all_cmap = plt.cm.Reds
    orb_cmap = plt.cm.Blues
    inf_cmap = plt.cm.Greens
    
    all_colors = [all_cmap(i) for i in np.linspace(0.3, 1, n_lines)]
    orb_colors = [orb_cmap(i) for i in np.linspace(0.3, 1, n_lines)]
    inf_colors = [inf_cmap(i) for i in np.linspace(0.3, 1, n_lines)]
    
    invis_calc_all, = all_ax_0.plot([0], [0], color=all_cmap(0.75), linestyle='-')
    invis_act_all, = all_ax_0.plot([0], [0], color=all_cmap(0.75), linestyle='--')
    invis_calc_orb, = all_ax_0.plot([0], [0], color=orb_cmap(0.75), linestyle='-')
    invis_act_orb, = all_ax_0.plot([0], [0], color=orb_cmap(0.75), linestyle='--')
    invis_calc_inf, = all_ax_0.plot([0], [0], color=inf_cmap(0.75), linestyle='-')
    invis_act_inf, = all_ax_0.plot([0], [0], color=inf_cmap(0.75), linestyle='--')
    
    all_plt_lines = [invis_calc_all, invis_act_all]
    all_plt_lbls = ["Predicted","Actual"]
    
    orb_plt_lines = [invis_calc_orb, invis_act_orb]
    orb_plt_lbls = ["Predicted","Actual"]
    
    inf_plt_lines = [invis_calc_inf, invis_act_inf]
    inf_plt_lbls = ["Predicted","Actual"]
    
    for i,nu_split in enumerate(plt_nu_splits):
        # Take the ratio of the calculated profiles and the actual profiles and center around 0
        ratio_all_prf = (all_prfs[i][0] / all_prfs[i][1]) - 1
        ratio_orb_prf = (orb_prfs[i][0] / orb_prfs[i][1]) - 1
        ratio_inf_prf = (inf_prfs[i][0] / inf_prfs[i][1]) - 1
        
        # Plot the calculated profiles
        all_lb, = all_ax_0.plot(bins, prf_func(all_prfs[i][0],axis=0), linestyle='-', color = all_colors[i], label = str(nu_split[0]) + r"$< \nu <$" + str(nu_split[1]))
        orb_lb, = orb_ax_0.plot(bins, prf_func(orb_prfs[i][0],axis=0), linestyle='-', color = orb_colors[i], label = str(nu_split[0]) + r"$< \nu <$" + str(nu_split[1]))
        inf_lb, = inf_ax_0.plot(bins, prf_func(inf_prfs[i][0],axis=0), linestyle='-', color = inf_colors[i], label = str(nu_split[0]) + r"$< \nu <$" + str(nu_split[1]))
        
        all_plt_lines.append(all_lb)
        all_plt_lbls.append(str(nu_split[0]) + r"$< \nu <$" + str(nu_split[1]))
        orb_plt_lines.append(orb_lb)
        orb_plt_lbls.append(str(nu_split[0]) + r"$< \nu <$" + str(nu_split[1]))
        inf_plt_lines.append(inf_lb)
        inf_plt_lbls.append(str(nu_split[0]) + r"$< \nu <$" + str(nu_split[1]))
        
        # Plot the SPARTA (actual) profiles 
        all_ax_0.plot(bins, prf_func(all_prfs[i][1],axis=0), linestyle='--', color = all_colors[i])
        orb_ax_0.plot(bins, prf_func(orb_prfs[i][1],axis=0), linestyle='--', color = orb_colors[i])
        inf_ax_0.plot(bins, prf_func(inf_prfs[i][1],axis=0), linestyle='--', color = inf_colors[i])

        all_ax_1.plot(bins, prf_func(ratio_all_prf,axis=0), color = all_colors[i])
        orb_ax_1.plot(bins, prf_func(ratio_orb_prf,axis=0), color = orb_colors[i])
        inf_ax_1.plot(bins, prf_func(ratio_inf_prf,axis=0), color = inf_colors[i])
        
        all_ax_1.fill_between(bins, np.nanpercentile(ratio_all_prf, q=15.9, axis=0),np.nanpercentile(ratio_all_prf, q=84.1, axis=0), color=all_colors[i], alpha=fill_alpha)
        orb_ax_1.fill_between(bins, np.nanpercentile(ratio_orb_prf, q=15.9, axis=0),np.nanpercentile(ratio_orb_prf, q=84.1, axis=0), color=orb_colors[i], alpha=fill_alpha)
        inf_ax_1.fill_between(bins, np.nanpercentile(ratio_inf_prf, q=15.9, axis=0),np.nanpercentile(ratio_inf_prf, q=84.1, axis=0), color=inf_colors[i], alpha=fill_alpha)
                
        
    all_ax_0.set_ylabel(r"$\rho / \rho_m$", fontsize=axisfntsize)
    all_ax_0.set_xscale("log")
    all_ax_0.set_yscale("log")
    all_ax_0.set_xlim(0.05,np.max(lin_rticks))
    all_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
    all_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
    all_ax_0.legend(all_plt_lines,all_plt_lbls,fontsize=legendfntsize, loc = "upper right")
    all_ax_0.text(0.05,0.05, "All Particles", ha="left", va="bottom", transform=all_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})
    
    orb_ax_0.set_xscale("log")
    orb_ax_0.set_yscale("log")
    orb_ax_0.set_xlim(0.05,np.max(lin_rticks))
    orb_ax_0.set_ylim(all_ax_0.get_ylim())
    orb_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
    orb_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
    orb_ax_0.tick_params(axis='y', which='both', labelleft=False)
    orb_ax_0.legend(orb_plt_lines,orb_plt_lbls,fontsize=legendfntsize, loc = "upper right")
    orb_ax_0.text(0.05,0.05, "Orbiting Particles", ha="left", va="bottom", transform=orb_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})
    
    inf_ax_0.set_xscale("log")
    inf_ax_0.set_yscale("log")
    inf_ax_0.set_xlim(0.05,np.max(lin_rticks))
    inf_ax_0.set_ylim(all_ax_0.get_ylim())
    inf_ax_0.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
    inf_ax_0.tick_params(axis='x', which='both', labelbottom=False) # we don't want the labels just the tick marks
    inf_ax_0.tick_params(axis='y', which='both', labelleft=False) 
    inf_ax_0.legend(inf_plt_lines,inf_plt_lbls,fontsize=legendfntsize, loc = "upper right")
    inf_ax_0.text(0.05,0.05, "Infalling Particles", ha="left", va="bottom", transform=inf_ax_0.transAxes, fontsize=textfntsize, bbox={"facecolor":'white',"alpha":0.9,})

    all_y_min, all_y_max = all_ax_0.get_ylim()
    orb_y_min, orb_y_max = orb_ax_0.get_ylim()
    inf_y_min, inf_y_max = inf_ax_0.get_ylim()

    global_y_min = min(all_y_min, orb_y_min, inf_y_min)
    global_y_max = max(all_y_max, orb_y_max, inf_y_max)

    # Set the same y-axis limits for all axes
    all_ax_0.set_ylim(0.1, global_y_max)
    orb_ax_0.set_ylim(0.1, global_y_max)
    inf_ax_0.set_ylim(0.1, global_y_max)

    
    # fig.legend([(invis_calc, invis_act),all_lb,orb_lb,inf_lb], ['Predicted, Actual','All','Orbiting','Infalling'], numpoints=1,handlelength=3,loc='upper left',bbox_to_anchor=(0.05, 0.97),handler_map={tuple: HandlerTuple(ndivide=None)},frameon=False,fontsize=legendfntsize)

    all_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
    all_ax_1.set_ylabel(r"$\frac{\rho_{pred}}{\rho_{act}} - 1$", fontsize=axisfntsize)
    orb_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
    inf_ax_1.set_xlabel(r"$r/R_{200m}$", fontsize=axisfntsize)
    
    all_ax_1.set_xlim(0.05,np.max(lin_rticks))
    all_ax_1.set_ylim(bottom=-0.3,top=0.3)
    all_ax_1.set_xscale("log")
    
    orb_ax_1.set_xlim(0.05,np.max(lin_rticks))
    orb_ax_1.set_ylim(bottom=-0.3,top=0.3)
    orb_ax_1.set_xscale("log")
    
    inf_ax_1.set_xlim(0.05,np.max(lin_rticks))
    inf_ax_1.set_ylim(bottom=-0.3,top=0.3)
    inf_ax_1.set_xscale("log")
    
    tick_locs = lin_rticks.copy()
    if 0 in tick_locs:
        tick_locs.remove(0)
    #TODO remove this and have it so ticks are specific to which plot is being made
    if 0.1 not in tick_locs:
        tick_locs.append(0.1)
        tick_locs = sorted(tick_locs)
    strng_ticks = list(map(str, tick_locs))
    strng_ticks = list(map(str, tick_locs))

    all_ax_1.set_xticks(tick_locs,strng_ticks)  
    all_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize)
    
    orb_ax_1.set_xticks(tick_locs,strng_ticks)
    orb_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize, labelleft=False)
    
    inf_ax_1.set_xticks(tick_locs,strng_ticks)  
    inf_ax_1.tick_params(axis='both',which='both',direction="in",labelsize=tickfntsize, labelleft=False)
    if use_med:
        fig.savefig(save_location + title + "med_prfl_rat_nu.png",bbox_inches='tight',dpi=300)
    else:
        fig.savefig(save_location + title + "avg_prfl_rat_nu.png",bbox_inches='tight')
        
def inf_orb_frac(p_corr_labels,p_r,p_rv,p_tv,c_r,c_rv,split_scale_dict,num_bins,save_loc):
    linthrsh = split_scale_dict["linthrsh"]
    
    p_r_range = [np.min(p_r),np.max(p_r)]
    p_rv_range = [np.min(p_rv),np.max(p_rv)]
    p_tv_range = [np.min(p_tv),np.max(p_tv)]
    
    act_min_ptl = 10
    set_ptl = 0
    scale_min_ptl = 1e-4
    
    inf_p_r, orb_p_r = split_orb_inf(p_r,p_corr_labels)
    inf_p_rv, orb_p_rv = split_orb_inf(p_rv,p_corr_labels)
    inf_p_tv, orb_p_tv = split_orb_inf(p_tv,p_corr_labels)
    inf_c_r, orb_c_r = split_orb_inf(c_r,p_corr_labels)
    inf_c_rv, orb_c_rv = split_orb_inf(c_rv,p_corr_labels)
    
    
    # Use the binning from all particles for the orbiting and infalling plots and the secondary snap to keep it consistent
    hist_inf_p_r_p_rv = histogram(inf_p_r,inf_p_rv,use_bins=[num_bins,num_bins],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
    hist_inf_p_r_p_tv = histogram(inf_p_r,inf_p_tv,use_bins=[num_bins,num_bins],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
    hist_inf_p_rv_p_tv = histogram(inf_p_rv,inf_p_tv,use_bins=[num_bins,num_bins],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
    hist_inf_c_r_c_rv = histogram(inf_c_r,inf_c_rv,use_bins=[hist_inf_p_r_p_rv["x_edge"],hist_inf_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
    hist_orb_p_r_p_rv = histogram(orb_p_r,orb_p_rv,use_bins=[num_bins,num_bins],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
    hist_orb_p_r_p_tv = histogram(orb_p_r,orb_p_tv,use_bins=[num_bins,num_bins],hist_range=[p_r_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)
    hist_orb_p_rv_p_tv = histogram(orb_p_rv,orb_p_tv,use_bins=[num_bins,num_bins],hist_range=[p_rv_range,p_tv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_xscale_dict=split_scale_dict,split_yscale_dict=split_scale_dict)
    hist_orb_c_r_c_rv = histogram(orb_c_r,orb_c_rv,use_bins=[hist_orb_p_r_p_rv["x_edge"],hist_orb_p_r_p_rv["y_edge"]],hist_range=[p_r_range,p_rv_range],min_ptl=act_min_ptl,set_ptl=set_ptl,split_yscale_dict=split_scale_dict)

    hist_frac_p_r_p_rv = scale_hists(hist_inf_p_r_p_rv, hist_orb_p_r_p_rv)
    hist_frac_p_r_p_tv = scale_hists(hist_inf_p_r_p_tv, hist_orb_p_r_p_tv)
    hist_frac_p_rv_p_tv = scale_hists(hist_inf_p_rv_p_tv, hist_orb_p_rv_p_tv)
    hist_frac_c_r_c_rv = scale_hists(hist_inf_c_r_c_rv, hist_orb_c_r_c_rv)
    
    max_ptl = np.max(np.array([np.max(hist_frac_p_r_p_rv["hist"]),np.max(hist_frac_p_r_p_tv["hist"]),np.max(hist_frac_p_rv_p_tv["hist"]),np.max(hist_frac_c_r_c_rv["hist"])]))
        
    cividis_cmap = plt.get_cmap("cividis")
    cividis_cmap.set_under(color='black')
    cividis_cmap.set_bad(color='black') 
    
    plot_kwargs = {
            "vmin":scale_min_ptl,
            "vmax":max_ptl,
            "norm":"log",
            "origin":"lower",
            "aspect":"auto",
            "cmap":cividis_cmap,
    }
    
    r_ticks = split_scale_dict["lin_rticks"] + split_scale_dict["log_rticks"]
    
    rv_ticks = split_scale_dict["lin_rvticks"] + split_scale_dict["log_rvticks"]
    rv_ticks = rv_ticks + [-x for x in rv_ticks if x != 0]
    rv_ticks.sort()

    tv_ticks = split_scale_dict["lin_tvticks"] + split_scale_dict["log_tvticks"]       
    
    widths = [4,4,4,4,.2]
    heights = [4]
    
    fig = plt.figure(constrained_layout=True, figsize=(45,10))
    gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
    
    imshow_plot(fig.add_subplot(gs[0,0]),hist_frac_p_r_p_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,kwargs=plot_kwargs)
    imshow_plot(fig.add_subplot(gs[0,1]),hist_frac_p_r_p_tv,x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$",xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,kwargs=plot_kwargs)
    imshow_plot(fig.add_subplot(gs[0,2]),hist_frac_p_rv_p_tv,x_label="$v_r/V_{200m}$",hide_ytick_labels=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,kwargs=plot_kwargs)
    imshow_plot(fig.add_subplot(gs[0,3]),hist_frac_c_r_c_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,kwargs=plot_kwargs)
 
    color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=scale_min_ptl, vmax=max_ptl),cmap=cividis_cmap), cax=plt.subplot(gs[:,-1]))
    color_bar.set_label(r"$N_{inf}/N_{orb}$",fontsize=26)
    color_bar.ax.tick_params(which="major",direction="in",labelsize=22,length=10,width=3)
    color_bar.ax.tick_params(which="minor",direction="in",labelsize=22,length=5,width=1.5)
    
    fig.savefig(save_loc + "inf_orb_frac.png")
    plt.close()
    
def plot_tree(bst,tree_num,save_loc):
    fig, ax = plt.subplots(figsize=(400, 10))
    xgb.plot_tree(bst, num_trees=tree_num, ax=ax,rankdir='LR')
    fig.savefig(save_loc + "/tree_plot.png")
    