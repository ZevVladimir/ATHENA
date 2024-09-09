import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, NullFormatter
from utils.data_and_loading_functions import split_orb_inf, timed
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

def imshow_plot(ax, img, x_label="", y_label="", text="", title="", hide_xticks=False, hide_yticks=False, xticks = None, yticks = None, xlinthrsh = None, ylinthrsh = None, axisfontsize=26, number = None, return_img=False, kwargs={}):
    plt.xticks(rotation=70)

    ret_img=ax.imshow(img["hist"].T, interpolation="nearest", **kwargs)
    ax.tick_params(axis="both",which="major",length=6,width=2)
    ax.tick_params(axis="both",which="minor",length=4,width=1.5)
    xticks_loc = []
    yticks_loc = []
    
    for tick in xticks:
        xticks_loc.append(get_bin_loc(img["x_edge"],tick))
    for tick in yticks:
        yticks_loc.append(get_bin_loc(img["y_edge"],tick))
    
    if not hide_xticks:
        ax.set_xticks(xticks_loc,xticks)
        ax.tick_params(axis="x",direction="in")
    if not hide_yticks:
        ax.set_yticks(yticks_loc,yticks)
        ax.tick_params(axis="y",direction="in")
        
    if ylinthrsh != None:
        ylinthrsh_loc = get_bin_loc(img["y_edge"],ylinthrsh)
        ax.axhline(y=ylinthrsh_loc, color='grey', linestyle='--', alpha=1)
        if np.where(np.array(yticks,dtype=np.float32) < 0)[0].size > 0:
            neg_ylinthrsh_loc = get_bin_loc(img["y_edge"],-ylinthrsh)
            ax.axhline(y=neg_ylinthrsh_loc, color='grey', linestyle='--', alpha=1)
    
    if xlinthrsh != None:
        xlinthrsh_loc = get_bin_loc(img["x_edge"],xlinthrsh)
        ax.axvline(x=xlinthrsh_loc, color='grey', linestyle='--', alpha=1)
        if np.where(np.array(xticks,dtype=np.float32) < 0)[0].size > 0:
            neg_xlinthrsh_loc = get_bin_loc(img["x_edge"],-xlinthrsh)
            ax.axvline(x=neg_xlinthrsh_loc, color='grey', linestyle='--', alpha=1)

    if text != "":
        ax.text(0.01,0.03, text, ha="left", va="bottom", transform=ax.transAxes, fontsize=axisfontsize, bbox={"facecolor":'white',"alpha":0.9,})
    if number != None:
        ax.text(0.02,0.93,number,ha="left",va="bottom",transform=ax.transAxes,fontsize=axisfontsize,bbox={"facecolor":'white',"alpha":0.9,})
    if title != "":
        ax.set_title(title,fontsize=28)
    if x_label != "":
        ax.set_xlabel(x_label,fontsize=axisfontsize)
    if y_label != "":
        ax.set_ylabel(y_label,fontsize=axisfontsize)
    if hide_xticks:
        ax.tick_params(axis='x', which='both',bottom=False,labelbottom=False)
    else:
        ax.tick_params(axis='x', which='major', labelsize=22)
        ax.tick_params(axis='x', which='minor', labelsize=20)
         
    if hide_yticks:
        ax.tick_params(axis='y', which='both',left=False,labelleft=False)
    else:
        ax.tick_params(axis='y', which='major', labelsize=22)
        ax.tick_params(axis='y', which='minor', labelsize=20)
           
    if return_img:
        return ret_img

# Uses np.histogram2d to create a histogram and the edges of the histogram in one dictionary
# Can also do a linear binning then a logarithmic binning (similar to symlog) but allows for 
# special case of only positive log and not negative log
def histogram(x,y,use_bins,hist_range,min_ptl,set_ptl,split_xscale_dict=None,split_yscale_dict=None):
    if split_yscale_dict != None:
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
        
    if split_xscale_dict != None:
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

def scale_hists(inc_hist, act_hist, act_min, inc_min):
    scaled_hist = {
        "x_edge":act_hist["x_edge"],
        "y_edge":act_hist["y_edge"]
    }
    scaled_hist["hist"] = np.divide(inc_hist["hist"],act_hist["hist"],out=np.zeros_like(inc_hist["hist"]), where=act_hist["hist"]!=0)
    
    scaled_hist["hist"] = np.where((inc_hist["hist"] < 1) & (act_hist["hist"] >= act_min), inc_min, scaled_hist["hist"])
    # Where there are miss classified particles but they won't show up on the image, set them to the min
    scaled_hist["hist"] = np.where((inc_hist["hist"] >= 1) & (scaled_hist["hist"] < inc_min) & (act_hist["hist"] >= act_min), inc_min, scaled_hist["hist"])
    
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

def plot_full_ptl_dist(p_corr_labels, p_r, p_rv, p_tv, c_r, c_rv, split_scale_dict, num_bins, save_loc):
    with timed("Finished Full Ptl Dist Plot"):
        print("Starting Full Ptl Dist Plot")
        
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
        cividis_cmap.set_under(color='white')  
        
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
        
        widths = [4,4,4,4,.5]
        heights = [0.15,4,4,4] # have extra row up top so there is space for the title
        
        fig = plt.figure(constrained_layout=True, figsize=(35,25))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)
        
        imshow_plot(fig.add_subplot(gs[1,0]),all_p_r_p_rv,y_label="$v_r/v_{200m}$",text="All Particles",title="Primary Snap",hide_xticks=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D1",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[1,1]),all_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xticks=True,xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="D2",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[1,2]),all_p_rv_p_tv,hide_xticks=True,hide_yticks=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="D3",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[1,3]),all_c_r_c_rv,y_label="$v_r/v_{200m}$",title="Secondary Snap",hide_xticks=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D4",kwargs=plot_kwargs)
        
        imshow_plot(fig.add_subplot(gs[2,0]),inf_p_r_p_rv,y_label="$v_r/v_{200m}$",text="Infalling Particles",hide_xticks=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D5",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[2,1]),inf_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xticks=True,xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="D6",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[2,2]),inf_p_rv_p_tv,hide_xticks=True,hide_yticks=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="D7",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[2,3]),inf_c_r_c_rv,y_label="$v_r/v_{200m}$",hide_xticks=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D8",kwargs=plot_kwargs)
                    
        imshow_plot(fig.add_subplot(gs[3,0]),orb_p_r_p_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",text="Orbiting Particles",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D9",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[3,1]),orb_p_r_p_tv,x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$",xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="D10",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[3,2]),orb_p_rv_p_tv,x_label="$v_r/v_{200m}$",hide_yticks=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="D11",kwargs=plot_kwargs)
        imshow_plot(fig.add_subplot(gs[3,3]),orb_c_r_c_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="D12",kwargs=plot_kwargs)

        color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=scale_min_ptl, vmax=max_ptl),cmap=cividis_cmap), cax=plt.subplot(gs[1:,-1]))
        color_bar.set_label(r"$dN / N dx dy$",fontsize=26)
        color_bar.ax.tick_params(which="major",direction="in",labelsize=22,length=10,width=3)
        color_bar.ax.tick_params(which="minor",direction="in",labelsize=22,length=5,width=1.5)
        
        fig.savefig(save_loc + "ptl_distr.png")
        plt.close()

def plot_miss_class_dist(p_corr_labels, p_ml_labels, p_r, p_rv, p_tv, c_r, c_rv, split_scale_dict, num_bins, save_loc, model_info,dataset_name):
    with timed("Finished Miss Class Dist Plot"):
        print("Starting Miss Class Dist Plot")

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
        
        magma_cmap = plt.get_cmap("magma_r")
        magma_cmap.set_under(color='white')
        
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
        
        fig = plt.figure(constrained_layout=True,figsize=(35,25))
        gs = fig.add_gridspec(len(heights),len(widths),width_ratios = widths, height_ratios = heights, hspace=0, wspace=0)

        imshow_plot(fig.add_subplot(gs[1,0]),scale_inc_all_p_r_p_rv,y_label="$v_r/v_{200m}$",hide_xticks=True,text="All Misclassified Scaled",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S1",kwargs=scale_miss_class_args, title="Primary Snap")
        imshow_plot(fig.add_subplot(gs[1,1]),scale_inc_all_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xticks=True,xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="S2",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[1,2]),scale_inc_all_p_rv_p_tv,hide_xticks=True,hide_yticks=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="S3",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[1,3]),scale_inc_all_c_r_c_rv,y_label="$v_r/v_{200m}$",hide_xticks=True,text="All Misclassified Scaled",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S4",kwargs=scale_miss_class_args, title="Secondary Snap")

        imshow_plot(fig.add_subplot(gs[2,0]),scale_inc_inf_p_r_p_rv,hide_xticks=True,y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S5",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[2,1]),scale_inc_inf_p_r_p_tv,y_label="$v_t/v_{200m}$",hide_xticks=True,xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="S6",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[2,2]),scale_inc_inf_p_rv_p_tv,hide_xticks=True,hide_yticks=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="S7",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[2,3]),scale_inc_inf_c_r_c_rv,y_label="$v_r/v_{200m}$",text="Label: Orbit\nReal: Infall",hide_xticks=True,xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S8",kwargs=scale_miss_class_args)
        
        imshow_plot(fig.add_subplot(gs[3,0]),scale_inc_orb_p_r_p_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S9",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[3,1]),scale_inc_orb_p_r_p_tv,x_label="$r/R_{200m}$",y_label="$v_t/v_{200m}$",xticks=r_ticks,yticks=tv_ticks,ylinthrsh=linthrsh,number="S10",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[3,2]),scale_inc_orb_p_rv_p_tv,x_label="$v_r/v_{200m}$",hide_yticks=True,xticks=rv_ticks,yticks=tv_ticks,xlinthrsh=linthrsh,ylinthrsh=linthrsh,number="S11",kwargs=scale_miss_class_args)
        imshow_plot(fig.add_subplot(gs[3,3]),scale_inc_orb_c_r_c_rv,x_label="$r/R_{200m}$",y_label="$v_r/v_{200m}$",text="Label: Infall\nReal: Orbit",xticks=r_ticks,yticks=rv_ticks,ylinthrsh=linthrsh,number="S12",kwargs=scale_miss_class_args)
        
        color_bar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=inc_min_ptl, vmax=1),cmap=magma_cmap), cax=plt.subplot(gs[1:,-1]))
        color_bar.set_label(r"$N_{\mathrm{bin, inc}} / N_{\mathrm{bin, tot}}$",fontsize=26)
        color_bar.ax.tick_params(which="major",direction="in",labelsize=22,length=10,width=3)
        color_bar.ax.tick_params(which="minor",direction="in",labelsize=22,length=5,width=1.5)
        
        fig.savefig(save_loc + "scaled_miss_class.png")
        plt.close()

def plot_perr_err():
    return
