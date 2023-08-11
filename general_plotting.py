import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Take in:
# 1. Type of plot (list of funcs)
# 2. Data (as np array or pd dataframe)
# 3. Labels (as lists to allow compatibility for subplots)
#   a. x axis
#   b. y axis
#   c. title
# 4. Subplot layout (tuple)
# 5. Colors
# 6. Cmap (if needed)
# 7. Bins (if needed)

class plot_determiner:
    def __init__(self, plot_types, plot_size, X, Y, xlim = None, ylim = None, x_label = None, y_label = None, fig_title = None, subplot_title = None, constrained  = False, save_location = None, save = False, show = False, args = [], kwargs = {}):
        self.plot_types = plot_types
        self.plot_size = plot_size
        self.X = X
        self.Y = Y
        self.xlim = xlim
        self.ylim = ylim
        self.x_label = x_label
        self.y_label = y_label
        self.fig_title = fig_title
        self.subplot_title = subplot_title
        self.constrained = constrained
        self.save_location = save_location
        self.save = save
        self.show = show
        self.args = args
        self.kwargs = kwargs
    
    def plot(self):
        norm = mpl.colors.Normalize(vmin=self.kwargs["vmin"], vmax=self.kwargs["vmax"])
        if self.constrained:
            fig = plt.figure(constrained_layout=True)
        else:
            fig = plt.figure()

        fig.suptitle(self.fig_title)
        subfigs = fig.subfigures(nrows=self.plot_size[0], ncols=1)

        curr_plot_num = 0

        for row,subfig in enumerate(subfigs):
            subfig.suptitle(self.subplot_title[row])

            axs = subfig.subplots(nrows=1, ncols=self.plot_size[1])
            for col,ax in enumerate(axs):
                for line in range(self.X.shape[1]):
                    if self.plot_size[0] > 1:
                        hist = ax.__getattribute__(self.plot_types[curr_plot_num])(self.X[:,line,curr_plot_num], self.Y[:,line,curr_plot_num], *self.args[curr_plot_num], **self.kwargs) 
                        if self.xlim != None:
                            ax.set_xlim(self.xlim)
                        if self.ylim != None:
                            ax.set_ylim(self.ylim)
                        if self.x_label != None:
                            ax.set_xlabel(self.x_label[curr_plot_num])
                        if self.y_label != None:
                            ax.set_ylabel(self.y_label[curr_plot_num])
                        
                    else:
                        hist = ax.__getattribute__(self.plot_types[curr_plot_num])(self.X[:,line,curr_plot_num], self.Y[:,line,curr_plot_num], *self.args[curr_plot_num], **self.kwargs) 
                        if self.xlim != None:
                            ax.set_xlim(self.xlim)
                        if self.ylim != None:
                            ax.set_ylim(self.ylim)
                        if self.x_label != None:
                            ax.set_xlabel(self.x_label[curr_plot_num])
                        if self.y_label != None:
                            ax.set_ylabel(self.y_label[curr_plot_num])
                
                curr_plot_num += 1
    
            axcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=self.kwargs["cmap"]), ax=axs.ravel().tolist(), pad=0.04, aspect = 30)

        if self.save:
            fig.savefig(self.save_location)
        if self.show:
            plt.show()

        return fig