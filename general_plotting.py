import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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
    def __init__(self, plot_types, plot_size, X, Y, xlim = None, ylim = None, x_label = None, y_label = None, fig_title = None, subplot_title = None, save_location = None, save = False, show = False, kwargs = None):
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
        self.save_location = save_location
        self.save = save
        self.show = show
        self.kwargs = kwargs
    
    def plot(self):
        fig, ax = plt.subplots(self.plot_size[0], self.plot_size[1])
        fig.suptitle(self.fig_title)

        curr_plot_num = 0

        for row in range(self.plot_size[0]):
            for col in range(self.plot_size[1]):
                for line in range(self.X.shape[1]):
                    if self.plot_size[0] > 1:
                        ax[row][col].__getattribute__(self.plot_types[curr_plot_num])(self.X[:,line,curr_plot_num], self.Y[:,line,curr_plot_num], **self.kwargs) 
                    else:
                        ax[col].__getattribute__(self.plot_types[curr_plot_num])(self.X[:,line,curr_plot_num], self.Y[:,line,curr_plot_num], **self.kwargs) 

                if self.xlim != None:
                    ax[row][col].set_xlim(self.xlim)
                if self.ylim != None:
                    ax[row][col].set_ylim(self.ylim)
                if self.x_label != None:
                    ax[row][col].set_xlabel(self.x_label[curr_plot_num])
                if self.y_label != None:
                    ax[row][col].set_ylabel(self.y_label[curr_plot_num])
                if self.subplot_title != None:
                    ax[row][col].set_title(self.subplot_title[curr_plot_num])

        if self.save:
            fig.savefig(self.save_location)
        if self.show:
            plt.show()

        return fig