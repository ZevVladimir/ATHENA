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
    def __init__(self, plot_types, plot_size, X, Y, x_label = None, y_label = None, title = None, color = None, save_location = None, save = True, show = True):
        self.plot_type = plot_types
        self.plot_size = plot_size
        self.X = X
        self.Y = Y
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.color = color
        self.save_location = save_location
        self.save = save
        self.show = show

    def __init__(self, plot_types, plot_size, X, Y, x_label = None, y_label = None, title = None, cmap = None, save_location = None, save = True, show = True):
        self.plot_type = plot_types
        self.plot_size = plot_size
        self.X = X
        self.Y = Y
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.cmap = cmap
        self.save_location = save_location
        self.save = save
        self.show = show
    
    def plot(self):
        fig, ax = plt.subplots(self.plot_size)

        curr_plot_num = 0

        for row in range(self.plot[0]):
            for col in range(self.plot[1]):
                curr_plot = self.plot_types[curr_plot_num]
                ax.curr_plot()
