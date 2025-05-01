import matplotlib.pyplot as plt
import matplotlib as mpl

def set_plt_defaults(font_size=12):
    """
    Set default plotting styles for matplotlib/pyplot.
    
    Args:
        font_size (int): Font size for all text elements (default: 12)
    """
    # Set the default color cycle to start with light blue and orange
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
                                                ['#8cc5e3', '#ea801c'])
    
    # Set font sizes
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.titlesize'] = font_size
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams['legend.fontsize'] = font_size
    
    # Optional: Improve overall aesthetics
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
