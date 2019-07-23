import matplotlib
import matplotlib.font_manager
import seaborn as sns

def update_style():
    sns.set_style("whitegrid")

    matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    font_color = 'black'

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times',
        #'figure.figsize': (6.4, 4.8),
        'errorbar.capsize': 10,
        #'text.usetex': 'true',
        'font.family': 'sans-serif',
        'font.sans-serif': 'DejaVu Serif',
        'font.size': 12,
        'text.color': font_color,
        'axes.edgecolor': font_color,
        'axes.labelcolor': font_color,
        'xtick.color': font_color,
        'ytick.color': font_color,
    })
