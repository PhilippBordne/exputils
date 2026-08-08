from .utils import create_global_legend
from .styles import FONTSIZES, STYLE_TEMPLATE
from .grid_lineplot import plot_lineplot_grid
from .grid_barplot import plot_barplot_grid
from .grid_scatterplot import plot_scatterplot_grid
from .grid_spec import GridSpec, HueSpec


import matplotlib.pyplot as plt

plt.style.use(STYLE_TEMPLATE)
plt.rcParams.update(FONTSIZES)

__all__ = [
    "create_global_legend",
    "FONTSIZES",
    "plot_lineplot_grid",
    "plot_barplot_grid",
    "plot_scatterplot_grid",
    "GridSpec",
    "HueSpec",
]
