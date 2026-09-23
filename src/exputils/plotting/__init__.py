import matplotlib.pyplot as plt

from .grid_barplot import plot_barplot_grid
from .grid_lineplot import plot_lineplot_grid
from .grid_scatterplot import plot_scatterplot_grid
from .grid_spec import GridSpec, HueSpec
from .styles import FONTSIZES, STYLE_TEMPLATE
from .utils import create_global_legend

plt.style.use(STYLE_TEMPLATE)
plt.rcParams.update(FONTSIZES)  # type: ignore

__all__ = [
    "FONTSIZES",
    "GridSpec",
    "HueSpec",
    "create_global_legend",
    "plot_barplot_grid",
    "plot_lineplot_grid",
    "plot_scatterplot_grid",
]
