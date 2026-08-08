"""Standardized grid plotting for noise-vs-method time series.

A 'plot group' is a 2D grid of axes where:
    * rows    -> different methods (e.g. r_space.type)
    * columns -> different noise levels (e.g. action_noise / transition_noise)
    * each cell draws one line per `hue` value (e.g. replay_ratio),
      showing mean +/- std over seeds.

The core function `plot_time_series_grid` either creates its own Figure or draws
into a pre-existing axes grid, so multiple groups can be composed into a single
bigger figure via `fig.subfigures(...)`.

Typical usage (single group)::

    from plotting.time_series_grid import plot_time_series_grid, GridSpec

    plot_time_series_grid(
        data=data,
        metric_col="corrected_plasticity",
        row=GridSpec("learn.r_space.type", ["none", "ln"], labels=["None", "LN"]),
        col=GridSpec("task.action_noise", [0, 0.05, 0.1, 0.2], title_fmt="noise={}"),
        hue=GridSpec("learn.replay_ratio", [1, 4, 8]),
        fixed={"task.env_id": "Ant-v5"},
    )

Composing multiple groups (e.g. one per env) into one figure::

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 3.5), constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    for sf, env in zip(subfigs, envs):
        axes = sf.subplots(2, 4, sharex=True, sharey="row")
        plot_time_series_grid(..., fig=sf, axes=axes, fixed={"task.env_id": env})
        sf.suptitle(env)
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from exputils.data.loading import filter_df_by_dict
from core.plotting.grid_spec import GridSpec, HueSpec
from core.plotting.utils import (
    create_global_legend,
)


def plot_lineplot_grid(
    data: pd.DataFrame,
    *,
    row: GridSpec | None = None,
    col: GridSpec | None = None,
    hue: HueSpec,
    fixed: dict | None = None,
    fig: Figure | None = None,
    figsize: tuple[float, float] = (5.5, 3.5),
    sharex: bool = True,
    sharey: bool | Literal["none", "all", "row", "col"] = "row",
    x_lims: tuple[float, float] | None = None,
    y_lims: tuple[float | None, float | None] | None = None,
    log_scale_y: bool = False,
    log_scale_x: bool = False,
    mean_col: str = "mean",
    err_low: str = "std",
    err_high: str = "std",
    plot_err: bool = True,
    row_ylabel: bool = True,
    col_title: bool = True,
    constrained_layout: bool = True,
    x_dim: str = "step",
    x_label: str | None = "steps",
    x_label_y: float = -0.02,
    y_label: str | None = None,
    y_label_x: float = -0.02,
    global_legend: bool = False,
    legend_title: str | None = None,
    legend_ncol: int | None = None,
    legend_placement: Literal["below", "right"] = "below",
    marker: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Draw a rows-x-cols grid of mean/std time series.

    If `fig` is None, a new Figure is created. If `fig` is passed and already
    contains axes, their shape must match `(len(row.values), len(col.values))`
    — otherwise a `ValueError` is raised. If `fig` has no axes yet, the grid
    is created on it via `fig.subplots(...)`.

    Returns:
        (fig, axes) where axes has shape (len(row.values), len(col.values)).
    """
    _row_values = row.values if row is not None else [None]
    _col_values = col.values if col is not None else [None]
    n_rows = len(_row_values)
    n_cols = len(_col_values)

    if fig is None:
        fig, axes_arr = plt.subplots(
            n_rows,
            n_cols,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            constrained_layout=constrained_layout,
            squeeze=False,
        )
        axes = np.asarray(axes_arr).reshape(n_rows, n_cols)
    else:
        existing = fig.get_axes()
        if not existing:
            axes_arr = fig.subplots(
                n_rows,
                n_cols,
                sharex=sharex,
                sharey=sharey,
                squeeze=False,
            )
            axes = np.asarray(axes_arr).reshape(n_rows, n_cols)
        else:
            if len(existing) != n_rows * n_cols:
                raise ValueError(
                    f"Provided figure has {len(existing)} axes but grid requires {n_rows * n_cols} ({n_rows} rows x {n_cols} cols)."
                )
            axes = np.asarray(existing).reshape(n_rows, n_cols)
    assert fig is not None and axes is not None

    colours = hue.get_colors()
    linestyles = hue.get_linestyles()
    fixed = fixed or {}

    for i, row_val in enumerate(_row_values):
        for j, col_val in enumerate(_col_values):
            ax: plt.Axes = axes[i, j]
            for colour, linestyle, hue_val in zip(colours, linestyles, hue.values):
                flt = {**fixed, hue.key: hue_val}
                if row is not None:
                    flt[row.key] = row_val
                if col is not None:
                    flt[col.key] = col_val
                ts = filter_df_by_dict(data, flt)
                ts = ts[ts[mean_col].notna()]
                ts = ts.sort_values(x_dim)
                x = ts[x_dim]
                y = ts[mean_col]

                if ts.empty:
                    continue
                ax.plot(
                    x,
                    y,
                    label=str(hue_val),
                    color=colour,
                    marker=marker,
                    # linestyle=linestyle,
                )
                if plot_err:
                    if err_low == err_high:
                        low = y - ts[err_low]
                        high = y + ts[err_high]
                    else:
                        low = ts[err_low]
                        high = ts[err_high]
                    ax.fill_between(
                        x,
                        low,
                        high,
                        color=colour,
                        alpha=0.3,
                    )

            if log_scale_y:
                ax.set_yscale("log")
            if log_scale_x:
                ax.set_xscale("log")
            if log_scale_x or log_scale_y:
                ax.minorticks_off()
            if x_lims is not None:
                ax.set_xlim(*x_lims)

    if y_lims is not None:
        for ax in axes.flatten():
            ax.set_ylim(*y_lims)

    if row_ylabel and row is not None:
        for i in range(n_rows):
            axes[i, 0].set_ylabel(row.label(i))
    if col_title and col is not None:
        for j in range(n_cols):
            axes[0, j].set_title(col.label(j))

    if x_label is not None:
        fig.supxlabel(x_label, y=x_label_y)

    if y_label is not None:
        fig.supylabel(y_label, x=y_label_x)

    if global_legend:
        create_global_legend(
            fig,
            title=legend_title,
            ncol=legend_ncol,
            placement=legend_placement,
        )

    return fig, axes
