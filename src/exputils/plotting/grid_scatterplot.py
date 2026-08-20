"""Standardized grid plotting for noise-vs-method scatter plots.

A 'plot group' is a 2D grid of axes where:
    * rows    -> different methods (e.g. r_space.type)
    * columns -> different noise levels (e.g. action_noise / transition_noise)
    * each cell draws one scatter cloud per `hue` value (e.g. replay_ratio).

The core function `plot_scatterplot_grid` either creates its own Figure or draws
into a pre-existing axes grid, so multiple groups can be composed into a single
bigger figure via `fig.subfigures(...)`.

Typical usage (single group)::

    from plotting.grid_scatterplot import plot_scatterplot_grid, GridSpec

    plot_scatterplot_grid(
        data=data,
        x_col="corrected_plasticity",
        y_col="return",
        row=GridSpec("learn.r_space.type", ["none", "ln"], labels=["None", "LN"]),
        col=GridSpec("task.action_noise", [0, 0.05, 0.1, 0.2], title_fmt="noise={}"),
        hue=HueSpec("learn.replay_ratio", [1, 4, 8]),
        fixed={"task.env_id": "Ant-v5"},
    )
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.optimize import minimize

from exputils.data.loading import filter_df_by_dict
from exputils.plotting.grid_spec import GridSpec, HueSpec
from exputils.plotting.utils import create_global_legend


def _fit_line(x: np.ndarray, y: np.ndarray, loss: Literal["l1", "l2"]) -> tuple[float, float]:
    """Return (slope, intercept) for a linear fit using the specified loss."""
    if loss == "l2":
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)
    result = minimize(
        lambda p: np.sum(np.abs(y - p[0] * x - p[1])),
        x0=[0.0, float(np.median(y))],
        method="Nelder-Mead",
    )
    return float(result.x[0]), float(result.x[1])


def plot_scatterplot_grid(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    row: GridSpec | None = None,
    col: GridSpec | None = None,
    hue: HueSpec,
    fixed: dict | None = None,
    fig: Figure | None = None,
    figsize: tuple[float, float] = (5.5, 3.5),
    sharex: bool = True,
    sharey: bool | Literal["none", "all", "row", "col"] = "row",
    x_lims: tuple[float | None, float | None] | None = None,
    y_lims: tuple[float | None, float | None] | None = None,
    log_scale_y: bool = False,
    log_scale_x: bool = False,
    marker_size: float = 2.0,
    regression: Literal["none", "per_hue", "across_hue"] = "none",
    regression_loss: Literal["l1", "l2"] = "l2",
    alpha: float = 0.7,
    row_ylabel: bool = True,
    col_title: bool = True,
    constrained_layout: bool = True,
    x_label: str | None = None,
    x_label_y: float = -0.02,
    y_label: str | None = None,
    y_label_x: float = -0.02,
    global_legend: bool = False,
    legend_title: str | None = None,
    legend_ncol: int | None = None,
    legend_placement: Literal["below", "right"] = "below",
) -> tuple[Figure, np.ndarray]:
    """Draw a rows-x-cols grid of scatter plots.

    If `fig` is None, a new Figure is created. If `fig` is passed and already
    contains axes, their shape must match `(len(row.values), len(col.values))`
    — otherwise a `ValueError` is raised. If `fig` has no axes yet, the grid
    is created on it via `fig.subplots(...)`.

    Args:
        x_col:            Column name for the x-axis values.
        y_col:            Column name for the y-axis values.
        regression:  "none" — no regression line; "per_hue" — one regression line
                     per hue value; "across_hue" — single regression across all
                     points in the cell regardless of hue.

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
    markers = hue.markerstyles if hue.markerstyles is not None else ["o"] * len(hue.values)
    fixed = fixed or {}

    for i, row_val in enumerate(_row_values):
        for j, col_val in enumerate(_col_values):
            ax: plt.Axes = axes[i, j]  # type: ignore[assignment]
            all_x: list[np.ndarray] = []
            all_y: list[np.ndarray] = []
            for colour, marker, hue_val in zip(colours, markers, hue.values):
                flt = {**fixed, hue.key: hue_val}
                if row is not None:
                    flt[row.key] = row_val
                if col is not None:
                    flt[col.key] = col_val
                subset = filter_df_by_dict(data, flt)
                subset = subset[subset[x_col].notna() & subset[y_col].notna()]

                if subset.empty:
                    continue

                x = subset[x_col].to_numpy()
                y = subset[y_col].to_numpy()

                ax.scatter(
                    x,
                    y,
                    label=str(hue_val),
                    color=colour,
                    s=marker_size,
                    alpha=alpha,
                    marker=marker,
                    edgecolors="none",
                )

                if regression == "per_hue":
                    slope, intercept = _fit_line(x, y, regression_loss)
                    x_line = np.array([x.min(), x.max()])
                    ax.plot(
                        x_line,
                        slope * x_line + intercept,  # type: ignore[operator]
                        color=colour,
                        linewidth=marker_size / 4,
                        label=f"{regression_loss.capitalize()} regression",
                    )
                elif regression == "across_hue":
                    all_x.append(x)
                    all_y.append(y)

            if regression == "across_hue" and all_x:
                xa = np.concatenate(all_x)
                ya = np.concatenate(all_y)
                slope, intercept = _fit_line(xa, ya, regression_loss)
                x_line = np.array([xa.min(), xa.max()])
                ax.plot(
                    x_line,
                    slope * x_line + intercept,
                    color="black",
                    linewidth=marker_size / 4,
                    label=f"{regression_loss.capitalize()} regression",
                )

            if log_scale_y:
                ax.set_yscale("log")
            if log_scale_x:
                ax.set_xscale("log")
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
