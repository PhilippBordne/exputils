"""Interval bar plot grid with configurable orientation.

Each cell shows one bar per `hue` value:
  * a coloured bar spanning the uncertainty interval [low, high]
  * a short dash (perpendicular to the value axis) marking the point estimate

With ``orientation="horizontal"`` (default) the `hue` values are arranged
along the y-axis and the metric is on the x-axis.  With
``orientation="vertical"`` the axes are swapped.  Row / column structure
follows the same conventions as ``grid_lineplot.py``.

Typical usage::

    plot_barplot_grid(
        data=summary_df,
        row=GridSpec("task.env_id", ["Ant-v5", "HalfCheetah-v5"]),
        col=GridSpec("task.action_noise", [0, 0.05, 0.1, 0.2], title_fmt="noise={}"),
        hue=HueSpec("learn.r_space.type", ["none", "ln"], labels=["None", "LN"]),
        point_col="mean",
        std_col="std",
    )
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from exputils.data.loading import filter_df_by_dict
from core.plotting.grid_spec import GridSpec, HueSpec
from core.plotting.utils import create_global_legend


def plot_barplot_grid(
    data: pd.DataFrame,
    *,
    row: GridSpec | None = None,
    col: GridSpec | None = None,
    hue: HueSpec,
    fixed: dict | None = None,
    fig: Figure | None = None,
    figsize: tuple[float, float] = (5.5, 3.5),
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    share_value_axis: bool | Literal["none", "all", "row", "col"] = "row",
    point_col: str = "mean",
    low_col: str | None = None,
    high_col: str | None = None,
    std_col: str | None = "std",
    bar_height: float = 0.5,
    dash_width_fraction: float = 1.0,
    cmap_name: str = "viridis",
    val_lims: tuple[float | None, float | None] | None = None,
    col_title: bool = True,
    row_ylabel: bool = True,
    x_label: str | None = None,
    x_label_y: float = -0.04,
    y_label: str | None = None,
    y_label_x: float = -0.02,
    global_legend: bool = False,
    legend_title: str | None = None,
    legend_ncol: int | None = None,
    legend_placement: Literal["below", "right"] = "below",
    constrained_layout: bool = True,
    rotate_ticks: int = 0,
) -> tuple[Figure, np.ndarray]:
    """Draw a rows × cols grid of interval bar plots.

    Each cell contains one bar per hue value.  The bar spans the uncertainty
    interval and a short dash (perpendicular to the value axis) marks the
    point estimate.

    ``orientation="horizontal"`` (default) places hue values on the y-axis
    and the metric on the x-axis.  ``orientation="vertical"`` swaps them.

    The ``sharex`` parameter controls sharing of the **value axis** regardless
    of orientation: for horizontal bars it is passed as ``sharex`` to
    :func:`plt.subplots`; for vertical bars it is passed as ``sharey``.

    ``x_lims`` applies to the **value axis** in both orientations.

    Interval bounds are resolved in priority order:
      1. ``low_col`` / ``high_col`` (asymmetric intervals, both required)
      2. ``point_col ± std_col`` (symmetric interval)
      3. Point-only: only the dash is drawn, no bar.

    Returns:
        (fig, axes) where axes has shape (n_rows, n_cols).
    """
    _row_values = row.values if row is not None else [None]
    _col_values = col.values if col is not None else [None]
    n_rows = len(_row_values)
    n_cols = len(_col_values)
    n_hue = len(hue.values)

    is_horiz = orientation == "horizontal"
    _sharex = share_value_axis if is_horiz else True
    _sharey = True if is_horiz else share_value_axis

    if fig is None:
        fig, axes_arr = plt.subplots(
            n_rows,
            n_cols,
            sharex=_sharex,
            sharey=_sharey,
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
                sharex=_sharex,
                sharey=_sharey,
                squeeze=False,
            )
            axes = np.asarray(axes_arr).reshape(n_rows, n_cols)
        else:
            if len(existing) != n_rows * n_cols:
                raise ValueError(f"Provided figure has {len(existing)} axes but grid requires {n_rows * n_cols} ({n_rows}×{n_cols}).")
            axes = np.asarray(existing).reshape(n_rows, n_cols)

    colours = hue.get_colors(cmap_name=cmap_name)
    fixed = fixed or {}

    dash_half = bar_height * dash_width_fraction / 2

    for i, row_val in enumerate(_row_values):
        for j, col_val in enumerate(_col_values):
            ax: plt.Axes = axes[i, j]

            for idx, (colour, hue_val) in enumerate(zip(colours, hue.values)):
                flt = {**fixed, hue.key: hue_val}
                if row is not None:
                    flt[row.key] = row_val
                if col is not None:
                    flt[col.key] = col_val

                row_data = filter_df_by_dict(data, flt)
                if row_data.empty or point_col not in row_data.columns:
                    continue

                point = row_data[point_col].iloc[0]
                label = hue.label(idx)

                # Resolve interval bounds
                if low_col is not None and high_col is not None:
                    low = row_data[low_col].iloc[0]
                    high = row_data[high_col].iloc[0]
                elif std_col is not None and std_col in row_data.columns:
                    std = row_data[std_col].iloc[0]
                    low, high = point - std, point + std
                else:
                    low, high = None, None

                has_interval = low is not None and high is not None
                dash_kwargs: dict = {"linewidths": 0.5, "linestyles": "-", "color": "black"}
                if not has_interval:
                    dash_kwargs["label"] = label

                if is_horiz:
                    pos = n_hue - 1 - idx
                    if has_interval:
                        ax.barh(pos, width=high - low, left=low, height=bar_height, color=colour, alpha=0.4, label=label)  # type: ignore[arg-type]
                    ax.vlines(point, pos - dash_half, pos + dash_half, **dash_kwargs)
                else:
                    pos = idx
                    if has_interval:
                        ax.bar(pos, height=high - low, bottom=low, width=bar_height, color=colour, alpha=0.4, label=label)  # type: ignore[arg-type]
                    ax.hlines(point, pos - dash_half, pos + dash_half, **dash_kwargs)

            # hue_tick_labels = [hue.label(n_hue - 1 - k) for k in range(n_hue)]
            hue_tick_labels = [hue.label(k) for k in range(n_hue)]
            rotation = rotate_ticks
            if is_horiz:
                hue_tick_labels.reverse()

            if is_horiz:
                ax.set_yticks(list(range(n_hue)))
                ax.set_yticklabels(hue_tick_labels, rotation=rotation)
                ax.set_ylim(-0.5, n_hue - 0.5)
                if val_lims is not None:
                    ax.set_xlim(*val_lims)
            else:
                ax.set_xticks(list(range(n_hue)))
                ax.set_xticklabels(hue_tick_labels, rotation=rotation)
                ax.set_xlim(-0.5, n_hue - 0.5)
                if val_lims is not None:
                    ax.set_ylim(*val_lims)

    if col_title and col is not None:
        for j in range(n_cols):
            axes[0, j].set_title(col.label(j))

    if row_ylabel and row is not None:
        for i in range(n_rows):
            axes[i, 0].set_ylabel(row.label(i))

    if x_label is not None:
        fig.supxlabel(x_label, y=x_label_y)

    if y_label is not None:
        fig.supylabel(y_label, x=y_label_x)

    for ax in axes.flatten():
        drop = "x" if orientation == "vertical" else "y"
        ax.grid(False, axis=drop)

    if global_legend:
        create_global_legend(
            fig,
            title=legend_title,
            ncol=legend_ncol,
            placement=legend_placement,
        )

    return fig, axes
