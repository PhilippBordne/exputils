"""Standardized 2.5D (pcolormesh) grid plotting.

Sibling of `time_series_grid.py`, with the same row/col layout. Each cell
draws a 2D heatmap where:

    * x-axis = time dimension (e.g. `step`)
    * y-axis = the hue dimension's values (e.g. replay_ratio)
    * color  = the metric value (mean over seeds)

Colormap sharing is configurable via `cmap_share`:

    * "none"  -> one colorbar per cell, placed by the user (none drawn
                 automatically)
    * "group" -> single shared colorbar for the entire grid, placed
                 according to `cbar_placement` ("right" or "below")
    * "row"   -> one shared colorbar per row, placed to the right of the row
    * "col"   -> one shared colorbar per column, placed below the column
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from exputils.data.loading import filter_df_by_dict
from core.plotting.grid_spec import ColumnSpec, GridSpec
from core.plotting.utils import create_global_x_label

CmapShare = Literal["none", "group", "row", "col"]


def plot_pcolor_grid(
    data: pd.DataFrame,
    *,
    row: GridSpec | None = None,
    col: ColumnSpec | None = None,
    x_dim: str,
    y_dim: str,
    fixed: dict | None = None,
    fig: Figure | None = None,
    figsize: tuple[float, float] = (5.5, 3.5),
    sharex: bool = True,
    sharey: bool | Literal["none", "all", "row", "col"] = "row",
    mean_col: str = "mean",
    cmap_name: str = "viridis",
    cmap_share: CmapShare = "group",
    cbar_placement: Literal["right", "below"] = "right",
    cbar_label: str | None = None,
    cbar_shrink: float = 0.9,
    cbar_pad: float = 0.02,
    row_ylabel: bool = True,
    col_title: bool = True,
    constrained_layout: bool = True,
    x_label: str | None = "steps",
    x_label_y: float = -0.04,
    y_label: str | None = None,
    y_label_x: float = -0.02,
    center_cmap: bool = False,
    shading: Literal["auto", "nearest", "gouraud", "flat"] = "auto",
) -> tuple[Figure, np.ndarray]:
    """Draw a rows-x-cols grid of 2.5D pcolormesh plots.

    The y-axis of every cell is the `hue` dimension; ticks are placed at
    integer indices and labelled with `hue.label(i)`.

    Colormap sharing is controlled by `cmap_share`:
        * "none":  nothing is drawn automatically, user decides.
        * "group": one shared colorbar for the whole grid.
        * "row":   one shared colorbar per row, placed to the right.
        * "col":   one shared colorbar per column, placed below.

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

    fixed = fixed or {}

    # 1st pass: build Z arrays per cell.
    cell_data: list[list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]] = [
        [(None, None, None)] * n_cols for _ in range(n_rows)
    ]
    for i, row_val in enumerate(_row_values):
        for j, col_val in enumerate(_col_values):
            flt = {**fixed}
            if row is not None:
                flt[row.key] = row_val
            if col is not None:
                flt[col.key] = col_val
            sub_data = filter_df_by_dict(data, flt)
            grid_data = sub_data.pivot(index=y_dim, columns=x_dim, values=mean_col)
            if grid_data.empty:
                continue
            x_vals = grid_data.columns.to_numpy()
            y_vals = grid_data.index.to_numpy()
            cell_data[i][j] = (x_vals, y_vals, grid_data.to_numpy())

    # Determine vmin/vmax scope.
    def _vrange(cells: list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]) -> tuple[float, float]:
        vals = [c[-1] for c in cells if c[-1] is not None]
        if not vals:
            return 0.0, 1.0
        stacked = np.concatenate([v[np.isfinite(v)].ravel() for v in vals if np.isfinite(v).any()])
        if stacked.size == 0:
            return 0.0, 1.0
        return float(np.nanmin(stacked)), float(np.nanmax(stacked))

    norms: dict[tuple[int, int], Normalize] = {}
    if cmap_share == "group":
        all_cells = [cell_data[i][j] for i in range(n_rows) for j in range(n_cols)]
        vmin, vmax = _vrange(all_cells)
        shared = Normalize(vmin=vmin, vmax=vmax)
        for i in range(n_rows):
            for j in range(n_cols):
                norms[(i, j)] = shared
    elif cmap_share == "row":
        for i in range(n_rows):
            vmin, vmax = _vrange([cell_data[i][j] for j in range(n_cols)])
            shared = Normalize(vmin=vmin, vmax=vmax)
            for j in range(n_cols):
                norms[(i, j)] = shared
    elif cmap_share == "col":
        for j in range(n_cols):
            vmin, vmax = _vrange([cell_data[i][j] for i in range(n_rows)])
            shared = Normalize(vmin=vmin, vmax=vmax)
            for i in range(n_rows):
                norms[(i, j)] = shared
    else:  # "none"
        for i in range(n_rows):
            for j in range(n_cols):
                x_vals, y_vals, Z = cell_data[i][j]
                if Z is None:
                    norms[(i, j)] = Normalize(vmin=0.0, vmax=1.0)
                    continue
                finite = Z[np.isfinite(Z)]
                vmin = float(finite.min()) if finite.size else 0.0
                vmax = float(finite.max()) if finite.size else 1.0
                norms[(i, j)] = Normalize(vmin=vmin, vmax=vmax)

    if center_cmap:
        for norm in norms.values():
            max_abs = max(abs(norm.vmin), abs(norm.vmax))
            norm.vmin = -max_abs
            norm.vmax = max_abs

    # 2nd pass: draw.
    meshes: dict[tuple[int, int], Any] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            ax: plt.Axes = axes[i, j]
            x_vals, y_vals, Z = cell_data[i][j]

            if Z is None or x_vals is None or y_vals is None:
                continue
            X_edges = np.arange(len(x_vals) + 1) - 0.5
            Y_edges = np.arange(len(y_vals) + 1) - 0.5
            mesh = ax.pcolormesh(
                X_edges,
                Y_edges,
                Z,
                cmap=cmap_name,
                norm=norms[(i, j)],
                shading="flat",
            )
            meshes[(i, j)] = mesh
            ax.set_yticks(np.arange(len(y_vals)))
            ax.set_yticklabels([y_vals[k] for k in range(len(y_vals))])
            ax.set_xticks(np.arange(len(x_vals)))
            ax.set_xticklabels([x_vals[k] for k in range(len(x_vals))])
            ax.set_aspect("equal")

    if row_ylabel and row is not None:
        for i in range(n_rows):
            axes[i, 0].set_ylabel(row.label(i))
    if col_title and col is not None:
        for j in range(n_cols):
            axes[0, j].set_title(col.title(j))

    if x_label is not None:
        create_global_x_label(fig, x_label, y=x_label_y)
    if y_label is not None:
        fig.supylabel(y_label, x=y_label_x)

    # Colorbars.
    sm = ScalarMappable(cmap=plt.get_cmap(cmap_name))

    def _mappable(norm: Normalize) -> ScalarMappable:
        m = ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
        m.set_array([])
        return m

    if cmap_share == "group":
        fig.colorbar(
            _mappable(norms[(0, 0)]),
            ax=axes.ravel().tolist(),
            location=cbar_placement,
            shrink=cbar_shrink,
            pad=cbar_pad,
            label=cbar_label or "",
        )
    elif cmap_share == "row":
        for i in range(n_rows):
            fig.colorbar(
                _mappable(norms[(i, 0)]),
                ax=list(axes[i, :]),
                location="right",
                shrink=cbar_shrink,
                pad=cbar_pad,
                label=cbar_label or "",
            )
    elif cmap_share == "col":
        for j in range(n_cols):
            fig.colorbar(
                _mappable(norms[(0, j)]),
                ax=list(axes[:, j]),
                location="bottom",
                shrink=cbar_shrink,
                pad=cbar_pad,
                label=cbar_label or "",
            )
    # "none": user draws their own colorbars from the returned axes/meshes.

    del sm
    return fig, axes
