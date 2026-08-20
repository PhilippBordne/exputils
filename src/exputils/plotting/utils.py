import logging
from collections.abc import Callable
from typing import Literal, Optional

import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from pandas.core.groupby import DataFrameGroupBy


def create_global_legend(
    fig: Figure,
    placement: Literal["below", "right"] = "below",
    spacing: float = 0.02,
    ncol: int | None = None,
    title: str | None = None,
    columnspacing: float | None = None,
    handletextpad: float | None = None,
    handlelength: float | None = None,
) -> None:
    """
    Collects all legend handles and labels from all axes/subplots in the figure and creates a single legend.

    Args:
        fig: Matplotlib figure.
        placement: Where to place the legend ("below" for below the figure, "right" for to the right of the figure).
        spacing: Spacing beneath the figure (if below) or to the right (if right), as a fraction of the figure.
        ncol: Number of columns in the legend.
        fontsize: Font size for legend.
    """
    handles = []
    labels = []
    for ax in fig.get_axes():
        h, label_list = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(label_list)
        ax.legend_.remove() if ax.legend_ is not None else None

    # Remove duplicates while preserving order
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    if unique_handles and unique_labels:
        if placement == "right":
            loc = "center left"
            bbox = (1.0 + spacing, 0.5)
            ncol = ncol or 1
        else:  # "below"
            loc = "upper center"
            bbox = (0.5, -spacing)
            ncol = ncol or len(unique_labels)
        fig.legend(
            unique_handles,
            unique_labels,
            loc=loc,
            bbox_to_anchor=bbox,
            ncol=ncol,
            title=title,
            frameon=True,
            columnspacing=columnspacing,
            handletextpad=handletextpad,
            handlelength=handlelength,
        )


def make_default_label_fn(data: pd.DataFrame) -> Callable[[dict], str]:
    """
    Creates a default labeling function for plotting based on the unique learner configurations in the data.
    Args:
        data: DataFrame containing results to plot.
    Returns:
        A function that generates labels for each learner configuration.
    """
    learn_cols = [col for col in data.columns if col.startswith("learn.")]
    unique_learners = data[learn_cols]
    keys = [col for col in learn_cols if unique_learners[col].nunique() > 1]

    def label_fn(learner_dict: dict) -> str:
        labels = []
        for key in keys:
            labels.append(f"{key.split('learn.')[1]}={learner_dict[key]}")
        return ", ".join(labels)

    return label_fn


def group_by_learn_and_task(df: pd.DataFrame) -> DataFrameGroupBy:
    group_cols = []
    group_cols += [col for col in df.columns if col.startswith("learn.")]
    group_cols += [col for col in df.columns if col.startswith("task.")]
    if "step" in df.columns:
        group_cols.append("step")
    return df.groupby(group_cols, dropna=False)


def smoothen_time_series(df: pd.DataFrame, metric_cols: str | list[str], window_size: int = 5) -> pd.DataFrame:
    if "step" not in df.columns:
        logging.warning("No 'step' column found in DataFrame. Cannot smoothen time series without a time dimension.")
        return df

    if isinstance(metric_cols, str):
        metric_cols = [metric_cols]

    for metric_col in metric_cols:
        if df[metric_col].isna().any():
            logging.warning(f"NaN values found in metric column '{metric_col}'. These will be ignored in smoothing.")

    group_cols = [col for col in df.columns if col.startswith("learn.") or col.startswith("task.") or col == "seed"]
    grouped = df.groupby(group_cols, dropna=False)

    for _, group_df in grouped:
        group_df = group_df.sort_values("step")

        for metric_col in metric_cols:
            group_df[metric_col] = group_df[metric_col].rolling(window=window_size, min_periods=1).mean()
            df.loc[group_df.index, metric_col] = group_df[metric_col]
    return df


def mean_std_by_learn_and_task(
    df: pd.DataFrame, metric_col: str | list[str], drop_na: bool = False, smooth_window: int = 1
) -> pd.DataFrame:

    if drop_na:
        if isinstance(metric_col, str):
            drop = df[metric_col].isna()
        else:
            drop = pd.Series(False, index=df.index)
            for col in metric_col:
                drop = drop | df[col].isna()
        df = df[~drop]

    if smooth_window > 1:
        df = smoothen_time_series(df, metric_cols=metric_col, window_size=smooth_window)

    grouped = group_by_learn_and_task(df)
    if isinstance(metric_col, str):
        return grouped.agg(n_seeds=("seed", "count"), mean=(metric_col, "mean"), std=(metric_col, "std")).reset_index()
    else:
        agg_dict = {f"{col} (mean)": (col, "mean") for col in metric_col}
        agg_dict.update({f"{col} (std)": (col, "std") for col in metric_col})
        agg_dict["n_seeds"] = ("seed", "count")
        return grouped.agg(**agg_dict).reset_index()


def get_colors_for_values(
    values: list[float] | list[int], cmap_name: str = "viridis", log: bool = False
) -> list[tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    cmap = plt.get_cmap(cmap_name)
    if log:
        norm = LogNorm(min(values), max(values))
    else:
        norm = Normalize(min(values), max(values))
    return [cmap(norm(value)) for value in values]
