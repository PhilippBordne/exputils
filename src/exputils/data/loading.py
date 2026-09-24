import logging
import os
from pathlib import Path

import pandas as pd
import yaml

from exputils.run.info import RunInfo

logger = logging.getLogger(__name__)

StrPath = str | os.PathLike[str]


def filter_df_by_dict(df: pd.DataFrame, filter: dict | list[dict]) -> pd.DataFrame:
    if isinstance(filter, dict):
        filter = [filter]

    df.reset_index(drop=True, inplace=True)
    mask = pd.Series([True] * len(df))

    for filter_dict in filter:
        for key, val in filter_dict.items():
            if isinstance(val, list):
                mask &= df[key].isin(val)
            else:
                mask &= df[key] == val
    filtered: pd.DataFrame = df[mask]
    filtered.reset_index(drop=True, inplace=True)

    return filtered


def _custom_df_summary(df: pd.DataFrame) -> None:
    """
    Summarizes a dataframe by printing the number of unique values.
    """
    data = []

    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        summary_df = data.append({"column": col, "nunique": nunique})
    summary_df = pd.DataFrame(data=data)
    print(summary_df)


def _expand_col_spec(spec: str | list | dict, prefix: str = "") -> list[str]:
    if isinstance(spec, str):
        return [f"{prefix}{spec}"]
    if isinstance(spec, list):
        return [col for item in spec for col in _expand_col_spec(item, prefix)]
    if isinstance(spec, dict):
        return [col for k, v in spec.items() for col in _expand_col_spec(v, f"{prefix}{k}.")]
    raise TypeError(f"Unsupported type in col spec: {type(spec)}")


def _load_configs_under_root(roots: StrPath | list[StrPath], keep_undone: bool = False) -> pd.DataFrame:
    if isinstance(roots, (str, os.PathLike)):
        roots = [roots]

    rows = []
    for root in roots:
        for config_path in Path(root).glob("**/config.yaml"):
            # skip hidden, _hydra, and wandb directories
            if any(p.startswith(".") or p in ("_hydra", "wandb") for p in config_path.parts):
                continue

            run_dir = str(config_path.parent)
            with open(config_path, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

            run_info = RunInfo.from_yaml(run_dir + "/run_info.yaml")
            if run_info.status != "done" and not keep_undone:
                continue

            config["path"] = run_dir
            config["done"] = run_info.status == "done"
            config["time_start"] = run_info.time_start
            rows.append(config)

    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows).dropna(axis=1, how="all")


def load_experiment_configs(
    roots: StrPath | list[StrPath],
    filter: dict | list[dict] | None = None,
    keep_undone: bool = False,
    keep_cols: str | list | dict | None = None,
    ignore_for_unique: list[str] | None = None,
    keep_path: bool = True,
) -> pd.DataFrame:
    """
    Load experiment configurations into a dataframe.

    Parameters
    ----------
    roots : str | list[str]
        Directory(ies) under which to search for the results.

    filter : Optional[Dict | list[Dict]], optional
        Filter(s) to apply. Supports nested dicts that get auto-flattened, e.g.
        ``{"learn": {"lr": 0.001}, "task": {"env_id": "Ant-v5"}}`` becomes
        ``{"learn.lr": 0.001, "task.env_id": "Ant-v5"}``.
        Values can be lists to match multiple values for a key.

    keep_cols : Optional[str | list | dict], optional
        Columns to keep. Supports nested specs, e.g.
        ``{"learn": ["lr", "batch_size"], "task": "env_id"}`` expands to
        ``["learn.lr", "learn.batch_size", "task.env_id"]``.
        If None, keep all columns.

    ignore_for_unique : Optional[list[str]], optional
        Columns in keep_cols to exclude from the uniqueness check. Defaults to ["run_id"].
    """
    ignore_for_unique = ignore_for_unique if ignore_for_unique is not None else []
    ignore_for_unique += ["run_id", "path", "time_start", "done"]

    ############# load the configurations into a dataframe #############
    experiment_df = _load_configs_under_root(roots=roots, keep_undone=keep_undone)

    ########### FILTERING #########
    if filter is not None:
        if not isinstance(filter, list):
            filter = [filter]
        filter = [pd.json_normalize(d, sep=".").to_dict(orient="records")[0] for d in filter]
        experiment_df = filter_df_by_dict(experiment_df, filter)

    if keep_cols is not None:
        keep_cols = _expand_col_spec(keep_cols)
        keep_cols += ["run_id"]

        if keep_path:
            keep_cols += ["path"]
        if keep_undone:
            keep_cols += ["done"]

        experiment_df = experiment_df.drop(columns=[c for c in experiment_df.columns if c not in keep_cols])

    ######### UNIQUENESS CHECK #########
    if keep_cols is not None and len(experiment_df) > 1:
        uniqueness_cols = [c for c in keep_cols if c not in ignore_for_unique and c in experiment_df.columns]
        if uniqueness_cols and experiment_df.duplicated(subset=uniqueness_cols).any():
            logger.warning("Kept columns do not uniquely identify runs.")

    logger.info(f"Loaded {len(experiment_df)} configs.")
    print("======== LOADED CONFIG SUMMARY ========")
    _custom_df_summary(experiment_df)

    return experiment_df


def load_results_dataframe(
    root: StrPath | list[StrPath],
    filter: dict | list[dict] | None = None,
    keep_undone: bool = False,
    eval_freq: int | None = None,
    last_n: int | None = None,
    metrics: str | list[str] | None = None,
    keep_cols: str | list | dict | None = None,
    ignore_for_unique: list[str] | None = None,
    time_column: str = "step",
) -> pd.DataFrame:
    """
    Loads the results dataframe for a given task and methods. Joining the configs with their respective results.

    Parameters
    ----------
    root : str | list[str]
        Directory(ies) under which to search for the results.

    filter : Optional[Dict | list[Dict]], optional
        Filter(s) to apply. Supports nested dicts that get auto-flattened, e.g.
        ``{"learn": {"lr": 0.001}, "task": {"env_id": "Ant-v5"}}`` becomes
        ``{"learn.lr": 0.001, "task.env_id": "Ant-v5"}``.

    keep_cols : Optional[str | list | dict], optional
        Columns to keep. Supports nested specs (see load_experiment_configs).
        If None, keep all config columns.

    eval_freq : Optional[int], optional
        If provided, only keeps every eval_freq step from the metrics. by default None.

    last_n : Optional[int], optional
        If provided, only keeps the last n steps from the metrics (after eval_freq filtering) by default None.
        It will remove the time column and "start" columns after aggregation.

    time_column : str, optional
        Name of the column used as the time axis (e.g. "step", "seconds", "minutes").
        Defaults to "step".

    metrics : Optional[str | list[str]], optional
        If provided, only keeps the specified metrics from the metrics.csv files otherwise keeps all metrics.

    ignore_for_unique : Optional[list[str]], optional
        Columns in keep_cols to exclude from the uniqueness check. Defaults to ["run_id"].
    """
    keep_cols = _expand_col_spec(keep_cols) if keep_cols is not None else None
    config_df = load_experiment_configs(
        roots=root,
        filter=filter,
        keep_undone=keep_undone,
        keep_cols=keep_cols,
        ignore_for_unique=ignore_for_unique,
        keep_path=True,
    )
    results = []

    for _, config_row in config_df.iterrows():
        metrics_path = os.path.join(config_row["path"], "metrics.csv")
        if os.path.exists(metrics_path):
            logger.debug(f"Loading metrics from {metrics_path}")
            try:
                metrics_df = pd.read_csv(metrics_path)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as e:
                logger.warning(f"Failed to load metrics from {metrics_path}: {e}")
                continue

            if eval_freq is not None:
                metrics_df = metrics_df[metrics_df[time_column] % eval_freq == 0].reset_index(drop=True)

            if last_n is not None:
                metrics_df = metrics_df.tail(last_n).reset_index(drop=True)
                metrics_df = metrics_df.agg("mean", axis="index").to_frame().T
                metrics_df.drop(columns=[time_column, "start"], inplace=True, errors="ignore")

            if metrics is not None:
                if not isinstance(metrics, list):
                    metrics = [metrics]
                metrics_df.drop(
                    columns=[col for col in metrics_df.columns if col not in metrics and col not in [time_column, "start"]],
                    inplace=True,
                    errors="ignore",
                )

            # prefix metric columns with "m." to distinguish from config columns
            if not metrics:
                metrics_df = metrics_df.rename(columns={c: f"m.{c}" for c in metrics_df.columns if c not in [time_column, "start"]})

            # repeat config for each metrics row
            config_data = config_row.to_dict()
            for _, metrics_row in metrics_df.iterrows():
                combined = {**config_data, **metrics_row.to_dict()}
                results.append(combined)
        else:
            logger.warning(f"metrics.csv not found for {config_row['path']}")

    df = pd.DataFrame(results)
    if keep_cols and "path" not in keep_cols:
        df.drop(columns=["path"], inplace=True, errors="ignore")

    # drop not needed columns
    if len(df) > 0 and "start" in df.columns and df["start"].nunique() == 1:
        df.drop(columns=["start"], inplace=True)

    return df
