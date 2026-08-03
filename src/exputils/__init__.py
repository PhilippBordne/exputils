from exputils.config.template import FullConfig

from .data.loading import filter_df_by_dict, load_experiment_configs, load_results_dataframe

__all__ = [
    "FullConfig",
    "filter_df_by_dict",
    "load_experiment_configs",
    "load_results_dataframe",
]

import logging

_log = logging.getLogger(__name__)  # "mypkg", since __name__ == package name here
if not _log.hasHandlers():
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s]",
            datefmt="%H:%M:%S",
        )
    )
    _log.addHandler(_handler)
_log.setLevel(logging.INFO)
_log.propagate = False  # don't also hit root, avoids double prints
