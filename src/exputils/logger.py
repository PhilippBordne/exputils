import logging
import sys

import wandb
from stable_baselines3.common.logger import (
    CSVOutputFormat,
    HumanOutputFormat,
    KVWriter,
    Logger,
)


def setup_logger(logger: logging.Logger) -> logging.Logger:
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class WandbWriter(KVWriter):
    """SB3 output format that forwards scalar metrics to Weights & Biases."""

    def write(self, key_values, key_excluded, step=0):
        wandb.log(
            {k: v for k, v in key_values.items() if isinstance(v, (int, float))},
            step=step,
        )

    def close(self):
        wandb.finish()


def make_sb3_logger(
    log_dir: str = ".",
    csv_filename: str = "metrics.csv",
    use_wandb: bool = False,
    use_stdout: bool = True,
) -> Logger:
    """Create an SB3 Logger that writes to CSV, optionally W&B and STDOUT.

    Parameters
    ----------
    log_dir : str
        Directory for the CSV file.
    csv_filename : str
        Name of the CSV file inside *log_dir*.
    use_wandb : bool
        Whether to include a WandbWriter (requires ``wandb.init()`` first).
    use_stdout : bool
        Whether to include human-readable STDOUT output.
    """
    import os

    os.makedirs(log_dir, exist_ok=True)

    formats = [CSVOutputFormat(os.path.join(log_dir, csv_filename))]
    if use_wandb:
        formats.append(WandbWriter())
    if use_stdout:
        formats.append(HumanOutputFormat(sys.stdout))

    return Logger(folder=log_dir, output_formats=formats)
