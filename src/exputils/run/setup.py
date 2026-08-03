import logging
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import wandb

from exputils.config.template import FullConfig

logger = logging.getLogger(__name__)


def setup_wandb(config: FullConfig, resume: bool = False):
    """Setup wandb logging based on configuration.
    If resume resume existing run with run_id from config.
    """

    # Setup wandb logging
    if config.log_to_wandb:
        wandb_dir = "/tmp/wandb"
        os.makedirs(wandb_dir, exist_ok=True)

        if resume:
            try:
                return wandb.init(
                    project=config.wandb_project,
                    id=config.run_id,
                    resume="must",
                    dir=wandb_dir,
                )
            finally:
                # If the run does not exist, just create a new one
                pass

        # if it is not a test run disable tracking of system stats
        if config.group is not None and config.group.startswith("test_"):
            wandb_settings = wandb.Settings(x_stats_sampling_interval=60.0)
        else:
            wandb_settings = wandb.Settings(x_disable_stats=True)

        run = wandb.init(
            project=config.wandb_project,
            config=asdict(config),
            group=config.group,
            id=config.run_id,
            dir=wandb_dir,
            name=config.get_run_name(),
            settings=wandb_settings,
        )

        return run
    return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
