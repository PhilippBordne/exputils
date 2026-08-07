# `exputils`

Reusable configuration and experiment state management for Hydra + SLURM projects.

## Why use this?
Configuration concept that ...
1. ... has typical challenges for *launching* ML experiments in mind (e.g. resuming after preemption, check for existing runs, launches on SLURM).
2. ... following the pattern implemented for 1. will make it straightforward to load experimental results and filter them.

## Quick Start

1. Copy this package into your project's `src/` directory.
2. Create your domain-specific config by inheriting from `FullConfig`.
3. Set up your experiment entry point with the OmegaConf resolvers (see below) and example Hydra config files under `./config/example_configs`.

## Key Concepts

### Directory-based resumption

With `hydra.job.chdir: true` (Hydra default), each run lands in a deterministic
directory derived from its config. On preemption + resubmit, the same CLI args
produce the same directory, and the experiment detects existing state and resumes.


### Identity comparison

`FullConfig.identical_to()` compares configs ignoring fields in `_identity_exclude`
(run_id, run_name, log_to_wandb, num_workers, path_results). Override
`_identity_exclude` in subclasses to exclude additional infrastructure fields.

### SLURM seed offset

When launching with `ntasks > 1`, each task gets `SLURM_PROCID` (0, 1, 2, ...).
The `slurm_seed` resolver adds this to the base seed, ensuring each task gets a
unique seed AND a unique directory (since `${seed}` is in the path).

## Entry Point with Resolvers

The code snippet below gives a template for an experiment entry point, that ...
1. implements resolvers to generate unique run names and determine seeds for use in the hydra `config.yaml`
2. detects if there is already a run with sanity checks for identical config and whether this one is already running on SLURM (think accidental re-submission)

```python
# experiments/train.py
import os
from typing import cast

import hydra
from omegaconf import OmegaConf

from exputils.config.loading import load_config_from_yaml
from exputils.run.info import RunInfo, get_slurm_id, is_job_running
from my_project.algorithms import run_experiment
from my_project.configuration import MyConfig

# --- Register OmegaConf resolvers ---

# Seed with SLURM task offset: ${slurm_seed:0} resolves to 0 + SLURM_PROCID.
# Use in cluster configs to give each SLURM task a unique seed.
# Locally (no SLURM_PROCID), it's a no-op.
# NOTE THIS ONLY WORKS IF THE RESOLVER RUNS PER TASK (i.e. if launched by batch script / does not work if done through hydra's submitit)
OmegaConf.register_new_resolver(
    "slurm_seed",
    lambda base: int(base) + int(os.environ.get("SLURM_PROCID", 0)),
    use_cache=True,
)

# Run name from config fields: ${gen_run_name:${field_a},${field_b}}
# Joins non-"none" values with "+". Returns "none" if all are "none".
def _gen_run_name(w_space: str, r_space: str, reset: str) -> str:
    parts = [str(x) for x in [w_space, r_space, reset]]  # cast to str avoids issues with Enum fields
    return "+".join(x for x in parts if x != "none") or "none"

OmegaConf.register_new_resolver("gen_run_name", _gen_run_name, use_cache=True)


@hydra.main(config_path="config", config_name="my_config", version_base=None)
def main(cfg: OmegaConf) -> None:
    config = cast(MyConfig, OmegaConf.to_object(cfg))

    # --- Resume detection ---
    # Hydra chdir's into the output dir. If config.yaml already exists,
    # a previous run landed here. Check if it matches and resume or error.
    resume = False
    if os.path.exists("config.yaml"):
        existing = load_config_from_yaml("config.yaml", MyConfig)
        if config.identical_to(existing):
            run_info = RunInfo.from_yaml("run_info.yaml")
            own_slurm_id = get_slurm_id()
            if run_info.status == "done":
                print("Run already finished. Exiting.")
                return
            elif own_slurm_id != run_info.slurm_id and is_job_running(run_info.slurm_id):
                print(f"Run {run_info.run_id} (SLURM: {run_info.slurm_id}) still running. Exiting.")
                return
            else:
                resume = True
                config = existing
        else:
            raise RuntimeError(
                "Directory contains a run with different config (path collision). "
                "Use a different group or add differing params to the path template."
            )

    # ... create experiment, run, etc.
    run_experiment(config, resume=resume)

    # ...


if __name__ == "__main__":
    main()
```


## Domain Config Example
Inherit from `FullConfig` to set all parameters that are expected by the util functions of this repository and the example config files.

```python
@dataclass
class FullConfig:
    group: str | None = field(default=None)
    log_to_wandb: bool = field(default=False)
    seed: int = field(default=42)
    run_id: str | None = field(default=None)
    run_name: str | None = field(default=None)
    path_results: str = field(default="results")
    num_workers: int = field(default=1)

    # Fields that are excluded from identical_to comparison (infrastructure, not experiment definition)
    _identity_exclude = frozenset({"run_id", "run_name", "log_to_wandb", "num_workers", "path_results"})
```

```python
# src/my_project/configuration.py
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from exputils.config.template import FullConfig

@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 256

@dataclass
class MyConfig(FullConfig):
    train: TrainConfig = field(default_factory=TrainConfig)

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)
```

## Tips and Tricks

### Use Literals when using dataclasses as OmegaConf config nodes (aka structured configs, aka schema)
OmegaConfs support for `Literal` types is not yet released.
To use `Literal` types in dataclasses for categorical hyperparameters, you can define the field as `Literal` only when `TYPE_CHECKING` and use compatible type (e.g. `str`) at runtime. Obviously at cost of losing type safety at runtime.

```python
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    OptimizerType = Literal["adam", "sgd"]
else:
    OptimizerType = str

@dataclass
class OptimizerConfig:
    type: OptimizerType = field(default="adam")
```
