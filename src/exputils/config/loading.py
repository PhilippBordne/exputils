import os
from typing import TypeVar, cast

from omegaconf import OmegaConf

from exputils.config.template import FullConfig

ConfigType = TypeVar("ConfigType")


def load_config_from_yaml(path: str, schema: type[ConfigType]) -> ConfigType:
    schema = OmegaConf.structured(schema)
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(schema, cfg)
    cfg = OmegaConf.to_object(cfg)
    return cast(ConfigType, cfg)


# Currently not really used because we find the existing run direct by deterministic directory creation of hydra.
def find_existing_run(config: FullConfig, base_path: str) -> str | None:
    """Check whether an identical run already exists in the results directory."""
    for dir, _, files in os.walk(base_path):
        if "config.yaml" in files:
            existing_config = load_config_from_yaml(os.path.join(dir, "config.yaml"), type(config))
            if config.identical_to(existing_config):
                return dir
    return None
