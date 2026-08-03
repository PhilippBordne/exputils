"""
Base configuration template for experiments.

Domain-specific configs (e.g., SACConfig) inherit from FullConfig
and add their own fields (e.g., learn, task).
"""

from dataclasses import dataclass, field, fields


@dataclass
class FullConfig:
    """
    Main configuration managing general information such as run_id, experiment seed.
    Domain-specific configs inherit from this and add their own fields.
    """

    # always required
    seed: int = field(default=42)

    # required to use the wandb run setup
    log_to_wandb: bool = field(default=False)
    wandb_project: str | None = field(default=None)
    run_id: str | None = field(default=None)
    run_name: str | None = field(default=None)
    group: str | None = field(default=None)

    # useful to freely define where to create hydra run dirs
    path_results: str = field(default="results")

    # Fields that are excluded from identical_to comparison (infrastructure, not experiment definition)
    _identity_exclude = frozenset({"run_id", "run_name", "log_to_wandb", "path_results"})

    def identical_to(self, other: "FullConfig") -> bool:
        """Check whether this configuration is identical to another, ignoring run metadata."""
        exclude = self._identity_exclude
        for f in fields(self):
            if f.name in exclude:
                continue
            elif getattr(self, f.name) != getattr(other, f.name):
                return False
        return True

    def get_run_name(self) -> str:
        return self.run_name or "unnamed"
