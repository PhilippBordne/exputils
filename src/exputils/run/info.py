import logging
import os
import secrets
import string
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import git
import yaml

logger = logging.getLogger(__name__)


def get_commit_hash(path: str | Path) -> str | None:
    """Return the HEAD commit of the git repo containing ``path``, suffixed with ``-dirty`` if
    tracked files have uncommitted changes, or None if ``path`` is not inside a git repo.

    Pass a path inside the code's repo (e.g. ``Path(__file__).parent``), not the cwd: runs may
    execute in a results directory outside the repo.
    """
    try:
        repo = git.Repo(path, search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        logger.warning(f"{path} is not inside a git repository, commit hash is not recorded")
        return None
    commit_hash = repo.head.object.hexsha
    if repo.is_dirty(untracked_files=False):
        commit_hash += "-dirty"
    return commit_hash


def get_slurm_id() -> str | None:
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", None)
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID", None)
    if array_job_id and array_task_id:
        slurm_id = f"{array_job_id}_{array_task_id}"
    else:
        slurm_id = os.getenv("SLURM_JOB_ID", None)
    return slurm_id


def get_timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_job_running(job_id):
    """
    Check if a SLURM job is still running.

    Args:
        job_id (str or int): SLURM job ID

    Returns:
        bool: True if job is running, False otherwise
    """
    try:
        # Run squeue command to check specific job
        result = subprocess.run(["squeue", "-j", str(job_id)], capture_output=True, text=True, timeout=10)

        # If job exists, squeue returns 0 and job appears in output
        return bool(result.returncode == 0 and str(job_id) in result.stdout)

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False


ExperimentStatus = Literal["running", "done", "preempted"]


@dataclass
class RunInfo:
    """Dataclass to store metadata about the experiment run."""

    commit_hash: str | None = field(default=None)
    run_id: str = field(default="")
    status: ExperimentStatus = field(default="running")
    slurm_id: str | None = field(default_factory=get_slurm_id)
    time_start: str = field(default_factory=get_timestamp_now)
    time_end: str | None = field(default=None)

    def finalize_run(self):
        """Mark the run as done and record the end timestamp."""
        self.status = "done"
        self.time_end = get_timestamp_now()

    def to_yaml(self, path: str) -> None:
        """Save the RunInfo to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def from_yaml(cls, path: str) -> "RunInfo":
        """Load a RunInfo instance from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


_RUN_ID_ALPHABET = string.ascii_letters + string.digits  # a-z, A-Z, 0-9 (62 chars)
_RUN_ID_LENGTH = 6  # 62^6 ≈ 56.8 billion unique values


def create_run_id(length: int = _RUN_ID_LENGTH) -> str:
    """Create a short, unique, case-sensitive run ID."""
    return "".join(secrets.choice(_RUN_ID_ALPHABET) for _ in range(length))
