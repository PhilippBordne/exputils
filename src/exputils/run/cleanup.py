import shutil
from pathlib import Path

from exputils.run.info import RunInfo, is_job_running


def delete_empty_dirs(root: str, ignore_files: list[str] | None = None) -> None:
    ignore_files = ignore_files or []
    for child in Path(root).iterdir():
        if child.is_dir():
            delete_empty_dirs(str(child), ignore_files)
    remaining = [f for f in Path(root).iterdir() if f.name not in ignore_files]
    if not remaining:
        shutil.rmtree(root)


def delete_unfinished_experiments(root: str, ignore_files: str | list[str] | None = None) -> None:
    if ignore_files is None:
        ignore_files = []
    elif isinstance(ignore_files, str):
        ignore_files = [ignore_files]

    for run_info_path in Path(root).glob("**/run_info.yaml"):
        run_dir = run_info_path.parent
        run_info = RunInfo.from_yaml(str(run_info_path))

        if run_info.status == "done":
            continue
        elif run_info.slurm_id is not None and is_job_running(run_info.slurm_id):
            print(f"Experiment in {run_dir} is still running (job_id: {run_info.slurm_id}). Skipping.")
        else:
            print(f"Experiment in {run_dir} is not marked as done (status: {run_info.status}). Deleting.")
            shutil.rmtree(run_dir)

    delete_empty_dirs(root, ignore_files)
