from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..constants import (
    DATA_DIRNAME,
    REPORTS_DIRNAME,
    RUNS_DIRNAME,
    TRAJECTORIES_DIRNAME,
)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return get_repo_root() / DATA_DIRNAME


def logs_dir() -> Path:
    return data_dir() / "logs"


def reports_dir() -> Path:
    return get_repo_root() / REPORTS_DIRNAME


def runs_dir() -> Path:
    return get_repo_root() / RUNS_DIRNAME


def create_run_dir(command_name: str, label: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    run_dir = runs_dir() / f"{timestamp}_{command_name}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def trajectories_dir(run_dir: Path) -> Path:
    path = run_dir / TRAJECTORIES_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def tasks_file() -> Path:
    return data_dir() / "tasks.jsonl"


def db_path() -> Path:
    return data_dir() / "sample.db"


def app_log_path() -> Path:
    return logs_dir() / "app.log"


def resolve_data_path(relative_path: str) -> Path:
    base = data_dir().resolve()
    target = (base / relative_path).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("Path escape")
    return target
