"""Absolute path helpers for Colab and local runs.

Canonical root is derived from this file location (src/paths.py → project root),
so callers do not need to mutate the process working directory.
"""

from __future__ import annotations

from pathlib import Path

from config import DATA_FILENAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


def data_csv_path() -> Path:
    """Path to the Colab/deployment copy of the dataset."""
    return PROJECT_ROOT / "data" / DATA_FILENAME


def figures_dir() -> Path:
    return PROJECT_ROOT / "outputs" / "figures"


def models_dir() -> Path:
    return PROJECT_ROOT / "outputs" / "models"


def metrics_dir() -> Path:
    return PROJECT_ROOT / "outputs" / "metrics"


def run_summary_dir() -> Path:
    """Staging directory for the run summary export."""
    return PROJECT_ROOT / "outputs" / "run_summary"


def run_summary_zip_path() -> Path:
    """Downloadable run summary ZIP written after a notebook run."""
    return PROJECT_ROOT / "outputs" / "run_summary.zip"


def requirements_path() -> Path:
    return PROJECT_ROOT / "requirements.txt"


def ensure_output_dirs() -> None:
    """Create output subdirectories if they do not exist."""
    for directory in (figures_dir(), models_dir(), metrics_dir(), run_summary_dir()):
        directory.mkdir(parents=True, exist_ok=True)
