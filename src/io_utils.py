"""Thin serialization helpers for later pipeline phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_json(path: Path | str, obj: Any) -> Path:
    """Write a JSON-serializable object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_json(path: Path | str) -> Any:
    """Load a JSON object from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_joblib(path: Path | str, obj: Any) -> Path:
    """Serialize an object with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path: Path | str) -> Any:
    """Deserialize a joblib object from disk."""
    return joblib.load(Path(path))
