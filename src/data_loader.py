"""Load and validate the Teen Mental Health CSV."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import (
    ALL_COLUMNS,
    EXPECTED_COLS,
    EXPECTED_ROWS,
    EXPECTED_TARGET_COUNTS,
    TARGET,
)
from paths import data_csv_path


def load_raw_data() -> pd.DataFrame:
    """Load the deployment CSV from data/."""
    path = data_csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Copy Teen_Mental_Health.csv into the data/ directory."
        )
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Validate schema strictly; soft-warn on target class count drift.

    Returns a summary dict for notebook display with keys:
    n_rows, n_cols, missing, target_counts, warnings.
    """
    warnings: list[str] = []

    if df.shape != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(
            f"Expected shape ({EXPECTED_ROWS}, {EXPECTED_COLS}), got {df.shape}."
        )

    if list(df.columns) != ALL_COLUMNS:
        raise ValueError(
            "Column names or order do not match ALL_COLUMNS in config.py.\n"
            f"Expected: {ALL_COLUMNS}\n"
            f"Got:      {list(df.columns)}"
        )

    missing = int(df.isnull().sum().sum())
    if missing != 0:
        raise ValueError(f"Expected 0 missing values, found {missing}.")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    target_counts = df[TARGET].value_counts().to_dict()
    for label, expected in EXPECTED_TARGET_COUNTS.items():
        actual = int(target_counts.get(label, 0))
        if actual != expected:
            warnings.append(
                f"Target count drift for '{label}': expected {expected}, got {actual}."
            )

    if warnings:
        for message in warnings:
            print(f"WARNING: {message}")

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing": missing,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "warnings": warnings,
    }
