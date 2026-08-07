"""Leakage-safe sklearn preprocessing pipeline for lifestyle predictors."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "scikit-learn is not installed. In Colab, run the setup cell that executes "
        "%pip install -r requirements.txt before the Phase 3 preprocessing cells."
    ) from exc

COLAB_REPO_PATH = Path("/content/teen-mental-health-predictive-model")


def _preprocessing_import_error(exc: ModuleNotFoundError) -> str:
    """Explain missing features/config imports when preprocessing cannot load."""
    src_dir = Path(__file__).resolve().parent
    expected_src = COLAB_REPO_PATH / "src"
    features_path = expected_src / "features.py"
    preprocessing_path = expected_src / "preprocessing.py"
    src_on_path = [entry for entry in sys.path if Path(entry).resolve() == src_dir]

    lines = [
        "Cannot import src/preprocessing.py dependencies for Phase 3.",
        "",
        f"Python looked for module: {exc.name!r}",
        "",
        "Colab checklist:",
        "  1) Run the git-clone cell (creates /content/teen-mental-health-predictive-model).",
        "  2) Run setup cells that install requirements and add src/ to sys.path.",
        "  3) Run Phase 3 cells only after setup finished without errors.",
        "",
        "File checks on Colab clone:",
        f"  repo exists:            {COLAB_REPO_PATH.is_dir()}",
        f"  src/features.py exists: {features_path.is_file()}",
        f"  src/preprocessing.py:   {preprocessing_path.is_file()}",
        f"  this file loaded from:  {Path(__file__).resolve()}",
        f"  cwd:                    {Path.cwd()}",
        f"  src/ on sys.path:       {len(src_on_path) > 0}",
    ]

    if not features_path.is_file():
        lines.extend(
            [
                "",
                f"Missing: {features_path}",
                "Push src/features.py to GitHub, then re-run the clone/pull cell.",
            ]
        )

    if not src_on_path:
        lines.extend(
            [
                "",
                "src/ is not on sys.path. Re-run:",
                "  SRC_PATH = PROJECT_ROOT / 'src'",
                "  sys.path.insert(0, str(SRC_PATH))",
            ]
        )

    return "\n".join(lines)


try:
    from config import LIFESTYLE_FEATURES, SEED, TARGET, TEST_SIZE
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(_preprocessing_import_error(exc)) from exc

try:
    from features import (
        BINARY_PASSTHROUGH,
        CATEGORICAL_FEATURES,
        LIFESTYLE_NUMERIC,
        add_engineered_features,
        format_colab_src_error,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(_preprocessing_import_error(exc)) from exc

PREPROCESS_PIPELINE_FILENAME = "preprocess_pipeline.joblib"
TRAIN_TEST_SPLIT_FILENAME = "train_test_split.joblib"

EXPECTED_PROCESSED_N_FEATURES = 14


def build_preprocessing_pipeline() -> Pipeline:
    """Build unfitted Pipeline: engineer feature, then ColumnTransformer.

    Numeric columns are scaled. Categoricals are one-hot encoded with
    drop='first'. The engineered binary column is passed through.
    """
    engineer = FunctionTransformer(
        add_engineered_features,
        validate=False,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), LIFESTYLE_NUMERIC),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="error",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_PASSTHROUGH),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        steps=[
            ("engineer", engineer),
            ("preprocess", preprocessor),
        ]
    )


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build lifestyle-only X and target y from the raw dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"make_xy expected a pandas DataFrame from the setup cells, got {type(df)!r}."
        )

    missing = [col for col in LIFESTYLE_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(
            format_colab_src_error(
                "preprocessing",
                extra=(
                    "Lifestyle columns missing from df.\n"
                    f"  missing: {missing}\n"
                    f"  df columns: {list(df.columns)}\n"
                    "  Run setup + load_raw_data cells before Phase 3."
                ),
            )
        )
    if TARGET not in df.columns:
        raise KeyError(
            format_colab_src_error(
                "preprocessing",
                extra=(
                    f"Target column {TARGET!r} missing from df.\n"
                    f"  df columns: {list(df.columns)}"
                ),
            )
        )
    X = df[LIFESTYLE_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split on the target."""
    if len(X) != len(y):
        raise ValueError(
            f"X and y length mismatch (X={len(X)}, y={len(y)}). "
            "Rebuild X, y with preprocessing.make_xy(df) after loading df."
        )
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    except ValueError as exc:
        raise ValueError(
            f"Stratified split failed: {exc}. "
            "Check that df loaded correctly and TARGET has all three classes."
        ) from exc


def fit_transform_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    pipeline: Pipeline | None = None,
) -> tuple[Pipeline, Any, Any, list[str]]:
    """Fit preprocessor on train only; transform train and test.

    Returns fitted pipeline, X_train_processed, X_test_processed, feature names.
    """
    if X_train.empty or X_test.empty:
        raise ValueError(
            "Train or test split is empty. Run stratified_split before preprocessing."
        )

    pipe = pipeline if pipeline is not None else build_preprocessing_pipeline()
    try:
        X_train_processed = pipe.fit_transform(X_train)
        X_test_processed = pipe.transform(X_test)
    except Exception as exc:
        raise RuntimeError(
            format_colab_src_error(
                "preprocessing",
                extra=(
                    "Pipeline fit/transform failed.\n"
                    f"  error: {exc}\n"
                    f"  X_train columns: {list(X_train.columns)}\n"
                    f"  expected lifestyle columns: {LIFESTYLE_FEATURES}\n"
                    "  Ensure Phase 3 runs after setup and df is loaded."
                ),
            )
        ) from exc

    feature_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    if len(feature_names) != EXPECTED_PROCESSED_N_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_PROCESSED_N_FEATURES} processed features, "
            f"got {len(feature_names)}: {feature_names}"
        )
    return pipe, X_train_processed, X_test_processed, feature_names


def build_split_artifact(
    *,
    train_idx: pd.Index,
    test_idx: pd.Index,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_processed: Any,
    X_test_processed: Any,
    feature_names_out: list[str],
) -> dict[str, Any]:
    """Package split + processed arrays for joblib persistence."""
    return {
        "train_idx": train_idx.to_numpy(),
        "test_idx": test_idx.to_numpy(),
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "feature_names_out": feature_names_out,
        "split_meta": {
            "test_size": TEST_SIZE,
            "random_state": SEED,
            "stratify": TARGET,
            "n_train": int(len(X_train_raw)),
            "n_test": int(len(X_test_raw)),
            "n_features_processed": int(len(feature_names_out)),
        },
    }
