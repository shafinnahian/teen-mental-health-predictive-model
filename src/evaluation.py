"""Phase 5 evaluation: baselines, train/test metrics, confusion matrix, report JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "scikit-learn/matplotlib are not installed. In Colab, run the setup cell "
        "that executes %pip install -r requirements.txt before Phase 5 cells."
    ) from exc

try:
    from config import SEED, TARGET_CLASSES
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import config from src/. Run the notebook setup cells that add "
        "src/ to sys.path before importing evaluation."
    ) from exc

try:
    from io_utils import save_json
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import io_utils from src/. Ensure src/ is on sys.path."
    ) from exc

REPORT_FILENAME = "report.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix_test.png"
PRIMARY_METRIC = "f1_macro"
OVERFIT_GAP_THRESHOLD = 0.05

# Fixed cutpoints for the social-media baseline (hours/day).
# Round midpoints near train-set class means; not tuned on the test set.
SOCIAL_MEDIA_FEATURE = "daily_social_media_hours"
SOCIAL_MEDIA_HEALTHY_MAX = 3.5  # <= 3.5 -> Healthy
SOCIAL_MEDIA_AT_RISK_MIN = 6.0  # > 6.0 -> At Risk; else Moderate

MODEL_KEYS = ("logistic", "random_forest")
BASELINE_KEYS = ("always_moderate", "social_media_rule")
ALL_RESULT_KEYS = BASELINE_KEYS + MODEL_KEYS

DEFAULT_LIMITATIONS = [
    "Synthetic Kaggle tabular data; not a clinical sample.",
    "Lifestyle-only predictors; stress/anxiety/addiction scales excluded.",
    "Moderate class dominates (~62%); accuracy alone is misleading.",
    "At Risk test support is small (n=30); per-class recall is noisy.",
    "Course project only — not for clinical use.",
]


def always_moderate_predictions(n: int, index: pd.Index | None = None) -> pd.Series:
    """Baseline 1: predict Moderate for every row."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    return pd.Series(["Moderate"] * n, index=index, name="prediction")


def social_media_threshold_predictions(X_raw: pd.DataFrame) -> pd.Series:
    """Baseline 2: rule on daily_social_media_hours only.

    Healthy if hours <= SOCIAL_MEDIA_HEALTHY_MAX,
    At Risk if hours > SOCIAL_MEDIA_AT_RISK_MIN,
    else Moderate.
    """
    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError(f"X_raw must be a DataFrame, got {type(X_raw)!r}.")
    if SOCIAL_MEDIA_FEATURE not in X_raw.columns:
        raise KeyError(
            f"Missing column {SOCIAL_MEDIA_FEATURE!r} required for social-media baseline."
        )

    hours = X_raw[SOCIAL_MEDIA_FEATURE]
    labels = np.where(
        hours <= SOCIAL_MEDIA_HEALTHY_MAX,
        "Healthy",
        np.where(hours > SOCIAL_MEDIA_AT_RISK_MIN, "At Risk", "Moderate"),
    )
    return pd.Series(labels, index=X_raw.index, name="prediction")


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, Any]:
    """Accuracy, macro/weighted F1, and per-class precision/recall/F1."""
    labels = list(TARGET_CLASSES)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        entry = report.get(label) or {}
        per_class[label] = {
            "precision": round(float(entry.get("precision", 0.0)), 6),
            "recall": round(float(entry.get("recall", 0.0)), 6),
            "f1": round(float(entry.get("f1-score", 0.0)), 6),
            "support": int(entry.get("support", 0)),
        }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6
        ),
        "per_class": per_class,
    }


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Score a prediction vector against ground truth."""
    return compute_metrics(y_true, y_pred)


def evaluate_pipeline(pipe: Any, X_raw: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Predict with a fitted Pipeline on raw lifestyle rows, then score."""
    if X_raw.empty:
        raise ValueError("X_raw is empty.")
    if len(X_raw) != len(y):
        raise ValueError(f"Length mismatch: X={len(X_raw)}, y={len(y)}.")
    y_pred = pipe.predict(X_raw)
    return compute_metrics(y, y_pred)


def evaluate_baselines(
    X_raw: pd.DataFrame,
    y: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Score both baselines on one split."""
    always = always_moderate_predictions(len(y), index=y.index)
    social = social_media_threshold_predictions(X_raw)
    return {
        "always_moderate": compute_metrics(y, always),
        "social_media_rule": compute_metrics(y, social),
    }


def select_best_model(model_results: dict[str, dict[str, Any]]) -> str:
    """Pick logistic or random_forest by highest test macro-F1."""
    best_name: str | None = None
    best_score = -1.0
    for name in MODEL_KEYS:
        if name not in model_results:
            continue
        test_block = model_results[name].get("test") or {}
        score = float(test_block.get(PRIMARY_METRIC, -1.0))
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is None:
        raise ValueError(
            "No model results with test scores. Expected keys: "
            f"{MODEL_KEYS}."
        )
    return best_name


def detect_overfitting_flags(
    results: dict[str, dict[str, Any]],
    *,
    gap_threshold: float = OVERFIT_GAP_THRESHOLD,
) -> list[str]:
    """Flag models whose train macro-F1 exceeds test by more than the gap."""
    flags: list[str] = []
    for name in MODEL_KEYS:
        block = results.get(name)
        if not block:
            continue
        train_f1 = float((block.get("train") or {}).get(PRIMARY_METRIC, 0.0))
        test_f1 = float((block.get("test") or {}).get(PRIMARY_METRIC, 0.0))
        gap = train_f1 - test_f1
        if gap > gap_threshold:
            flags.append(
                f"{name}: train {PRIMARY_METRIC}={train_f1:.3f} vs "
                f"test={test_f1:.3f} (gap={gap:.3f} > {gap_threshold})."
            )
    return flags


def plot_confusion_matrix_test(
    y_test: pd.Series,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    out_path: Path | str,
    *,
    dpi: int = 150,
) -> Path:
    """Save a counts confusion matrix for the best model on the test set."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(TARGET_CLASSES)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix — {model_name} (n={len(y_test)})")
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Flat table: model × split × accuracy / macro-F1 / weighted-F1."""
    rows: list[dict[str, Any]] = []
    for name in ALL_RESULT_KEYS:
        block = results.get(name)
        if not block:
            continue
        for split in ("train", "test"):
            metrics = block.get(split)
            if not metrics:
                continue
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_weighted": metrics["f1_weighted"],
                    "at_risk_recall": metrics["per_class"]["At Risk"]["recall"],
                }
            )
    return pd.DataFrame(rows)


def build_report_json(
    *,
    results: dict[str, dict[str, Any]],
    best_model: str,
    best_cm: list[list[int]],
    cv_meta: dict[str, Any] | None = None,
    overfitting_flags: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the Phase 5 metrics report schema."""
    return {
        "schema_version": 1,
        "seed": SEED,
        "primary_metric": PRIMARY_METRIC,
        "best_model": best_model,
        "baseline_rules": {
            "always_moderate": "predict Moderate for every row",
            "social_media_threshold": {
                "feature": SOCIAL_MEDIA_FEATURE,
                "healthy_max_hours": SOCIAL_MEDIA_HEALTHY_MAX,
                "at_risk_min_hours": SOCIAL_MEDIA_AT_RISK_MIN,
            },
        },
        "cv_meta": cv_meta or {},
        "results": results,
        "best_model_test": {
            "name": best_model,
            "confusion_matrix": best_cm,
            "class_labels": list(TARGET_CLASSES),
        },
        "overfitting_flags": list(overfitting_flags or []),
        "limitations": list(limitations if limitations is not None else DEFAULT_LIMITATIONS),
    }


def save_report(path: Path | str, report: dict[str, Any]) -> Path:
    """Write report.json via io_utils.save_json."""
    return save_json(path, report)


def run_full_evaluation(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logistic_pipe: Any,
    rf_pipe: Any,
    cv_meta: dict[str, Any] | None = None,
    figures_dir: Path | str,
    metrics_dir: Path | str,
) -> tuple[dict[str, Any], pd.DataFrame, Path, Path]:
    """Score baselines + models, pick best, save CM figure and report.json.

    Returns (report, comparison_df, report_path, cm_path).
    """
    figures_dir = Path(figures_dir)
    metrics_dir = Path(metrics_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_baselines = evaluate_baselines(X_train, y_train)
    test_baselines = evaluate_baselines(X_test, y_test)
    results: dict[str, dict[str, Any]] = {
        "always_moderate": {
            "train": train_baselines["always_moderate"],
            "test": test_baselines["always_moderate"],
        },
        "social_media_rule": {
            "train": train_baselines["social_media_rule"],
            "test": test_baselines["social_media_rule"],
        },
        "logistic": {
            "train": evaluate_pipeline(logistic_pipe, X_train, y_train),
            "test": evaluate_pipeline(logistic_pipe, X_test, y_test),
        },
        "random_forest": {
            "train": evaluate_pipeline(rf_pipe, X_train, y_train),
            "test": evaluate_pipeline(rf_pipe, X_test, y_test),
        },
    }

    best_model = select_best_model(results)
    best_pipe = logistic_pipe if best_model == "logistic" else rf_pipe
    y_pred_test = best_pipe.predict(X_test)
    labels = list(TARGET_CLASSES)
    cm = confusion_matrix(y_test, y_pred_test, labels=labels).tolist()

    cm_path = plot_confusion_matrix_test(
        y_test,
        y_pred_test,
        best_model,
        figures_dir / CONFUSION_MATRIX_FILENAME,
    )

    flags = detect_overfitting_flags(results)
    at_risk_recall = float(
        results[best_model]["test"]["per_class"]["At Risk"]["recall"]
    )
    if at_risk_recall < 0.5:
        flags.append(
            f"Best model ({best_model}) At Risk test recall={at_risk_recall:.3f} < 0.5 "
            "(n=30 At Risk test rows)."
        )

    report = build_report_json(
        results=results,
        best_model=best_model,
        best_cm=cm,
        cv_meta=cv_meta,
        overfitting_flags=flags,
    )
    report_path = save_report(metrics_dir / REPORT_FILENAME, report)
    table = comparison_table(results)
    return report, table, report_path, cm_path
