"""Build a downloadable run summary ZIP (Markdown + JSON + small CSVs).

Collects facts from the notebook run into short lab-note prose and optional notes.
Phase detection is file-based so later notebook sections can enrich the same zip.
"""

from __future__ import annotations

import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    EXPECTED_COLS,
    EXPECTED_ROWS,
    EXPECTED_TARGET_COUNTS,
    SEED,
    TARGET,
    TARGET_CLASSES,
    TEST_SIZE,
)
from features import ENGINEERED_FEATURE, SCREEN_BEFORE_BED_THRESHOLD, add_engineered_features
from io_utils import save_json
from paths import (
    figures_dir,
    get_project_root,
    run_summary_dir,
    run_summary_zip_path,
    metrics_dir,
    models_dir,
)
from preprocessing import (
    EXPECTED_PROCESSED_N_FEATURES,
    PREPROCESS_PIPELINE_FILENAME,
    TRAIN_TEST_SPLIT_FILENAME,
)

SCHEMA_VERSION = 1

EXPECTED_FIGURES = (
    "01_class_balance.png",
    "02_social_media_by_wellbeing.png",
    "03_lifestyle_correlation.png",
    "04_platform_crosstab.png",
)

EXPECTED_TRAIN_COUNTS = {"Moderate": 594, "Healthy": 245, "At Risk": 121}
EXPECTED_TEST_COUNTS = {"Moderate": 149, "Healthy": 61, "At Risk": 30}

LOGISTIC_PIPELINE_FILENAME = "logistic_pipeline.joblib"
RF_PIPELINE_FILENAME = "rf_pipeline.joblib"
METRICS_REPORT_FILENAME = "report.json"


def _git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"pandas": pd.__version__}
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except ImportError:
        versions["sklearn"] = "missing"
    return versions


def _ordered_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts()
    return {label: int(counts.get(label, 0)) for label in TARGET_CLASSES}


def _rel_if_under_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def collect_run_meta(*, phase_completed: int) -> dict[str, Any]:
    """UTC timestamp, detected phase, git commit, library versions, seed."""
    root = get_project_root()
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase_completed": phase_completed,
        "git_commit": _git_commit(root),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "repo_root": str(root),
        "likely_colab": Path("/content").exists(),
        "versions": _package_versions(),
    }


def collect_data_facts(df: pd.DataFrame) -> dict[str, Any]:
    """Shape, missing/duplicates, target counts, risk-score leakage check."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"collect_data_facts expected a DataFrame, got {type(df)!r}.")

    target_counts = _ordered_counts(df[TARGET]) if TARGET in df.columns else {}
    risk_ok: bool | None = None
    risk_mismatch_n: int | None = None
    needed = ("mental_health_risk_score", "stress_level", "anxiety_level", "addiction_level")
    if all(col in df.columns for col in needed):
        expected = df["stress_level"] + df["anxiety_level"] + df["addiction_level"]
        mismatch = df["mental_health_risk_score"] != expected
        risk_mismatch_n = int(mismatch.sum())
        risk_ok = risk_mismatch_n == 0

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": target_counts,
        "expected_target_counts": dict(EXPECTED_TARGET_COUNTS),
        "risk_score_equals_sum": risk_ok,
        "risk_score_mismatch_n": risk_mismatch_n,
        "matches_expected_shape": (
            int(df.shape[0]) == EXPECTED_ROWS and int(df.shape[1]) == EXPECTED_COLS
        ),
    }


def collect_eda_facts(df: pd.DataFrame) -> dict[str, Any]:
    """Social-media means by class, engineered prevalence, figure presence."""
    social_means: dict[str, float] = {}
    if TARGET in df.columns and "daily_social_media_hours" in df.columns:
        grouped = df.groupby(TARGET, observed=True)["daily_social_media_hours"].mean()
        for label in TARGET_CLASSES:
            if label in grouped.index:
                social_means[label] = round(float(grouped[label]), 2)

    eng_rate: float | None = None
    eng_n: int | None = None
    if "screen_time_before_sleep" in df.columns:
        eng = add_engineered_features(df)
        eng_n = int(eng[ENGINEERED_FEATURE].sum())
        eng_rate = round(float(eng[ENGINEERED_FEATURE].mean()), 4)

    fig_dir = figures_dir()
    figures_present = [
        name for name in EXPECTED_FIGURES if (fig_dir / name).is_file() and (fig_dir / name).stat().st_size > 0
    ]
    figures_missing = [name for name in EXPECTED_FIGURES if name not in figures_present]

    return {
        "social_media_means_by_class": social_means,
        "engineered_feature": ENGINEERED_FEATURE,
        "engineered_threshold": SCREEN_BEFORE_BED_THRESHOLD,
        "engineered_positive_n": eng_n,
        "engineered_positive_rate": eng_rate,
        "figures_present": figures_present,
        "figures_missing": figures_missing,
    }


def collect_preprocessing_facts(
    *,
    y_train: pd.Series | None,
    y_test: pd.Series | None,
    feature_names: list[str] | None,
) -> dict[str, Any] | None:
    """Split sizes, class counts, processed feature names. None if split not in memory."""
    if y_train is None or y_test is None or feature_names is None:
        return None

    train_counts = _ordered_counts(y_train)
    test_counts = _ordered_counts(y_test)
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_counts": train_counts,
        "test_counts": test_counts,
        "n_features_processed": int(len(feature_names)),
        "feature_names": list(feature_names),
        "encoding": "one-hot drop=first for gender, platform_usage, social_interaction_level",
        "scaling": "StandardScaler on 6 numeric lifestyle columns; binary passthrough",
        "engineered_formula": (
            f"{ENGINEERED_FEATURE} = 1 if screen_time_before_sleep > "
            f"{SCREEN_BEFORE_BED_THRESHOLD} else 0"
        ),
    }


def collect_model_facts() -> dict[str, Any] | None:
    """Stub for Phase 4: report which model joblibs exist."""
    model_dir = models_dir()
    present = {
        "logistic_pipeline": (model_dir / LOGISTIC_PIPELINE_FILENAME).is_file(),
        "rf_pipeline": (model_dir / RF_PIPELINE_FILENAME).is_file(),
    }
    if not any(present.values()):
        return None
    return {
        "artifacts_present": present,
        "note": "Phase 4 model summaries not yet included in this export.",
    }


def collect_evaluation_facts() -> dict[str, Any] | None:
    """Stub for Phase 5: report whether metrics/report.json exists."""
    report_path = metrics_dir() / METRICS_REPORT_FILENAME
    if not report_path.is_file():
        return None
    return {
        "report_path": _rel_if_under_root(report_path, get_project_root()),
        "note": "Phase 5 evaluation metrics not yet included in this export.",
    }


def _list_artifacts() -> list[str]:
    root = get_project_root()
    candidates = [
        models_dir() / PREPROCESS_PIPELINE_FILENAME,
        models_dir() / TRAIN_TEST_SPLIT_FILENAME,
        models_dir() / LOGISTIC_PIPELINE_FILENAME,
        models_dir() / RF_PIPELINE_FILENAME,
        metrics_dir() / METRICS_REPORT_FILENAME,
        *[figures_dir() / name for name in EXPECTED_FIGURES],
    ]
    return [
        _rel_if_under_root(path, root)
        for path in candidates
        if path.is_file() and path.stat().st_size > 0
    ]


def detect_phase_completed(
    *,
    preprocessing: dict[str, Any] | None,
    models: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    eda: dict[str, Any],
) -> int:
    """Infer highest completed phase from in-memory facts and on-disk artifacts."""
    if evaluation is not None:
        return 5
    if models is not None:
        return 4
    if preprocessing is not None:
        return 3
    if eda.get("figures_present"):
        return 2
    return 1


def detect_flags(facts: dict[str, Any]) -> list[str]:
    """Auto 'worth a look' items from collected facts."""
    flags: list[str] = []
    data = facts.get("data") or {}
    eda = facts.get("eda") or {}
    prep = facts.get("preprocessing")

    if data.get("n_rows") != EXPECTED_ROWS or data.get("n_cols") != EXPECTED_COLS:
        flags.append(
            f"Shape looks off: got {data.get('n_rows')}×{data.get('n_cols')}, "
            f"expected {EXPECTED_ROWS}×{EXPECTED_COLS}."
        )
    if data.get("missing_cells", 0) > 0:
        flags.append(f"Missing cells: {data['missing_cells']}.")
    if data.get("duplicate_rows", 0) > 0:
        flags.append(f"Duplicate rows: {data['duplicate_rows']}.")

    target_counts = data.get("target_counts") or {}
    expected = data.get("expected_target_counts") or {}
    if target_counts and expected and target_counts != {
        k: int(v) for k, v in expected.items()
    }:
        flags.append(
            f"Target counts differ from locked snapshot: got {target_counts}, "
            f"expected {expected}."
        )

    if data.get("risk_score_equals_sum") is False:
        flags.append(
            f"Risk score ≠ stress+anxiety+addiction on "
            f"{data.get('risk_score_mismatch_n')} rows."
        )

    missing_figs = eda.get("figures_missing") or []
    if missing_figs:
        flags.append(f"Missing EDA figures: {', '.join(missing_figs)}.")

    moderate = target_counts.get("Moderate", 0)
    n_rows = data.get("n_rows") or 0
    if n_rows and moderate / n_rows >= 0.5:
        flags.append(
            "Moderate dominates — macro-F1 will matter more than accuracy."
        )

    if prep is not None:
        if prep.get("n_features_processed") != EXPECTED_PROCESSED_N_FEATURES:
            flags.append(
                f"Processed feature width is {prep.get('n_features_processed')}, "
                f"expected {EXPECTED_PROCESSED_N_FEATURES}."
            )
        if prep.get("train_counts") != EXPECTED_TRAIN_COUNTS:
            flags.append(
                f"Train class counts differ from seed=42 baseline: "
                f"{prep.get('train_counts')}."
            )
        if prep.get("test_counts") != EXPECTED_TEST_COUNTS:
            flags.append(
                f"Test class counts differ from seed=42 baseline: "
                f"{prep.get('test_counts')}."
            )
        pipe_path = models_dir() / PREPROCESS_PIPELINE_FILENAME
        split_path = models_dir() / TRAIN_TEST_SPLIT_FILENAME
        if not pipe_path.is_file():
            flags.append(f"Missing artifact: {PREPROCESS_PIPELINE_FILENAME}.")
        if not split_path.is_file():
            flags.append(f"Missing artifact: {TRAIN_TEST_SPLIT_FILENAME}.")

    return flags


def _fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{label} {counts.get(label, 0)}" for label in TARGET_CLASSES)


def render_run_log_md(facts: dict[str, Any], notes: str = "") -> str:
    """Markdown lab notes from structured facts + optional user notes."""
    meta = facts["run_meta"]
    data = facts["data"]
    eda = facts["eda"]
    prep = facts.get("preprocessing")
    models = facts.get("models")
    evaluation = facts.get("evaluation")
    flags = facts.get("flags") or []
    phase = meta["phase_completed"]

    when = meta["timestamp_utc"][:10]
    where = "Colab" if meta.get("likely_colab") else "local"
    commit = meta.get("git_commit") or "unknown"

    lines: list[str] = [
        "# Run summary",
        "",
        f"Ran the notebook on {where}, {when}. Phase {phase} done. "
        f"Git commit `{commit}`, seed={meta['seed']}.",
        "",
        "## Data",
        "",
        f"Dataset is {data['n_rows']}×{data['n_cols']}, "
        f"{data['missing_cells']} missing cells, "
        f"{data['duplicate_rows']} duplicate rows.",
    ]

    counts = data.get("target_counts") or {}
    if counts:
        total = data["n_rows"] or 1
        lines.append(
            f"Target counts: {_fmt_counts(counts)}. "
            f"Moderate is {counts.get('Moderate', 0)}/{total}."
        )

    if data.get("risk_score_equals_sum") is True:
        lines.append(
            "Checked: mental_health_risk_score equals stress + anxiety + addiction "
            "on all rows (still excluded from predictors)."
        )
    elif data.get("risk_score_equals_sum") is False:
        lines.append(
            f"Risk-score check failed on {data.get('risk_score_mismatch_n')} rows."
        )

    lines.extend(["", "## EDA", ""])
    means = eda.get("social_media_means_by_class") or {}
    if means:
        healthy = means.get("Healthy")
        at_risk = means.get("At Risk")
        lines.append(
            f"daily_social_media_hours means — Healthy {healthy}, "
            f"Moderate {means.get('Moderate')}, At Risk {at_risk}. "
            "Hours separate classes more than platform choice in the EDA plots."
        )
    else:
        lines.append("Social-media means by class were not available.")

    if eda.get("engineered_positive_n") is not None:
        lines.append(
            f"Engineered `{eda['engineered_feature']}` "
            f"(threshold {eda['engineered_threshold']} h): "
            f"{eda['engineered_positive_n']} positives "
            f"({eda['engineered_positive_rate']:.1%} of rows)."
        )

    present = eda.get("figures_present") or []
    missing = eda.get("figures_missing") or []
    if present:
        lines.append("Figures saved: " + ", ".join(present) + ".")
    if missing:
        lines.append("Figures missing: " + ", ".join(missing) + ".")

    lines.extend(["", "## Preprocessing", ""])
    if prep is None:
        lines.append("Preprocessing facts not in memory for this export.")
    else:
        lines.append(
            f"Split: stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} "
            f"on `{TARGET}` → {prep['n_train']} train / {prep['n_test']} test."
        )
        lines.append(f"Train counts: {_fmt_counts(prep['train_counts'])}.")
        lines.append(f"Test counts: {_fmt_counts(prep['test_counts'])}.")
        lines.append(
            f"Processed width: {prep['n_features_processed']} columns. "
            f"{prep['encoding']}. {prep['scaling']}."
        )
        lines.append(f"Formula: `{prep['engineered_formula']}`.")

    if models is not None:
        lines.extend(["", "## Models", ""])
        present_models = models.get("artifacts_present") or {}
        lines.append(
            "Model artifacts on disk: "
            + ", ".join(f"{k}={v}" for k, v in present_models.items())
            + "."
        )
        if models.get("note"):
            lines.append(models["note"])

    if evaluation is not None:
        lines.extend(["", "## Evaluation", ""])
        lines.append(f"Found `{evaluation.get('report_path')}`.")
        if evaluation.get("note"):
            lines.append(evaluation["note"])

    lines.extend(["", "## Flags", ""])
    if flags:
        for item in flags:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")

    artifacts = facts.get("artifacts") or []
    lines.extend(["", "## Artifacts", ""])
    if artifacts:
        for path in artifacts:
            lines.append(f"- `{path}`")
    else:
        lines.append("- None found under outputs/.")

    notes_clean = (notes or "").strip()
    lines.extend(["", "## Notes", ""])
    if notes_clean:
        lines.append(notes_clean)
    else:
        lines.append("(none)")
    lines.append("")
    return "\n".join(lines)


def _write_tables(
    tables_dir: Path,
    *,
    data: dict[str, Any],
    eda: dict[str, Any],
    preprocessing: dict[str, Any] | None,
) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    target_path = tables_dir / "target_counts.csv"
    target_rows = [
        {"class": label, "count": data.get("target_counts", {}).get(label, 0)}
        for label in TARGET_CLASSES
    ]
    pd.DataFrame(target_rows).to_csv(target_path, index=False)
    written.append(target_path)

    means = eda.get("social_media_means_by_class") or {}
    social_path = tables_dir / "social_media_by_class.csv"
    pd.DataFrame(
        [
            {
                "class": label,
                "daily_social_media_hours_mean": means.get(label),
            }
            for label in TARGET_CLASSES
        ]
    ).to_csv(social_path, index=False)
    written.append(social_path)

    split_path = tables_dir / "split_counts.csv"
    if preprocessing is not None:
        pd.DataFrame(
            [
                {
                    "class": label,
                    "train": preprocessing["train_counts"].get(label, 0),
                    "test": preprocessing["test_counts"].get(label, 0),
                }
                for label in TARGET_CLASSES
            ]
        ).to_csv(split_path, index=False)
    else:
        pd.DataFrame(columns=["class", "train", "test"]).to_csv(split_path, index=False)
    written.append(split_path)

    feat_path = tables_dir / "processed_feature_names.csv"
    if preprocessing is not None:
        names = preprocessing.get("feature_names") or []
        pd.DataFrame(
            {"index": range(len(names)), "feature_name": names}
        ).to_csv(feat_path, index=False)
    else:
        pd.DataFrame(columns=["index", "feature_name"]).to_csv(feat_path, index=False)
    written.append(feat_path)

    return written


def write_run_summary(
    *,
    df: pd.DataFrame,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
    feature_names: list[str] | None = None,
    notes: str = "",
) -> Path:
    """Write staging files and zip them to outputs/run_summary.zip."""
    from paths import ensure_output_dirs

    ensure_output_dirs()

    data = collect_data_facts(df)
    eda = collect_eda_facts(df)
    preprocessing = collect_preprocessing_facts(
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
    models = collect_model_facts()
    evaluation = collect_evaluation_facts()
    phase = detect_phase_completed(
        preprocessing=preprocessing,
        models=models,
        evaluation=evaluation,
        eda=eda,
    )
    run_meta = collect_run_meta(phase_completed=phase)

    facts: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase_completed": phase,
        "run_meta": run_meta,
        "data": data,
        "target_counts": data.get("target_counts") or {},
        "eda": eda,
        "preprocessing": preprocessing,
        "models": models,
        "evaluation": evaluation,
        "artifacts": _list_artifacts(),
        "notes": (notes or "").strip(),
    }
    facts["flags"] = detect_flags(facts)

    staging = run_summary_dir()
    staging.mkdir(parents=True, exist_ok=True)
    tables_dir = staging / "tables"
    table_paths = _write_tables(
        tables_dir, data=data, eda=eda, preprocessing=preprocessing
    )

    md_path = staging / "run_log.md"
    md_path.write_text(render_run_log_md(facts, notes=facts["notes"]), encoding="utf-8")

    # Persist a JSON-friendly copy (drop None leaves as null).
    manifest_path = staging / "manifest.json"
    save_json(manifest_path, facts)

    zip_path = run_summary_zip_path()
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(md_path, arcname="run_log.md")
        zf.write(manifest_path, arcname="manifest.json")
        for table_path in table_paths:
            zf.write(table_path, arcname=f"tables/{table_path.name}")

    return zip_path
