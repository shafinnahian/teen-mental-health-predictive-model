"""Lifestyle feature groups and the single engineered predictor for Phase 3+."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Canonical Colab clone target (see notebook git-clone cell).
COLAB_REPO_PATH = Path("/content/teen-mental-health-predictive-model")
PHASE3_SRC_FILES = (
    "config.py",
    "features.py",
    "preprocessing.py",
    "data_loader.py",
    "paths.py",
    "io_utils.py",
    "run_summary.py",
    "models.py",
)


def colab_src_diagnostics() -> dict[str, object]:
    """Collect paths and checks to debug Colab src/ import problems."""
    module_file = Path(__file__).resolve()
    src_dir = module_file.parent
    repo_root = src_dir.parent
    src_on_path = [entry for entry in sys.path if Path(entry).resolve() == src_dir]

    colab_repo = COLAB_REPO_PATH
    expected_src = colab_repo / "src"
    missing_in_colab_clone = [
        name
        for name in PHASE3_SRC_FILES
        if not (expected_src / name).is_file()
    ]

    return {
        "module_file": str(module_file),
        "src_dir": str(src_dir),
        "repo_root": str(repo_root),
        "cwd": str(Path.cwd()),
        "likely_colab": Path("/content").exists(),
        "colab_repo_exists": colab_repo.is_dir(),
        "colab_src_exists": expected_src.is_dir(),
        "colab_features_exists": (expected_src / "features.py").is_file(),
        "colab_preprocessing_exists": (expected_src / "preprocessing.py").is_file(),
        "missing_in_colab_clone": missing_in_colab_clone,
        "src_on_sys_path": src_on_path,
        "src_on_sys_path_count": len(src_on_path),
    }


def format_colab_src_error(
    module_name: str,
    *,
    missing_dependency: str | None = None,
    extra: str | None = None,
) -> str:
    """Build an actionable error message for notebook / Colab setup failures."""
    diag = colab_src_diagnostics()
    lines = [
        f"Cannot use src/{module_name}.py from the Colab notebook.",
        "",
        "What this usually means:",
        "  1) The git-clone cell was not run (repo missing under /content/).",
        "  2) The setup cells that add src/ to sys.path were skipped.",
        "  3) Colab cloned an old GitHub commit without Phase 3 files (git pull / push).",
        "  4) You opened the .ipynb as plain text instead of as a notebook.",
        "",
        "Fix in Colab:",
        "  - Runtime -> Restart session",
        "  - Run ALL cells from the top (clone cell first, then setup imports).",
        "  - Confirm files exist:",
        f"      {COLAB_REPO_PATH / 'src' / 'features.py'}",
        f"      {COLAB_REPO_PATH / 'src' / 'preprocessing.py'}",
        "",
        "Diagnostics:",
        f"  this module loaded from: {diag['module_file']}",
        f"  cwd:                    {diag['cwd']}",
        f"  Colab repo exists:      {diag['colab_repo_exists']}",
        f"  Colab src/ exists:      {diag['colab_src_exists']}",
        f"  features.py on clone:   {diag['colab_features_exists']}",
        f"  preprocessing.py clone: {diag['colab_preprocessing_exists']}",
        f"  src/ entries on sys.path: {diag['src_on_sys_path_count']}",
    ]

    if diag["src_on_sys_path"]:
        lines.append(f"  sys.path src entry:     {diag['src_on_sys_path'][0]}")
    else:
        lines.append(
            f"  expected sys.path entry: {COLAB_REPO_PATH / 'src'}"
        )

    missing = diag["missing_in_colab_clone"]
    if missing:
        lines.extend(
            [
                "",
                "Missing under Colab clone src/:",
                *(f"  - {name}" for name in missing),
                "",
                "If files exist locally but not here, push to GitHub and re-run the clone cell.",
            ]
        )

    if missing_dependency:
        lines.extend(
            [
                "",
                f"Failed dependency import: {missing_dependency}",
            ]
        )

    if extra:
        lines.extend(["", extra])

    return "\n".join(lines)


def verify_colab_src_setup() -> dict[str, object]:
    """Run before Phase 3 imports in the notebook; raises if setup looks wrong."""
    diag = colab_src_diagnostics()
    problems: list[str] = []

    if diag["likely_colab"] and not diag["colab_repo_exists"]:
        problems.append(
            f"Colab repo folder missing: {COLAB_REPO_PATH}. Run the git-clone cell."
        )

    if diag["likely_colab"] and diag["missing_in_colab_clone"]:
        missing = ", ".join(diag["missing_in_colab_clone"])
        problems.append(
            f"Phase 3 src files missing from Colab clone: {missing}. "
            "Run git pull in the clone cell or push latest code to GitHub."
        )

    if diag["src_on_sys_path_count"] == 0:
        problems.append(
            "src/ is not on sys.path. Run the setup cell that executes "
            "sys.path.insert(0, str(PROJECT_ROOT / 'src'))."
        )

    module_file = Path(str(diag["module_file"]))
    if diag["likely_colab"] and diag["colab_repo_exists"]:
        colab_features = COLAB_REPO_PATH / "src" / "features.py"
        if colab_features.is_file() and module_file != colab_features.resolve():
            problems.append(
                "Imported features.py from a different path than the Colab clone. "
                f"loaded: {module_file} expected: {colab_features.resolve()}"
            )

    if problems:
        raise RuntimeError(
            format_colab_src_error(
                "features",
                extra="Setup problems detected:\n" + "\n".join(f"  - {p}" for p in problems),
            )
        )

    return diag


try:
    from config import LIFESTYLE_FEATURES
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        format_colab_src_error("features", missing_dependency="config")
    ) from exc

# Fixed threshold (hours). Not learned from labels.
SCREEN_BEFORE_BED_THRESHOLD = 2.0
ENGINEERED_FEATURE = "high_screen_before_bed"

LIFESTYLE_NUMERIC = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
]

CATEGORICAL_FEATURES = [
    "gender",
    "platform_usage",
    "social_interaction_level",
]

BINARY_PASSTHROUGH = [ENGINEERED_FEATURE]


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with high_screen_before_bed from screen_time_before_sleep.

    Formula: high_screen_before_bed = 1 if screen_time_before_sleep > 2.0 else 0
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"add_engineered_features expected a pandas DataFrame, got {type(frame)!r}."
        )

    out = frame.copy()
    if "screen_time_before_sleep" not in out.columns:
        raise KeyError(
            format_colab_src_error(
                "features",
                extra=(
                    "Column 'screen_time_before_sleep' is missing from the input frame.\n"
                    f"  columns received: {list(out.columns)}\n"
                    "  Run setup + data_loader cells first so df has all lifestyle columns."
                ),
            )
        )
    out[ENGINEERED_FEATURE] = (
        out["screen_time_before_sleep"] > SCREEN_BEFORE_BED_THRESHOLD
    ).astype(int)
    return out


def lifestyle_feature_columns() -> list[str]:
    """Locked lifestyle predictors used to build X (before engineering)."""
    return list(LIFESTYLE_FEATURES)
