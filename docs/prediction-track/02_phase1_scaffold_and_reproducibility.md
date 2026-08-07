# Phase 1 - Scaffold and reproducibility

Phase 1 built a Colab-runnable skeleton: config, paths, data load/validate, and output directories. No EDA plots or models yet. The goal was a green **Run all** path that always loads the same CSV schema.

## Repository layout

```text
teen-mental-health-predictive-model/
├── data/
│   └── Teen_Mental_Health.csv          # Deployment CSV for Colab + local
├── notebooks/
│   └── digital_wellbeing_classifier.ipynb
├── src/
│   ├── config.py                       # Locked constants
│   ├── paths.py                        # Project-root-relative paths
│   ├── data_loader.py                  # Load + schema validation
│   ├── io_utils.py                     # joblib / JSON helpers
│   ├── features.py                     # Feature groups + engineering
│   ├── preprocessing.py                # Split + ColumnTransformer pipeline
│   ├── models.py                       # Classifiers + GridSearchCV
│   ├── evaluation.py                   # Baselines + metrics + report.json
│   └── run_summary.py                  # Downloadable ZIP after a run
├── outputs/                            # Runtime artifacts (gitignored)
│   ├── figures/
│   ├── models/
│   ├── metrics/
│   └── run_summary/
├── docs/                               # This decision documentation
└── requirements.txt
```

## Why constants live in `src/config.py`

Notebooks drift. Hard-coding the target name, seed, or feature list in several cells invites silent mismatch. `config.py` holds:

- `SEED = 42`, `TEST_SIZE = 0.2`
- `TARGET = "digital_wellbeing_flag"` and `TARGET_CLASSES`
- `LIFESTYLE_FEATURES` and `EXCLUDED_COLUMNS`
- Expected shape `(1200, 16)` and expected target counts (soft-check)

Smoke cells assert `config.TARGET == "digital_wellbeing_flag"` so a wrong checkout fails early.

## Why `paths.py` derives the project root from `__file__`

Colab working directories change when users `cd` or open notebooks from Drive. `PROJECT_ROOT` is the parent of `src/`, not `Path.cwd()`. That makes `data/Teen_Mental_Health.csv` and `outputs/` resolve the same way on a laptop and under `/content/teen-mental-health-predictive-model`.

Canonical Colab clone root:

```text
/content/teen-mental-health-predictive-model
```

## Why the CSV is validated strictly

[`src/data_loader.py`](../../src/data_loader.py) refuses to proceed if:

- shape ≠ `(EXPECTED_ROWS, EXPECTED_COLS)`
- column names / order ≠ `ALL_COLUMNS`
- any missing values exist

Target class counts are soft-warned if they drift from the locked snapshot. Strict schema checks catch the wrong file (for example an older Kaggle download) before modeling burns time on garbage.

## Why dependencies are pinned by minimum versions

[`requirements.txt`](../../requirements.txt) lists pandas, numpy, scikit-learn, matplotlib, seaborn, and joblib. The notebook installs from that file on Colab so teammates share one environment story for citations and reproducibility notes.

## Artifacts directories

| Directory | Purpose |
| --------- | ------- |
| `outputs/figures/` | EDA PNGs + test confusion matrix |
| `outputs/models/` | Preprocess / split / model joblibs + Phase 4 meta JSON |
| `outputs/metrics/` | `report.json` |
| `outputs/run_summary/` + `.zip` | Downloadable lab-note package after Run all |

These paths are gitignored. Teammates should regenerate them with Colab **Run all**, then download the ZIP for report figures and tables.

## What Phase 1 deliberately skipped

- Feature engineering logic (Phase 3)
- Plots (Phase 2)
- Training (Phase 4)

Keeping scaffold separate made path and import failures easy to isolate on Colab before scientific work began.

## Rubric use

Phase 1 supports the **dataset introduction** and reproducibility narrative: origin (Kaggle URL in `config.DATASET_URL`), structure (1200×16, column list), and how the CSV is loaded in the course runtime.
