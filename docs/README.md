# Prediction-track decision documentation

This folder explains **what** the multiclass digital-wellbeing pipeline does and **why** each design choice was locked. It is for teammates writing the Data Science 1 final report. It is **not** the report itself.

## Audience

- Teammates drafting report sections (objective, EDA, preprocessing, models, evaluation, citations)
- Anyone reproducing the Colab notebook who needs the rationale behind locked constants

## How to reproduce results

1. Open [`notebooks/digital_wellbeing_classifier.ipynb`](../notebooks/digital_wellbeing_classifier.ipynb) in Google Colab.
2. Run the git-clone cell first (repo root: `/content/teen-mental-health-predictive-model`).
3. Runtime → **Run all**.
4. Download `outputs/run_summary.zip`. Confirm `phase_completed: 5` in `manifest.json`, and that `tables/metrics_comparison.csv` plus the confusion-matrix figure are present.

Authoritative metric tables live in `outputs/metrics/report.json` after a full run. Numbers quoted in these docs come from a local Phase 5 smoke run with `SEED=42`; re-run the notebook if you need a fresh artifact set.

## Document map

| File | Contents |
| ---- | -------- |
| [prediction-track/00_scope_and_locked_decisions.md](prediction-track/00_scope_and_locked_decisions.md) | Locked settings and short rationale for each |
| [prediction-track/01_phase0_problem_framing.md](prediction-track/01_phase0_problem_framing.md) | Problem framing, target, feature policy |
| [prediction-track/02_phase1_scaffold_and_reproducibility.md](prediction-track/02_phase1_scaffold_and_reproducibility.md) | Repo layout, config, Colab paths |
| [prediction-track/03_phase2_eda_rationale.md](prediction-track/03_phase2_eda_rationale.md) | Four EDA figures and decisions they drove |
| [prediction-track/04_phase3_preprocessing_rationale.md](prediction-track/04_phase3_preprocessing_rationale.md) | Split, encoding, scaling, engineered feature |
| [prediction-track/05_phase4_modeling_rationale.md](prediction-track/05_phase4_modeling_rationale.md) | Logistic + random forest, CV, hyperparameters |
| [prediction-track/06_phase5_evaluation_rationale.md](prediction-track/06_phase5_evaluation_rationale.md) | Baselines, metrics, limitations |
| [prediction-track/rubric_crosswalk.md](prediction-track/rubric_crosswalk.md) | DS1 checklist → where to find evidence |

## Hypothesis testing (out of scope here)

The DS1 project can combine hypothesis testing with predictive modeling. **This repository implements the prediction track only.** Hypothesis testing (statement of hypothesis, choice of test, p-values / intervals, conclusion) is a separate teammate responsibility and is not documented in these files.

## Framing rule

Treat all results as an educational research prototype on a public Kaggle table. Do not claim clinical deployment, diagnosis, or treatment guidance.
