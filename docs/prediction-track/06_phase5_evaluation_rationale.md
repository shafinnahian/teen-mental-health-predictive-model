# Phase 5 - Evaluation rationale

Phase 5 scores baselines and the two saved pipelines on **train and test**, writes `outputs/metrics/report.json`, and saves a test confusion matrix for the best ML model. Implementation: [`src/evaluation.py`](../../src/evaluation.py), notebook Part 5.

Numbers below come from a local Phase 5 run with `SEED=42` (`outputs/metrics/report.json`). Re-run the notebook if your artifact differs.

## Evaluation questions

1. Do the ML models beat trivial baselines under **macro-F1**?
2. How large is the train vs test gap (overfitting check)?
3. How does the minority class (At Risk) fare on the holdout test set?
4. Which of logistic vs random forest wins on **test** macro-F1?

## Baselines

### Baseline 1 - Always Moderate

Predict `Moderate` for every row.

**Why:** majority-class lower bound. Shows that ~62% accuracy is achievable without learning. On the local run:

| Split | Accuracy | Macro-F1 | At Risk recall |
| ----- | -------- | -------- | -------------- |
| Train | 0.619 | 0.255 | 0.00 |
| Test | 0.621 | 0.255 | 0.00 |

### Baseline 2 - Social-media hours rule

Using only `daily_social_media_hours`:

```text
hours <= 3.5  → Healthy
hours >  6.0  → At Risk
otherwise     → Moderate
```

**Why:** EDA showed the strongest lifestyle separation on this column (Healthy mean ≈ 2.55 h, At Risk mean ≈ 7.09 h). Cutpoints are fixed round values near midpoints between class means, documented in code as train-motivated constants. They are **not** grid-searched on the test set.

Local run:

| Split | Accuracy | Macro-F1 | At Risk recall |
| ----- | -------- | -------- | -------------- |
| Train | 0.652 | 0.647 | 0.959 |
| Test | 0.650 | 0.645 | 0.967 |

The rule catches many At Risk rows (high recall) but pays in precision. Macro-F1 (~0.65) far beats always-Moderate, so a univariate lifestyle signal is already informative.

## Metrics reported

For every model and baseline, on train and test:

- accuracy
- macro-F1 (primary)
- weighted-F1
- per-class precision / recall / F1 / support (includes At Risk recall)

**Why train and test both:** a train≫test gap flags overfitting. Code raises an overfitting flag when train macro-F1 exceeds test by more than 0.05 (`OVERFIT_GAP_THRESHOLD`).

**Why At Risk recall:** educational framing cares about missing the minority "At Risk" class. With only 30 test At Risk rows, treat recall as noisy, not a clinical sensitivity estimate.

## Best-model rule

```text
best_model = argmax_{logistic, random_forest} test macro-F1
```

Baselines are reported for context but are not eligible. Primary metric key: `f1_macro`.

## Local ML results (SEED=42 holdout)

| Model | Split | Accuracy | Macro-F1 | At Risk recall |
| ----- | ----- | -------- | -------- | -------------- |
| Logistic | Train | 0.932 | 0.924 | 0.884 |
| Logistic | Test | 0.921 | 0.908 | 0.900 |
| Random forest | Train | 1.000 | 1.000 | 1.000 |
| Random forest | Test | 1.000 | 1.000 | 1.000 |

Best model by test macro-F1: **random_forest**. Overfitting flags list was empty under the 0.05 gap rule (forest train and test both perfect; logistic gap ≈ 0.016).

## Confusion matrix

Saved as `outputs/figures/confusion_matrix_test.png` for the best model. Local best-model test counts (labels Healthy, Moderate, At Risk in that order) are diagonal: 61 / 149 / 30 with zeros off-diagonal for random forest.

## Critical honesty (read carefully)

Perfect random-forest test scores on a 240-row holdout are a **red flag**, not a triumph.

Possible explanations (not mutually exclusive):

1. The Kaggle table may be synthetic or constructed so that lifestyle features nearly separate the label.
2. Unlimited tree depth (`max_depth=None`) can memorize structure that happens to repeat in the holdout when the generating process is simple.
3. With only 30 At Risk test rows, a few lucky splits still look "perfect" without proving external validity.

**How report authors should write this:**

- Report the forest numbers honestly (they are what the code produced).
- State that perfect holdout performance on this table is suspicious and limits external claims.
- Keep logistic regression (~0.91 test macro-F1, ~0.90 At Risk recall) as the more credible interpretable reference for generalization talk.
- Do not claim clinical readiness, population prevalence estimates, or screening utility.

## Limitations (also stored in `report.json`)

From `DEFAULT_LIMITATIONS` in `src/evaluation.py`:

1. Synthetic Kaggle tabular data; not a clinical sample.
2. Lifestyle-only predictors; stress / anxiety / addiction scales excluded.
3. Moderate class dominates (~62%); accuracy alone is misleading.
4. At Risk test support is small (n=30); per-class recall is noisy.
5. Course project only; not for clinical use.

## Artifacts

| File | Role |
| ---- | ---- |
| `outputs/metrics/report.json` | Full metrics schema for tables |
| `outputs/figures/confusion_matrix_test.png` | Best-model test CM |
| `outputs/run_summary.zip` → `tables/metrics_comparison.csv` | Flat comparison for teammates |

## What Phase 5 rejected

| Idea | Why rejected |
| ---- | ------------ |
| Accuracy as model-selection metric | Majority-class bias |
| Tuning baseline cutpoints on test | Holdout leakage |
| Hypothesis tests between classifiers | Separate teammate track |
| `class_weight="balanced"` retrain | Locked out of this run; optional future work |

## How to refresh numbers

Runtime → **Run all** on the Colab notebook, then replace tables from the new `report.json`. Do not invent metrics that are not in that file.
