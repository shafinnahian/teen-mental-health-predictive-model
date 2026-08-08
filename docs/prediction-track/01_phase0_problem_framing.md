# Phase 0 - Problem framing and study decisions

Phase 0 locked the scientific question before any modeling code. Report authors can lift the framing below into the DS1 introduction (objective, dataset, problem statement) and keep the technical exclusions in methods.

## Objective (non-technical)

We ask whether everyday lifestyle signals (screen use, sleep, school performance, physical activity, social interaction) can classify a teen's digital wellbeing into Healthy, Moderate, or At Risk. A usable answer would help educators and parents see which observable habits track with poorer digital wellbeing, without requiring clinical questionnaires as model inputs.

This remains an educational analysis of a public dataset. It is not a diagnostic tool.

## Problem statement

**Given:** a tabular sample of 1,200 rows with lifestyle fields and a three-level wellbeing flag.

**Predict:** `digital_wellbeing_flag`.

**Constraints:**

1. Use only lifestyle predictors (nine columns listed in [`00_scope_and_locked_decisions.md`](00_scope_and_locked_decisions.md)).
2. Do not use stress, anxiety, addiction, or derived risk / depression / sleep-quality labels as predictors.
3. Compare learned classifiers against simple baselines under class imbalance.
4. Report train and test metrics, with **macro-F1** as the primary test metric.

**Challenges:** Moderate dominates (~62% of rows). At Risk is rare (151 rows; 30 in the holdout test with seed 42). The table is complete (no missing values) but not a clinical sample, so perfect scores should be interpreted with skepticism.

## Dataset choice

| Item | Detail |
| ---- | ------ |
| Source | [Teen Mental Health on Kaggle (argonnxx)](https://www.kaggle.com/datasets/argonnxx/teen-mental-health) |
| Local / Colab path | `data/Teen_Mental_Health.csv` |
| Shape | 1200 rows × 16 columns |
| Missing | 0 |
| Target | `digital_wellbeing_flag` |

Why this dataset: it is small enough for Colab, rich enough for multiclass classification plus EDA, and mixes lifestyle fields with psychological scales so we can **demonstrate** leakage risk by excluding the scales rather than ignoring the issue.

## Why this target

We considered other columns as responses (`depression_label`, `sleep_quality`, `mental_health_risk_score`). We rejected them to keep one coherent prediction story:

- `digital_wellbeing_flag` is already a three-class label suitable for classification metrics.
- A continuous risk score would shift the project into regression and still sit next to the stress/anxiety/addiction identity.
- Multiple targets would dilute the report and invite inconsistent feature policies.

## Why psychological scales are excluded

On every row, the notebook asserts:

```text
mental_health_risk_score == stress_level + anxiety_level + addiction_level
```

That identity passed for all 1,200 rows. Including those three scales (or the risk score) as predictors would let the model solve a near-tautology relative to mental-health constructs that likely co-vary with the wellbeing flag. The lifestyle-only policy answers a harder and more honest question: can non-scale behavior fields still separate classes?

Excluded from `X` (see `EXCLUDED_COLUMNS` in `src/config.py`):

- `stress_level`, `anxiety_level`, `addiction_level`
- `mental_health_risk_score`, `depression_label`, `sleep_quality`

## Why macro-F1 is the primary metric

Class counts: Moderate 743, Healthy 306, At Risk 151.

A classifier that always predicts Moderate achieves about 62% accuracy on the test split (0.621 in `outputs/metrics/report.json`) while scoring about 0.255 macro-F1. Accuracy would declare that trivial rule "mostly right." Macro-F1 averages per-class F1 without weighting by class size, so minority errors matter.

Weighted F1 and accuracy are still reported for completeness. Model selection between logistic regression and random forest uses **test macro-F1** only.

## What this phase rejected

| Idea | Reason rejected |
| ---- | --------------- |
| Using psycho scales as features | Construct leakage via the verified risk-score identity |
| Predicting several targets at once | Scope creep; muddies methods and metrics |
| Accuracy as the headline metric | Inflated by Moderate dominance |
| Clinical claims or deployment framing | Out of scope for a course prototype |

## Teammate ownership note

Hypothesis testing sections of the DS1 rubric (clear hypothesis, test statistic, steps, conclusion) are **not** implemented in this repository. Another teammate owns that track. Prediction-track authors should not invent placeholder tests in the report.
