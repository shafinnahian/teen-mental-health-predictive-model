# Phase 2 - EDA rationale

Phase 2 explored the data **for the lifestyle-only prediction question**. Plots guide later encoding, baselines, and metric choices. They are not exploratory fishing for a new target.

Figures save to `outputs/figures/` (not only inline display). Filenames are fixed in `src/run_summary.py`:

- `01_class_balance.png`
- `02_social_media_by_wellbeing.png`
- `03_lifestyle_correlation.png`
- `04_platform_crosstab.png`

## Leakage check (before plots)

The notebook verifies that on every row:

```text
mental_health_risk_score = stress_level + anxiety_level + addiction_level
```

That check **passed for all 1,200 rows**. Psychological columns stay out of modeling feature panels. A mean-by-class table of those scales is shown only for documentation, so readers see why exclusion matters.

## Figure 1 - Class balance

**What it shows:** counts of Healthy (306), Moderate (743), At Risk (151).

**Why we plotted it:** imbalance is the main evaluation threat. It justifies:

- stratified splitting (Phase 3)
- macro-F1 as the primary metric (Phase 4-5)
- an always-Moderate baseline (Phase 5)

**Decision driven:** do not lead the report with accuracy alone. Expect a majority-class baseline near 62% accuracy and near 0.25 macro-F1.

## Figure 2 - Daily social media hours by wellbeing class

**What it shows:** boxplot of `daily_social_media_hours` across Healthy / Moderate / At Risk, plus a group means table.

**Located means** (from `data/Teen_Mental_Health.csv`):

| Class | Mean hours | Median hours |
| ----- | ---------- | ------------ |
| Healthy | 2.55 | 2.6 |
| Moderate | 4.84 | 5.0 |
| At Risk | 7.09 | 7.2 |

**Why we plotted it:** among lifestyle numerics, hours of social media show the clearest class separation in EDA. That motivates keeping the column as a core predictor and motivates Baseline 2 (fixed hour thresholds) in Phase 5.

**Decision driven:** social-media threshold baseline with cutpoints 3.5 and 6.0 (round values near midpoints between class means on the **train** view; not optimized on the test set). Details in [`06_phase5_evaluation_rationale.md`](06_phase5_evaluation_rationale.md).

## Figure 3 - Lifestyle numeric correlation heatmap

**What it shows:** Pearson correlations among the six lifestyle numerics: `age`, `daily_social_media_hours`, `sleep_hours`, `screen_time_before_sleep`, `academic_performance`, `physical_activity`.

**Why we plotted it:** check for severe multicollinearity before scaling and logistic regression. Psycho scales are **omitted on purpose** so the heatmap matches the modeling feature set.

**Decision driven:** keep all nine base lifestyle predictors (six numeric + three categorical). EDA did not show redundancy strong enough to drop a lifestyle column at this stage. Formal VIF or stepwise selection was not run; that is a limitation, not a claim of independence.

## Figure 4 - Platform usage × wellbeing (row %)

**What it shows:** row-normalized crosstab heatmap of `platform_usage` vs wellbeing class, with raw counts displayed separately.

**Why we plotted it:** platform is a categorical lifestyle signal. Row percentages answer "given a platform, what is the class mix?" without letting platform sample size dominate the eye.

**Decision driven:** keep `platform_usage` and one-hot encode it. Separation by platform is weaker than separation by social-media hours, so platform alone would be a poor model; as part of the lifestyle set it still belongs.

## What EDA deliberately omitted

| Omitted | Why |
| ------- | --- |
| Psycho-scale panels as modeling evidence | Would encourage leakage into the feature story |
| Train/test split during EDA | EDA is descriptive on the full table; leakage-safe fitting starts in Phase 3 |
| Formal hypothesis tests | Owned by another teammate / track |
| Interaction heatmaps / PCA | Scope control; one clear engineered binary comes later |

## Strengths

- Complete cases (0 missing), so EDA and modeling share the same rows.
- Explicit algebraic leakage check before feature plots.
- Figures persist on disk for the report and the run-summary ZIP.
- Summary markdown in the notebook states the modeling implications (imbalance, social-media signal, exclusions).

## Concerns (critical view)

- The CSV is a public benchmark table. Patterns may be partly constructed; strong visual separation can overstate real-world signal.
- Most views are univariate or bivariate. They do not prove that a multivariate model will generalize.
- No formal interaction testing or multiple-comparison control in EDA (appropriate for descriptive work; do not overclaim).
- Class means for social media informed baseline cutpoints. Those cutpoints must stay fixed and train-motivated, never retuned on the holdout test set.

## How teammates should use this in the report

For the DS1 EDA points: include the four figures, the target count table, the social-media means table, and a short paragraph that links each plot to a modeling decision (metric choice, baseline, feature retention, encoding). Stay inside lifestyle-only framing.
