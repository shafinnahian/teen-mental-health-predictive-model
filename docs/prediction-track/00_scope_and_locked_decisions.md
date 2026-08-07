# Scope and locked decisions

Constants live in [`src/config.py`](../../src/config.py), [`src/features.py`](../../src/features.py), [`src/models.py`](../../src/models.py), and [`src/evaluation.py`](../../src/evaluation.py). This page records **what** was locked and **why**, so report authors do not reopen settled choices.

## Limitations (read first)

- The table is a public Kaggle CSV labeled as teen mental-health related data. Treat it as a synthetic or curated tabular benchmark, not a clinical cohort.
- Moderate is the majority class (743 / 1200 ≈ 62%). Accuracy alone is a weak headline metric.
- After a stratified 80/20 split with `SEED=42`, the test set has only 30 At Risk rows. Per-class recall for that class is noisy.
- This pipeline is a course project. It is not for clinical use.

## Locked decision register

| Decision | Locked value | Why |
| -------- | ------------ | --- |
| Analytical track | Prediction-only (DS1 Option A style) | Keep one auditable classification pipeline. Hypothesis testing is owned by another teammate. |
| Target | `digital_wellbeing_flag` ∈ {Healthy, Moderate, At Risk} | Natural multiclass response in the CSV; readable for a non-technical objective statement. |
| Task | Multiclass classification | Matches the label type; enables macro-F1 and per-class recall under imbalance. |
| Predictors | Lifestyle-only (9 columns) | Observable digital / sleep / school / activity variables without using psychological scales. |
| Excluded predictors | `stress_level`, `anxiety_level`, `addiction_level` | They sum exactly to `mental_health_risk_score` on all 1200 rows. Using them would leak construct overlap and inflate scores. |
| Excluded alternate targets | `mental_health_risk_score`, `depression_label`, `sleep_quality` | One prediction problem only; avoids mixing response definitions mid-project. |
| Engineered feature | `high_screen_before_bed = 1 if screen_time_before_sleep > 2.0 else 0` | Simple sleep-hygiene flag. Threshold fixed a priori, not learned from labels. |
| Encoding | One-hot with `drop="first"` for `gender`, `platform_usage`, `social_interaction_level` | Standard categorical encoding without a full dummy trap. |
| Scaling | `StandardScaler` on 6 numeric lifestyle columns; binary passthrough | Needed for regularized logistic regression; shared with the forest for one pipeline API. |
| Split | Stratified 80/20, `SEED=42` (960 / 240) | Preserve class shares; fixed seed for reproducibility. |
| Models | Multinomial logistic regression (`lbfgs`) + random forest (`n_estimators=100`) | Interpretable linear model vs nonlinear ensemble. Two models is enough for DS1 justification. |
| Tuning | Train-only stratified 5-fold `GridSearchCV`, `scoring=f1_macro` | Select hyperparameters without touching the holdout test set. |
| Logistic grid | `C ∈ {0.1, 1.0, 10.0}` | Single regularization knob; small grid keeps Colab runtime short. |
| RF grid | `max_depth ∈ {None, 10, 20}` | Main complexity control for trees; `n_estimators` held fixed. |
| `class_weight` | Default `None` | Simpler first pass. Balanced weights left as optional future sensitivity, not run. |
| Baselines | Always-Moderate + social-media hours rule (≤3.5 Healthy, >6.0 At Risk, else Moderate) | Majority-class floor plus an EDA-motivated univariate rule. |
| Primary metric | Test **macro-F1** | Treats classes equally despite Moderate dominance. |
| Best ML model rule | Max test macro-F1 among logistic and random forest | Baselines are reported but not eligible for "best model." |
| Runtime | Google Colab + git clone | Matches the course demo environment. |
| Dataset | [Kaggle - argonnxx/teen-mental-health](https://www.kaggle.com/datasets/argonnxx/teen-mental-health) | Public CSV with 1200 rows × 16 columns, 0 missing. |
| Deployment CSV | `data/Teen_Mental_Health.csv` | Bundled path used by Colab and local loaders. |

## Lifestyle predictors (exact list)

From `LIFESTYLE_FEATURES` in `src/config.py`:

`age`, `gender`, `daily_social_media_hours`, `platform_usage`, `sleep_hours`, `screen_time_before_sleep`, `academic_performance`, `physical_activity`, `social_interaction_level`

## Decision notes (expanded)

### Why prediction-only in this repo

DS1 allows either separate hypothesis testing and predictive modeling, or hypothesis tests that compare model performance. This codebase implements **predictive modeling only**: train/test split, two classifiers, baselines, and metrics. That keeps leakage control and evaluation rules in one place. Statistical hypothesis tests between models are intentionally absent.

### Why lifestyle-only features

If stress, anxiety, and addiction enter the feature matrix, the model can recover nearly the same information as the composite risk score that is algebraically tied to those columns. That would look strong on paper and fail the scientific claim "lifestyle variables predict wellbeing class." Lifestyle-only features make the claim honest: can digital behavior, sleep, academics, and activity sort Healthy / Moderate / At Risk without psycho scales?

### Why two models, not more

Logistic regression supplies a clear multinomial (softmax) math story and coefficient-style interpretation after scaling. Random forest supplies nonlinear interactions without hand-written terms. Adding more learners would widen the report without improving the pedagogical signal for DS1.

### Why one combined notebook

Graders and teammates get a single Colab **Run all** path: setup → EDA → preprocessing → modeling → evaluation → run summary. Logic still lives in `src/` so the notebook stays thin.

## Schema facts used everywhere

Verified from `data/Teen_Mental_Health.csv` and enforced in `src/data_loader.py` / `src/config.py`:

| Fact | Value |
| ---- | ----- |
| Shape | 1200 × 16 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Target counts | Moderate 743, Healthy 306, At Risk 151 |
| Risk identity | `mental_health_risk_score = stress_level + anxiety_level + addiction_level` on all rows |
| Processed width after Phase 3 | 14 columns |
