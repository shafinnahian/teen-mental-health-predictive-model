# DS1 rubric crosswalk (prediction track)

Use this table when drafting the final report. Point values match the Data Science 1 final project checklist. Items need not appear as separate report headings; they must appear clearly in the prose.

This repository covers **predictive modeling**. Hypothesis testing is a separate teammate responsibility.

## Introduction and data

| Rubric item | Pts | Where to find content | Evidence / artifacts |
| ----------- | --- | --------------------- | -------------------- |
| Objective | 5 | [`01_phase0_problem_framing.md`](01_phase0_problem_framing.md); notebook intro | Non-technical goal: classify digital wellbeing from lifestyle signals |
| Dataset introduction | 5 | [`01_phase0_problem_framing.md`](01_phase0_problem_framing.md), [`02_phase1_scaffold_and_reproducibility.md`](02_phase1_scaffold_and_reproducibility.md) | Kaggle URL; 1200×16; 0 missing; `data/Teen_Mental_Health.csv` |
| Problem statement | 5 | [`01_phase0_problem_framing.md`](01_phase0_problem_framing.md), [`00_scope_and_locked_decisions.md`](00_scope_and_locked_decisions.md) | Multiclass prediction; lifestyle-only; imbalance challenges |
| Some preprocessing | 10 | [`04_phase3_preprocessing_rationale.md`](04_phase3_preprocessing_rationale.md); notebook Part 3 | Stratified split; one-hot; scaling; train-only fit; artifacts in `outputs/models/` |

## Exploratory analysis and features

| Rubric item | Pts | Where to find content | Evidence / artifacts |
| ----------- | --- | --------------------- | -------------------- |
| Useful plot(s) and/or table(s) | 20 | [`03_phase2_eda_rationale.md`](03_phase2_eda_rationale.md); notebook Part 2 | `outputs/figures/01_class_balance.png` … `04_platform_crosstab.png`; target counts; social-media means |
| Feature extraction / engineering | 10 | [`04_phase3_preprocessing_rationale.md`](04_phase3_preprocessing_rationale.md) | `high_screen_before_bed = 1 if screen_time_before_sleep > 2.0 else 0` |

## Hypothesis testing (not in this repo)

| Rubric item | Pts | Ownership |
| ----------- | --- | --------- |
| Clear statement of hypothesis | 3 | **Separate teammate** |
| Choice of test statistic | 4 | **Separate teammate** |
| Testing steps, values, conclusion | 8 | **Separate teammate** |

Do not invent placeholder tests from the prediction pipeline. If the team uses DS1 Option A, keep hypothesis testing independent of these model metrics unless the hypothesis teammate designs that link explicitly.

## Classification models

| Rubric item | Pts | Where to find content | Evidence / artifacts |
| ----------- | --- | --------------------- | -------------------- |
| Data splitting strategy | 3 | [`04_phase3_preprocessing_rationale.md`](04_phase3_preprocessing_rationale.md) | Stratified 80/20, `SEED=42`, 960/240; class count tables |
| Mathematical expression of model(s) + justification | 5 | [`05_phase4_modeling_rationale.md`](05_phase4_modeling_rationale.md); notebook Part 4 | Softmax / multinomial logistic; random forest; why each was chosen |
| How hyperparameters were chosen | 4 | [`05_phase4_modeling_rationale.md`](05_phase4_modeling_rationale.md) | Train-only stratified 5-fold `GridSearchCV`, `f1_macro`; `phase4_model_meta.json` |
| Performance on train and test + metrics | 8 | [`06_phase5_evaluation_rationale.md`](06_phase5_evaluation_rationale.md) | `outputs/metrics/report.json`; baselines; confusion matrix PNG |
| Conclusion | 5 | Synthesize from [`00_scope_and_locked_decisions.md`](00_scope_and_locked_decisions.md) + Phase 5 limitations | Key findings, RF perfect-score caution, lifestyle-only scope, future work (e.g. `class_weight` sensitivity). Teammates write the prose. |
| Proper citation | 5 | See bibliography seeds below | Dataset, libraries, any external methods papers the team actually read |

## Bibliography seeds (verify before citing)

Cite only what you used. Confirm versions from your Colab run (`run_summary` packages section) when possible.

| Source | Suggested citation target |
| ------ | ------------------------- |
| Dataset | argonnxx. *Teen Mental Health*. Kaggle. https://www.kaggle.com/datasets/argonnxx/teen-mental-health |
| pandas | https://pandas.pydata.org/ |
| scikit-learn | https://scikit-learn.org/ |
| matplotlib | https://matplotlib.org/ |
| seaborn | https://seaborn.pydata.org/ |
| NumPy | https://numpy.org/ |
| joblib | https://joblib.readthedocs.io/ |

Do not invent DOIs, paper titles, or statistics that are not in the run artifacts or primary sources you opened.

## Suggested report writing order

1. Objective + problem + dataset (Phase 0-1 docs).
2. EDA figures and implications (Phase 2).
3. Preprocessing + feature engineering + split (Phase 3).
4. Model math + hyperparameter protocol (Phase 4).
5. Baselines + train/test tables + confusion matrix + limitations (Phase 5).
6. Conclusion that stays inside evidence (no clinical claims).
7. Citations.
8. Hand off to the hypothesis-testing teammate for their 15 points.

## Framing checklist before submission

- [ ] Lifestyle-only predictors stated explicitly
- [ ] Leakage identity (`risk = stress + anxiety + addiction`) mentioned as reason for exclusion
- [ ] Macro-F1 justified via class imbalance
- [ ] Hyperparameters tuned on train CV only
- [ ] Test set described as a single final comparison
- [ ] Random forest perfect scores discussed with skepticism
- [ ] "Not for clinical use" stated
- [ ] Hypothesis testing left to the teammate who owns it
