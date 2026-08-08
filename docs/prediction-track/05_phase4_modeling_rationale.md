# Phase 4 - Modeling rationale

Phase 4 trains two multiclass classifiers on lifestyle features only. Implementation: [`src/models.py`](../../src/models.py), notebook Part 4. **No test-set scoring happens here.** Final comparison is Phase 5.

## Design principle: full Pipeline on raw X

Each model is:

```text
engineer → preprocess → classifier
```

built by `build_model_pipeline`. Training inputs are raw lifestyle rows (`X_train` with the nine columns), not a separately transformed matrix.

**Why:**

1. Predict-time transforms match train-time transforms.
2. When `GridSearchCV` refits, preprocessing statistics are recomputed inside each training fold (no CV leakage from a single global scaler fit).
3. The saved joblib is a single deployable object for later evaluation cells.

## Model A - Multinomial logistic regression

For classes \(k \in \{\mathrm{Healthy}, \mathrm{Moderate}, \mathrm{At\ Risk}\}\):

$$
P(y=k \mid \mathbf{x}) = \mathrm{softmax}(\mathbf{w}_k^\top \mathbf{x} + b_k).
$$

Code: `LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)`.

With three classes and `lbfgs`, scikit-learn uses the multinomial (softmax) loss. The deprecated `multi_class` argument is omitted on purpose (sklearn 1.5+).

**Why this model:**

- Clear mathematical form for the DS1 "model math" rubric item.
- Coefficients (on scaled features) are interpretable as log-odds directions.
- Serves as a strong linear reference next to a nonlinear ensemble.

**Tuned hyperparameter:** regularization strength `C ∈ {0.1, 1.0, 10.0}` (inverse regularization). Larger `C` means less regularization.

**Local train-only CV result** (`outputs/models/phase4_model_meta.json` / `report.json` `cv_meta`):

| Best params | Best CV macro-F1 |
| ----------- | ---------------- |
| `classifier__C=10.0` | 0.904 |

## Model B - Random forest

Code: `RandomForestClassifier(n_estimators=100, random_state=42)`.

**Why this model:**

- Captures nonlinear interactions (for example social media hours × sleep) without hand-written terms.
- Robust to feature scaling (still shares the preprocess pipeline for API uniformity).
- Standard multiclass classifier taught in DS1-style courses.

**Fixed:** `n_estimators=100` for a stable ensemble without a second tuning axis.

**Tuned hyperparameter:** `max_depth ∈ {None, 10, 20}`. Depth is the main overfitting control; `None` allows full trees.

**Local train-only CV result:**

| Best params | Best CV macro-F1 |
| ----------- | ---------------- |
| `classifier__max_depth=None` | 0.991 |

## Hyperparameter search protocol

| Setting | Value | Why |
| ------- | ----- | --- |
| Search | `GridSearchCV` | Small grids; exact enumeration is fine |
| Folds | Stratified 5-fold, `shuffle=True`, `random_state=42` | Preserve class mix in every fold |
| Scoring | `f1_macro` | Aligns with primary evaluation metric |
| Data used | **Training set only** | Holdout test untouched until Phase 5 |
| `refit` | `True` | Refit best params on full train for saving |
| `n_jobs` | 1 | Predictable Colab CPU use |

**Why not tune more knobs:** one hyperparameter per model keeps the report honest and Colab runtime short. Expanding grids invites multiple-comparison noise without better science on this table.

## Why `class_weight` stayed at default `None`

Imbalance is real, so `class_weight="balanced"` was considered. It was **not** enabled for the locked run, to keep the first pass simple and comparable to the unweighted baselines. Teammates may mention it as future sensitivity work. It was not executed in Phase 4 or 5.

## What Phase 4 intentionally does not do

- Score the holdout test set (avoids peeking / informal multiple testing before the final table).
- Compare models with a statistical hypothesis test (separate teammate track).
- Train on psycho-scale features.
- Persist only the classifier without preprocessing (would break predict consistency).

## Artifacts

| File | Contents |
| ---- | -------- |
| `outputs/models/logistic_pipeline.joblib` | Best logistic full pipeline |
| `outputs/models/rf_pipeline.joblib` | Best RF full pipeline |
| `outputs/models/phase4_model_meta.json` | Best params + CV macro-F1 per model |

Smoke tests reload both pipelines and predict on a tiny **train** slice only.

## Critical notes for report authors

1. High CV macro-F1 for the forest (~0.99) already hints that the table may be unusually separable with lifestyle features. Perfect later test scores should not be sold as clinical-grade generalization.
2. Prefer describing logistic regression as the interpretable reference and the forest as the flexible comparator.
3. Always state that hyperparameters were chosen on train CV with `f1_macro`, not on the test set.
