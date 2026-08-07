"""Phase 4 model builders: multinomial logistic regression + random forest pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "scikit-learn is not installed. In Colab, run the setup cell that executes "
        "%pip install -r requirements.txt before the Phase 4 modeling cells."
    ) from exc

try:
    from config import SEED
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import config from src/. Run the notebook setup cells that add "
        "src/ to sys.path before importing models."
    ) from exc

try:
    from preprocessing import build_preprocessing_pipeline
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import preprocessing from src/. Ensure Phase 3 modules are present "
        "and src/ is on sys.path."
    ) from exc

LOGISTIC_PIPELINE_FILENAME = "logistic_pipeline.joblib"
RF_PIPELINE_FILENAME = "rf_pipeline.joblib"
PHASE4_MODEL_META_FILENAME = "phase4_model_meta.json"

CV_N_SPLITS = 5
CV_SCORING = "f1_macro"
CV_N_JOBS = 1

LOGISTIC_C_GRID = [0.1, 1.0, 10.0]
RF_MAX_DEPTH_GRID = [None, 10, 20]


def build_logistic_classifier() -> LogisticRegression:
    """Multinomial logistic regression via lbfgs (3-class softmax loss).

    multi_class is omitted: deprecated in sklearn 1.5+; with solver='lbfgs' and
    n_classes >= 3 the multinomial loss is used automatically.
    """
    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    )


def build_rf_classifier() -> RandomForestClassifier:
    """Random forest with locked n_estimators; max_depth tuned in GridSearchCV."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
    )


def build_model_pipeline(classifier: Any) -> Pipeline:
    """Full Pipeline: Phase 3 preprocess steps + classifier on raw lifestyle X."""
    prep = build_preprocessing_pipeline()
    steps = list(prep.steps) + [("classifier", classifier)]
    return Pipeline(steps=steps)


def logistic_param_grid() -> dict[str, list[Any]]:
    """One tunable hyperparameter for logistic regression: C."""
    return {"classifier__C": list(LOGISTIC_C_GRID)}


def rf_param_grid() -> dict[str, list[Any]]:
    """One tunable hyperparameter for random forest: max_depth."""
    return {"classifier__max_depth": list(RF_MAX_DEPTH_GRID)}


def tune_and_fit_pipeline(
    base_pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Stratified 5-fold GridSearchCV on train only; scoring=f1_macro.

    Returns the fitted GridSearchCV. Caller should persist search.best_estimator_.
    Does not touch the held-out test set.
    """
    if X_train.empty:
        raise ValueError("X_train is empty. Run Phase 3 stratified_split first.")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train/y_train length mismatch (X={len(X_train)}, y={len(y_train)})."
        )

    cv = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )
    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=CV_SCORING,
        cv=cv,
        refit=True,
        return_train_score=False,
        n_jobs=CV_N_JOBS,
    )
    search.fit(X_train, y_train)
    return search
