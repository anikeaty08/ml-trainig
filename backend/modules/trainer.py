from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from ..utils.metrics import classification_metrics, primary_metric_name, regression_metrics


ProgressCallback = Callable[[str, int, dict[str, Any] | None], None]


@dataclass
class Candidate:
    name: str
    estimator: Any
    params: dict[str, list[Any]]
    family: str


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [column for column in X.columns if is_numeric_dtype(X[column])]
    categorical_features = [column for column in X.columns if column not in numeric_features]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _candidates(task_type: str) -> list[Candidate]:
    if task_type == "classification":
        return [
            Candidate("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced"), {"model__C": [0.1, 1.0, 3.0]}, "baseline"),
            Candidate("Random Forest", RandomForestClassifier(random_state=42, class_weight="balanced"), {"model__n_estimators": [150, 250], "model__max_depth": [None, 12]}, "tree"),
            Candidate("Extra Trees", ExtraTreesClassifier(random_state=42, class_weight="balanced"), {"model__n_estimators": [150, 250], "model__max_depth": [None, 12]}, "tree"),
            Candidate("Hist Gradient Boosting", HistGradientBoostingClassifier(random_state=42), {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]}, "boosting"),
            Candidate("SVM", SVC(probability=True, class_weight="balanced"), {"model__C": [0.5, 1.0, 3.0], "model__gamma": ["scale", "auto"]}, "kernel"),
            Candidate("KNN", KNeighborsClassifier(), {"model__n_neighbors": [5, 11, 21]}, "instance"),
            Candidate("Naive Bayes", GaussianNB(), {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}, "probabilistic"),
        ]

    return [
        Candidate("Linear Regression", LinearRegression(), {}, "baseline"),
        Candidate("Ridge", Ridge(), {"model__alpha": [0.1, 1.0, 5.0]}, "baseline"),
        Candidate("Random Forest Regressor", RandomForestRegressor(random_state=42), {"model__n_estimators": [150, 250], "model__max_depth": [None, 12]}, "tree"),
        Candidate("Extra Trees Regressor", ExtraTreesRegressor(random_state=42), {"model__n_estimators": [150, 250], "model__max_depth": [None, 12]}, "tree"),
        Candidate("Hist Gradient Boosting Regressor", HistGradientBoostingRegressor(random_state=42), {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]}, "boosting"),
        Candidate("SVR", SVR(), {"model__C": [0.5, 1.0, 3.0], "model__gamma": ["scale", "auto"]}, "kernel"),
        Candidate("KNN Regressor", KNeighborsRegressor(), {"model__n_neighbors": [5, 11, 21]}, "instance"),
    ]


def _build_pipeline(preprocessor: ColumnTransformer, estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(estimator))])


def _evaluate(task_type: str, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    if task_type == "classification":
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") and len(np.unique(y)) == 2 else None
        return classification_metrics(y, predictions, probabilities)
    return regression_metrics(y, predictions)


def _feature_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        raw = np.abs(coefficients[0] if coefficients.ndim > 1 else coefficients)
    else:
        sample_size = min(len(X), 500)
        X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
        y_sample = y.loc[X_sample.index]
        raw = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42).importances_mean

    pairs = [{"feature": feature, "importance": round(float(score), 6)} for feature, score in zip(feature_names, raw)]
    return sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)[:15]


def _learning_curves(task_type: str, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
    scorer = get_scorer(primary_metric_name(task_type))
    train_sizes, train_scores, validation_scores = learning_curve(
        clone(model),
        X,
        y,
        train_sizes=np.linspace(0.3, 1.0, 4),
        cv=3,
        scoring=scorer,
        n_jobs=1,
    )
    summary = []
    for size, train_score, val_score in zip(train_sizes, train_scores, validation_scores):
        summary.append(
            {
                "train_size": int(size),
                "train_score": round(float(np.mean(train_score)), 4),
                "validation_score": round(float(np.mean(val_score)), 4),
            }
        )
    return summary


def _build_ensemble(task_type: str, preprocessor: ColumnTransformer, winners: list[dict[str, Any]]) -> Pipeline | None:
    if len(winners) < 2:
        return None

    estimators = []
    for item in winners[:3]:
        estimators.append(
            (
                item["name"].lower().replace(" ", "_"),
                clone(item["search"].best_estimator_.named_steps["model"]),
            )
        )

    model = VotingClassifier(estimators=estimators, voting="soft") if task_type == "classification" else VotingRegressor(estimators=estimators)
    return Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])


def train_candidate_models(
    df: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    preprocessor = _build_preprocessor(X)
    metric_name = primary_metric_name(task_type)
    candidates = _candidates(task_type)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y if task_type == "classification" else None,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp if task_type == "classification" else None,
    )

    train_val_X = pd.concat([X_train, X_val], axis=0)
    train_val_y = pd.concat([y_train, y_val], axis=0)

    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if progress_callback:
            progress_callback(
                f"Training {candidate.name}",
                55 + int(((index - 1) / max(len(candidates), 1)) * 25),
                {"current_model": candidate.name, "completed_models": index - 1, "total_models": len(candidates)},
            )

        started_at = time.perf_counter()
        search = GridSearchCV(
            estimator=_build_pipeline(preprocessor, candidate.estimator),
            param_grid=candidate.params or {},
            scoring=metric_name,
            cv=5,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        validation_model = search.best_estimator_
        validation_metrics = _evaluate(task_type, validation_model, X_val, y_val)

        final_model = clone(validation_model)
        final_model.fit(train_val_X, train_val_y)
        test_metrics = _evaluate(task_type, final_model, X_test, y_test)
        elapsed = round(time.perf_counter() - started_at, 3)

        results.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "search": search,
                "pipeline": final_model,
                "cv_score_mean": round(float(search.best_score_), 4),
                "cv_score_std": round(float(search.cv_results_["std_test_score"][search.best_index_]), 4),
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_time_seconds": elapsed,
                "params": search.best_params_,
                "evaluation": {
                    "X_test": X_test,
                    "y_test": y_test,
                    "predictions": final_model.predict(X_test),
                    "probabilities": final_model.predict_proba(X_test)[:, 1].tolist()
                    if task_type == "classification" and hasattr(final_model, "predict_proba") and len(np.unique(y_test)) == 2
                    else [],
                },
            }
        )

    sorted_results = sorted(results, key=lambda item: item["test_metrics"][metric_name], reverse=True)
    winners = sorted_results[:3]
    ensemble_model = _build_ensemble(task_type, preprocessor, winners)

    if ensemble_model is not None:
        ensemble_model.fit(train_val_X, train_val_y)
        ensemble_metrics = _evaluate(task_type, ensemble_model, X_test, y_test)
        sorted_results.append(
            {
                "name": "Voting Ensemble",
                "family": "ensemble",
                "search": winners[0]["search"],
                "pipeline": ensemble_model,
                "cv_score_mean": round(np.mean([winner["cv_score_mean"] for winner in winners]), 4),
                "cv_score_std": round(np.mean([winner["cv_score_std"] for winner in winners]), 4),
                "validation_metrics": winners[0]["validation_metrics"],
                "test_metrics": ensemble_metrics,
                "training_time_seconds": round(sum(winner["training_time_seconds"] for winner in winners), 3),
                "params": {"members": [winner["name"] for winner in winners]},
                "evaluation": {
                    "X_test": X_test,
                    "y_test": y_test,
                    "predictions": ensemble_model.predict(X_test),
                    "probabilities": ensemble_model.predict_proba(X_test)[:, 1].tolist()
                    if task_type == "classification" and hasattr(ensemble_model, "predict_proba") and len(np.unique(y_test)) == 2
                    else [],
                },
            }
        )
        sorted_results = sorted(sorted_results, key=lambda item: item["test_metrics"][metric_name], reverse=True)

    best = sorted_results[0]
    feature_importance = _feature_importance(best["pipeline"], X_test, y_test)
    learning_curves = _learning_curves(task_type, best["pipeline"], train_val_X, train_val_y)

    best_output = {
        "name": best["name"],
        "pipeline": best["pipeline"],
        "test_metrics": best["test_metrics"],
        "params": best["params"],
        "feature_importance": feature_importance,
        "learning_curves": learning_curves,
        "evaluation": best["evaluation"],
    }

    comparison = []
    for rank, item in enumerate(sorted_results, start=1):
        comparison.append(
            {
                "rank": rank,
                "name": item["name"],
                "family": item["family"],
                "primary_metric": metric_name,
                "primary_score": round(float(item["test_metrics"][metric_name]), 4),
                "cv_score_mean": item["cv_score_mean"],
                "cv_score_std": item["cv_score_std"],
                "training_time_seconds": item["training_time_seconds"],
                "params": item["params"],
                **item["test_metrics"],
            }
        )

    summary = {
        "metric_name": metric_name,
        "winner": best["name"],
        "winner_test_metrics": best["test_metrics"],
        "winner_params": best["params"],
        "learning_curves": learning_curves,
        "feature_importance": feature_importance,
    }
    return {"best_model": best_output, "comparison": comparison, "summary": summary}
