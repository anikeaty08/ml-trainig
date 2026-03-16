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
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from .advanced_models import train_optional_text_transformers, train_optional_timeseries_specialists
from .deep_learning import train_optional_deep_models
from ..utils.metrics import classification_metrics, primary_metric_name, regression_metrics


ProgressCallback = Callable[[str, int, dict[str, Any] | None], None]


@dataclass
class Candidate:
    name: str
    estimator: Any
    params: dict[str, list[Any]]
    family: str


def _optional_estimator(module_name: str, class_name: str) -> Any | None:
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None


def _optional_boosting_candidates(task_type: str) -> list[Candidate]:
    candidates: list[Candidate] = []

    xgb_classifier = _optional_estimator("xgboost", "XGBClassifier")
    xgb_regressor = _optional_estimator("xgboost", "XGBRegressor")
    lgbm_classifier = _optional_estimator("lightgbm", "LGBMClassifier")
    lgbm_regressor = _optional_estimator("lightgbm", "LGBMRegressor")

    if task_type == "classification":
        if xgb_classifier is not None:
            candidates.append(
                Candidate(
                    "XGBoost",
                    xgb_classifier(
                        random_state=42,
                        n_estimators=180,
                        eval_metric="logloss",
                        max_depth=6,
                        learning_rate=0.08,
                    ),
                    {"model__n_estimators": [120, 180], "model__max_depth": [4, 6]},
                    "boosting",
                )
            )
        if lgbm_classifier is not None:
            candidates.append(
                Candidate(
                    "LightGBM",
                    lgbm_classifier(random_state=42, n_estimators=180, learning_rate=0.08),
                    {"model__n_estimators": [120, 180], "model__num_leaves": [31, 63]},
                    "boosting",
                )
            )
    else:
        if xgb_regressor is not None:
            candidates.append(
                Candidate(
                    "XGBoost Regressor",
                    xgb_regressor(random_state=42, n_estimators=180, max_depth=6, learning_rate=0.08),
                    {"model__n_estimators": [120, 180], "model__max_depth": [4, 6]},
                    "boosting",
                )
            )
        if lgbm_regressor is not None:
            candidates.append(
                Candidate(
                    "LightGBM Regressor",
                    lgbm_regressor(random_state=42, n_estimators=180, learning_rate=0.08),
                    {"model__n_estimators": [120, 180], "model__num_leaves": [31, 63]},
                    "boosting",
                )
            )

    return candidates


def _dense_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _dense_categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def _build_tabular_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [column for column in X.columns if is_numeric_dtype(X[column])]
    categorical_features = [column for column in X.columns if column not in numeric_features]

    return ColumnTransformer(
        transformers=[
            ("num", _dense_numeric_pipeline(), numeric_features),
            ("cat", _dense_categorical_pipeline(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _build_text_preprocessor(X: pd.DataFrame, text_column: str) -> ColumnTransformer:
    numeric_features = [column for column in X.columns if column != text_column and is_numeric_dtype(X[column])]
    categorical_features = [column for column in X.columns if column not in numeric_features and column != text_column]

    transformers: list[tuple[str, Any, Any]] = [
        (
            "text",
            TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english"),
            text_column,
        )
    ]
    if numeric_features:
        transformers.append(("num", _dense_numeric_pipeline(), numeric_features))
    if categorical_features:
        transformers.append(("cat", _dense_categorical_pipeline(), categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _tabular_candidates(task_type: str) -> list[Candidate]:
    if task_type == "classification":
        return [
            Candidate("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced"), {"model__C": [0.1, 1.0, 3.0]}, "baseline"),
            Candidate("Random Forest", RandomForestClassifier(random_state=42, class_weight="balanced"), {"model__n_estimators": [150, 250], "model__max_depth": [None, 12]}, "tree"),
            Candidate("Extra Trees", ExtraTreesClassifier(random_state=42, class_weight="balanced"), {"model__n_estimators": [150, 250], "model__max_depth": [None, 12]}, "tree"),
            Candidate("Gradient Boosting", GradientBoostingClassifier(random_state=42), {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]}, "boosting"),
            Candidate("Hist Gradient Boosting", HistGradientBoostingClassifier(random_state=42), {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]}, "boosting"),
            Candidate("SVM", SVC(probability=True, class_weight="balanced"), {"model__C": [0.5, 1.0, 3.0], "model__gamma": ["scale", "auto"]}, "kernel"),
            Candidate(
                "Neural Network (MLP)",
                MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
                {
                    "model__hidden_layer_sizes": [(128,), (256, 128), (256, 128, 64)],
                    "model__activation": ["relu", "tanh"],
                    "model__alpha": [0.0001, 0.001],
                    "model__learning_rate_init": [0.001, 0.0005],
                },
                "neural",
            ),
            Candidate("KNN", KNeighborsClassifier(), {"model__n_neighbors": [5, 11, 21]}, "instance"),
            Candidate("Naive Bayes", GaussianNB(), {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}, "probabilistic"),
            *_optional_boosting_candidates("classification"),
        ]

    return [
        Candidate("Linear Regression", LinearRegression(), {}, "baseline"),
        Candidate("Ridge", Ridge(), {"model__alpha": [0.1, 1.0, 5.0]}, "baseline"),
        Candidate("Lasso", Lasso(max_iter=5000), {"model__alpha": [0.001, 0.01, 0.1]}, "baseline"),
        Candidate("Random Forest Regressor", RandomForestRegressor(random_state=42), {"model__n_estimators": [150, 250], "model__max_depth": [None, 12]}, "tree"),
        Candidate("Extra Trees Regressor", ExtraTreesRegressor(random_state=42), {"model__n_estimators": [150, 250], "model__max_depth": [None, 12]}, "tree"),
        Candidate("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42), {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]}, "boosting"),
        Candidate("Hist Gradient Boosting Regressor", HistGradientBoostingRegressor(random_state=42), {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]}, "boosting"),
        Candidate("SVR", SVR(), {"model__C": [0.5, 1.0, 3.0], "model__gamma": ["scale", "auto"]}, "kernel"),
        Candidate(
            "Neural Network Regressor",
            MLPRegressor(random_state=42, max_iter=500, early_stopping=True),
            {
                "model__hidden_layer_sizes": [(128,), (256, 128), (256, 128, 64)],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001, 0.0005],
            },
            "neural",
        ),
        Candidate("KNN Regressor", KNeighborsRegressor(), {"model__n_neighbors": [5, 11, 21]}, "instance"),
        *_optional_boosting_candidates("regression"),
    ]


def _text_candidates(task_type: str) -> list[Candidate]:
    if task_type == "classification":
        return [
            Candidate("Logistic Regression + TF-IDF", LogisticRegression(max_iter=1000, class_weight="balanced"), {"model__C": [0.5, 1.0, 3.0]}, "baseline"),
            Candidate("Naive Bayes + TF-IDF", GaussianNB(), {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}, "probabilistic"),
            Candidate("Random Forest + TF-IDF", RandomForestClassifier(random_state=42, class_weight="balanced"), {"model__n_estimators": [100, 150], "model__max_depth": [None, 12]}, "tree"),
            Candidate("Gradient Boosting + TF-IDF", GradientBoostingClassifier(random_state=42), {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]}, "boosting"),
            Candidate("Hist Gradient Boosting + TF-IDF", HistGradientBoostingClassifier(random_state=42), {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]}, "boosting"),
            Candidate("SVM + TF-IDF", SVC(probability=True, class_weight="balanced"), {"model__C": [0.5, 1.0, 2.0]}, "kernel"),
            Candidate(
                "Neural Network (MLP) + TF-IDF",
                MLPClassifier(random_state=42, max_iter=400, early_stopping=True),
                {
                    "model__hidden_layer_sizes": [(256,), (512, 256), (512, 256, 128)],
                    "model__activation": ["relu", "tanh"],
                    "model__alpha": [0.0001, 0.001],
                    "model__learning_rate_init": [0.001, 0.0005],
                },
                "neural",
            ),
            *_optional_boosting_candidates("classification"),
        ]

    return [
        Candidate("Ridge + TF-IDF", Ridge(), {"model__alpha": [0.1, 1.0, 5.0]}, "baseline"),
        Candidate("Random Forest Regressor + TF-IDF", RandomForestRegressor(random_state=42), {"model__n_estimators": [100, 150], "model__max_depth": [None, 12]}, "tree"),
        Candidate("Hist Gradient Boosting Regressor + TF-IDF", HistGradientBoostingRegressor(random_state=42), {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]}, "boosting"),
        Candidate("SVR + TF-IDF", SVR(), {"model__C": [0.5, 1.0, 2.0]}, "kernel"),
    ]


def _timeseries_candidates() -> list[Candidate]:
    return [
        Candidate("Lagged Ridge Regression", Ridge(), {"model__alpha": [0.1, 1.0, 5.0]}, "baseline"),
        Candidate("Random Forest Regressor", RandomForestRegressor(random_state=42), {"model__n_estimators": [120, 180], "model__max_depth": [None, 12]}, "tree"),
        Candidate("Extra Trees Regressor", ExtraTreesRegressor(random_state=42), {"model__n_estimators": [120, 180], "model__max_depth": [None, 12]}, "tree"),
        Candidate("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42), {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]}, "boosting"),
        Candidate("Hist Gradient Boosting Regressor", HistGradientBoostingRegressor(random_state=42), {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]}, "boosting"),
        Candidate(
            "Neural Network Regressor",
            MLPRegressor(random_state=42, max_iter=500, early_stopping=True),
            {
                "model__hidden_layer_sizes": [(128,), (256, 128), (256, 128, 64)],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001, 0.0005],
            },
            "neural",
        ),
        Candidate("SVR", SVR(), {"model__C": [0.5, 1.0, 2.0]}, "kernel"),
        *_optional_boosting_candidates("regression"),
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
    if hasattr(model, "feature_importance_summary"):
        return list(getattr(model, "feature_importance_summary"))
    if not hasattr(model, "named_steps"):
        return []
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        raw = np.abs(coefficients[0] if coefficients.ndim > 1 else coefficients)
    else:
        sample_size = min(len(X), 400)
        X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
        y_sample = y.loc[X_sample.index]
        raw = permutation_importance(model, X_sample, y_sample, n_repeats=3, random_state=42).importances_mean

    pairs = [{"feature": feature, "importance": round(float(score), 6)} for feature, score in zip(feature_names, raw)]
    return sorted(pairs, key=lambda item: abs(item["importance"]), reverse=True)[:15]


def _learning_curves(task_type: str, model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: Any) -> list[dict[str, Any]]:
    if hasattr(model, "learning_curve_"):
        return list(getattr(model, "learning_curve_"))
    if not hasattr(model, "named_steps"):
        return []
    if len(X) < 15:
        return []
    scorer = get_scorer(primary_metric_name(task_type))
    train_sizes, train_scores, validation_scores = learning_curve(
        clone(model),
        X,
        y,
        train_sizes=np.linspace(0.3, 1.0, 4),
        cv=cv,
        scoring=scorer,
        n_jobs=1,
    )
    return [
        {
            "train_size": int(size),
            "train_score": round(float(np.mean(train_score)), 4),
            "validation_score": round(float(np.mean(val_score)), 4),
        }
        for size, train_score, val_score in zip(train_sizes, train_scores, validation_scores)
    ]


def _build_ensemble(task_type: str, preprocessor: ColumnTransformer, winners: list[dict[str, Any]]) -> Pipeline | None:
    eligible_winners = [
        winner for winner in winners if winner.get("search") is not None and hasattr(winner.get("pipeline"), "named_steps")
    ]
    if len(eligible_winners) < 2:
        return None
    estimators = [
        (
            winner["name"].lower().replace(" ", "_").replace("+", "").replace("-", "_"),
            clone(winner["search"].best_estimator_.named_steps["model"]),
        )
        for winner in eligible_winners[:3]
    ]
    model = VotingClassifier(estimators=estimators, voting="soft") if task_type == "classification" else VotingRegressor(estimators=estimators)
    return Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])


def _run_candidate_searches(
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
    preprocessor: ColumnTransformer,
    candidates: list[Candidate],
    task_type: str,
    cv: Any,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    metric_name = primary_metric_name(task_type)
    results: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        if progress_callback:
            extra = {"current_model": candidate.name, "completed_models": index - 1, "total_models": len(candidates)}
            if candidate.family == "neural":
                extra["training_strategy"] = "Exploring multiple hidden-layer layouts, activations, and regularization settings"
            progress_callback(
                f"Training {candidate.name}",
                55 + int(((index - 1) / max(len(candidates), 1)) * 25),
                extra,
            )
        started_at = time.perf_counter()
        search = GridSearchCV(
            estimator=_build_pipeline(preprocessor, candidate.estimator),
            param_grid=candidate.params or {},
            scoring=metric_name,
            cv=cv,
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
                    "X_test": X_test.copy(),
                    "y_test": y_test.copy(),
                    "predictions": final_model.predict(X_test),
                    "probabilities": final_model.predict_proba(X_test)[:, 1].tolist()
                    if task_type == "classification" and hasattr(final_model, "predict_proba") and len(np.unique(y_test)) == 2
                    else [],
                },
            }
        )
    return results


def _prepare_timeseries_frames(
    df: pd.DataFrame,
    *,
    target_column: str,
    time_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_df = df.sort_values(time_column).reset_index(drop=True).copy()
    time_values = pd.to_datetime(ts_df[time_column], errors="coerce")
    lags = [1, 2, 3] + ([7] if len(ts_df) > 20 else [])

    supervised = pd.DataFrame(index=ts_df.index)
    supervised["time_ordinal"] = time_values.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    supervised["month"] = time_values.dt.month
    supervised["day"] = time_values.dt.day
    supervised["day_of_week"] = time_values.dt.dayofweek
    supervised["hour"] = time_values.dt.hour

    exogenous_columns = [column for column in ts_df.columns if column not in {target_column, time_column}]
    for column in exogenous_columns:
        supervised[column] = ts_df[column]

    for lag in lags:
        supervised[f"{target_column}_lag_{lag}"] = ts_df[target_column].shift(lag)
    supervised[f"{target_column}_rolling_mean_3"] = ts_df[target_column].shift(1).rolling(3).mean()
    supervised[f"{target_column}_rolling_std_3"] = ts_df[target_column].shift(1).rolling(3).std()
    supervised[target_column] = ts_df[target_column].shift(-1)
    supervised["prediction_time"] = time_values.shift(-1)

    supervised = supervised.dropna().reset_index(drop=True)
    display = pd.DataFrame(
        {
            time_column: supervised["prediction_time"],
        }
    )
    for column in exogenous_columns[:5]:
        display[column] = supervised[column]

    model_frame = supervised.drop(columns=["prediction_time"])
    return model_frame, display


def _package_results(
    *,
    task_type: str,
    preprocessor: ColumnTransformer,
    results: list[dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
    cv: Any,
) -> dict[str, Any]:
    metric_name = primary_metric_name(task_type)
    sorted_results = sorted(results, key=lambda item: item["test_metrics"][metric_name], reverse=True)
    winners = sorted_results[:3]
    ensemble_model = _build_ensemble(task_type, preprocessor, winners)

    if ensemble_model is not None:
        ensemble_members = [
            winner for winner in winners if winner.get("search") is not None and hasattr(winner.get("pipeline"), "named_steps")
        ][:3]
        ensemble_model.fit(train_val_X, train_val_y)
        ensemble_metrics = _evaluate(task_type, ensemble_model, X_test, y_test)
        sorted_results.append(
            {
                "name": "Voting Ensemble",
                "family": "ensemble",
                "search": ensemble_members[0]["search"],
                "pipeline": ensemble_model,
                "cv_score_mean": round(np.mean([winner["cv_score_mean"] for winner in ensemble_members]), 4),
                "cv_score_std": round(np.mean([winner["cv_score_std"] for winner in ensemble_members]), 4),
                "validation_metrics": ensemble_members[0]["validation_metrics"],
                "test_metrics": ensemble_metrics,
                "training_time_seconds": round(sum(winner["training_time_seconds"] for winner in ensemble_members), 3),
                "params": {"members": [winner["name"] for winner in ensemble_members]},
                "evaluation": {
                    "X_test": X_test.copy(),
                    "y_test": y_test.copy(),
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
    learning_curves = _learning_curves(task_type, best["pipeline"], train_val_X, train_val_y, cv=cv)

    best_output = {
        "name": best["name"],
        "pipeline": best["pipeline"],
        "test_metrics": best["test_metrics"],
        "params": best["params"],
        "feature_importance": feature_importance,
        "learning_curves": learning_curves,
        "evaluation": best["evaluation"],
    }

    comparison = [
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
        for rank, item in enumerate(sorted_results, start=1)
    ]

    return {
        "best_model": best_output,
        "comparison": comparison,
        "summary": {
            "metric_name": metric_name,
            "winner": best["name"],
            "winner_test_metrics": best["test_metrics"],
            "winner_params": best["params"],
            "learning_curves": learning_curves,
            "feature_importance": feature_importance,
        },
    }


def _train_tabular_or_text(
    df: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
    dataset_type: str,
    dataset_context: dict[str, Any] | None,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if dataset_type == "text":
        text_column = (dataset_context or {}).get("text_column")
        if not text_column:
            raise ValueError("Text dataset detected without a text_column in dataset_context")
        preprocessor = _build_text_preprocessor(X, text_column=text_column)
        candidates = _text_candidates(task_type)
    else:
        preprocessor = _build_tabular_preprocessor(X)
        candidates = _tabular_candidates(task_type)

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
    cv = 5

    results = _run_candidate_searches(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_val_X=train_val_X,
        train_val_y=train_val_y,
        preprocessor=preprocessor,
        candidates=candidates,
        task_type=task_type,
        cv=cv,
        progress_callback=progress_callback,
    )
    results.extend(
        train_optional_deep_models(
            preprocessor=preprocessor,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            train_val_X=train_val_X,
            train_val_y=train_val_y,
            task_type=task_type,
            dataset_type=dataset_type,
            progress_callback=(
                (lambda message, extra=None: progress_callback(message, 82, extra))
                if progress_callback
                else None
            ),
        )
    )
    if dataset_type == "text" and dataset_context and dataset_context.get("text_column"):
        results.extend(
            train_optional_text_transformers(
                task_type=task_type,
                text_column=dataset_context["text_column"],
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                train_val_X=train_val_X,
                train_val_y=train_val_y,
                progress_callback=(
                    (lambda message, extra=None: progress_callback(message, 84, extra))
                    if progress_callback
                    else None
                ),
            )
        )
    return _package_results(
        task_type=task_type,
        preprocessor=preprocessor,
        results=results,
        X_test=X_test,
        y_test=y_test,
        train_val_X=train_val_X,
        train_val_y=train_val_y,
        cv=cv,
    )


def _train_timeseries(
    df: pd.DataFrame,
    *,
    target_column: str,
    dataset_context: dict[str, Any] | None,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    time_column = (dataset_context or {}).get("time_column")
    if not time_column:
        raise ValueError("Time-series dataset detected without a time_column in dataset_context")

    model_df, display_df = _prepare_timeseries_frames(df, target_column=target_column, time_column=time_column)
    if len(model_df) < 18:
        raise ValueError("Time-series dataset is too short after lag generation; need at least 18 usable rows")

    X_model = model_df.drop(columns=[target_column])
    y = model_df[target_column]
    preprocessor = _build_tabular_preprocessor(X_model)
    candidates = _timeseries_candidates()

    train_end = max(int(len(model_df) * 0.6), 8)
    val_end = max(int(len(model_df) * 0.8), train_end + 4)
    val_end = min(val_end, len(model_df) - 2)

    X_train = X_model.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()
    X_val_model = X_model.iloc[train_end:val_end].copy()
    y_val = y.iloc[train_end:val_end].copy()
    X_test_model = X_model.iloc[val_end:].copy()
    y_test = y.iloc[val_end:].copy()
    X_train_display = display_df.iloc[:train_end].copy()
    X_val_display = display_df.iloc[train_end:val_end].copy()
    X_test_display = display_df.iloc[val_end:].copy()

    train_val_X_model = X_model.iloc[:val_end].copy()
    train_val_y = y.iloc[:val_end].copy()
    cv_splits = min(4, max(2, len(train_val_X_model) // 8))
    cv = TimeSeriesSplit(n_splits=cv_splits)

    results = _run_candidate_searches(
        X_train=X_train,
        X_val=X_val_model,
        X_test=X_test_model,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_val_X=train_val_X_model,
        train_val_y=train_val_y,
        preprocessor=preprocessor,
        candidates=candidates,
        task_type="regression",
        cv=cv,
        progress_callback=progress_callback,
    )
    results.extend(
        train_optional_deep_models(
            preprocessor=preprocessor,
            X_train=X_train,
            X_val=X_val_model,
            X_test=X_test_model,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            train_val_X=train_val_X_model,
            train_val_y=train_val_y,
            task_type="regression",
            dataset_type="timeseries",
            progress_callback=(
                (lambda message, extra=None: progress_callback(message, 82, extra))
                if progress_callback
                else None
            ),
        )
    )
    results.extend(
        train_optional_timeseries_specialists(
            X_train=X_train,
            X_val=X_val_model,
            X_test=X_test_model,
            X_train_display=X_train_display,
            X_val_display=X_val_display,
            X_test_display=X_test_display,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            train_val_X=train_val_X_model,
            train_val_y=train_val_y,
            time_column=time_column,
            progress_callback=(
                (lambda message, extra=None: progress_callback(message, 86, extra))
                if progress_callback
                else None
            ),
        )
    )

    for item in results:
        item["evaluation"]["X_test"] = X_test_display.copy()

    packaged = _package_results(
        task_type="regression",
        preprocessor=preprocessor,
        results=results,
        X_test=X_test_model,
        y_test=y_test,
        train_val_X=train_val_X_model,
        train_val_y=train_val_y,
        cv=cv,
    )
    packaged["best_model"]["evaluation"]["X_test"] = X_test_display.copy()
    packaged["summary"]["forecast_horizon"] = 1
    packaged["summary"]["time_column"] = time_column
    return packaged


def train_candidate_models(
    df: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
    dataset_type: str = "tabular",
    dataset_context: dict[str, Any] | None = None,
    source_path: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if dataset_type == "timeseries":
        return _train_timeseries(
            df,
            target_column=target_column,
            dataset_context=dataset_context,
            progress_callback=progress_callback,
        )
    return _train_tabular_or_text(
        df,
        target_column=target_column,
        task_type=task_type,
        dataset_type=dataset_type,
        dataset_context=dataset_context,
        progress_callback=progress_callback,
    )
