from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder

from ..utils.metrics import classification_metrics, primary_metric_name, regression_metrics


def _optional_module(name: str) -> Any | None:
    try:
        return __import__(name)
    except Exception:
        return None


def _sorted_lag_columns(frame: pd.DataFrame) -> list[str]:
    lag_columns = [column for column in frame.columns if "_lag_" in column]
    return sorted(
        lag_columns,
        key=lambda value: int(str(value).rsplit("_lag_", 1)[-1]),
        reverse=True,
    )


def _score_result(task_type: str, metrics: dict[str, float]) -> float:
    return float(metrics[primary_metric_name(task_type)])


@dataclass
class StatsForecastModel:
    model: Any
    time_column: str
    framework: str
    learning_curve_: list[dict[str, Any]] = field(default_factory=list)
    feature_importance_summary: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        horizon = max(len(X), 1)
        values = self.model.forecast(steps=horizon)
        return np.asarray(values, dtype=np.float32).reshape(-1)


@dataclass
class ProphetForecastModel:
    model: Any
    time_column: str
    framework: str = "prophet"
    learning_curve_: list[dict[str, Any]] = field(default_factory=list)
    feature_importance_summary: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        forecast_frame = pd.DataFrame({"ds": pd.to_datetime(X[self.time_column], errors="coerce")})
        forecast = self.model.predict(forecast_frame)
        return forecast["yhat"].to_numpy(dtype=np.float32)


@dataclass
class TensorFlowLagSequenceModel:
    model: Any
    lag_columns: list[str]
    framework: str = "tensorflow"
    learning_curve_: list[dict[str, Any]] = field(default_factory=list)
    feature_importance_summary: list[dict[str, Any]] = field(default_factory=list)

    def _reshape(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.lag_columns].to_numpy(dtype=np.float32)
        return values.reshape(len(values), len(self.lag_columns), 1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        outputs = self.model.predict(self._reshape(X), verbose=0)
        return np.asarray(outputs, dtype=np.float32).reshape(-1)


@dataclass
class TransformerEmbeddingModel:
    model_id: str
    model_label: str
    task_type: str
    text_column: str
    head_model: Any
    label_encoder: LabelEncoder | None = None
    framework: str = "transformers"
    learning_curve_: list[dict[str, Any]] = field(default_factory=list)
    feature_importance_summary: list[dict[str, Any]] = field(default_factory=list)
    max_length: int = 256
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _backbone: Any | None = field(default=None, init=False, repr=False)
    _torch: Any | None = field(default=None, init=False, repr=False)

    def _load_backbone(self) -> None:
        if self._tokenizer is not None and self._backbone is not None and self._torch is not None:
            return
        transformers = _optional_module("transformers")
        torch_module = _optional_module("torch")
        if transformers is None or torch_module is None:
            raise RuntimeError(f"{self.model_label} requires transformers and torch")
        self._torch = torch_module
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
        self._backbone = transformers.AutoModel.from_pretrained(self.model_id, local_files_only=True)
        self._backbone.eval()

    def _embed_text(self, X: pd.DataFrame) -> np.ndarray:
        self._load_backbone()
        assert self._tokenizer is not None
        assert self._backbone is not None
        assert self._torch is not None
        texts = X[self.text_column].fillna("").astype(str).tolist()
        batches: list[np.ndarray] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                outputs = self._backbone(**encoded)
            hidden_state = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden_state * attention_mask
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            batches.append(pooled.cpu().numpy())
        return np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        embeddings = self._embed_text(X)
        predictions = self.head_model.predict(embeddings)
        if self.task_type == "classification" and self.label_encoder is not None:
            return self.label_encoder.inverse_transform(np.asarray(predictions, dtype=int))
        return np.asarray(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.head_model, "predict_proba"):
            raise AttributeError("Transformer embedding head does not expose predict_proba")
        embeddings = self._embed_text(X)
        return self.head_model.predict_proba(embeddings)


def _evaluate_timeseries_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = np.asarray(model.predict(X)).reshape(-1)
    return regression_metrics(y, predictions)


def _evaluate_text_model(task_type: str, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    if task_type == "classification":
        probabilities = None
        if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
            probabilities = model.predict_proba(X)[:, 1]
        return classification_metrics(y, predictions, probabilities)
    return regression_metrics(y, predictions)


def _statsmodels_arima_results(
    *,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    X_val_display: pd.DataFrame,
    X_test_display: pd.DataFrame,
    time_column: str,
) -> list[dict[str, Any]]:
    statsmodels = _optional_module("statsmodels")
    if statsmodels is None:
        return []

    from statsmodels.tsa.arima.model import ARIMA

    candidate_orders = [(1, 0, 0), (2, 1, 0), (1, 1, 1), (2, 1, 2)]
    history = []
    best_fit = None
    best_order = None
    best_validation = None

    for order in candidate_orders:
        try:
            model = ARIMA(y_train.astype(float), order=order).fit()
            validation_predictions = np.asarray(model.forecast(steps=len(y_val)), dtype=np.float32).reshape(-1)
            validation_metrics = regression_metrics(y_val, validation_predictions)
            history.append({"candidate": str(order), "validation_score": _score_result("regression", validation_metrics)})
            if best_validation is None or _score_result("regression", validation_metrics) > _score_result("regression", best_validation):
                best_validation = validation_metrics
                best_fit = model
                best_order = order
        except Exception:
            continue

    if best_order is None or best_validation is None:
        return []

    started_at = time.perf_counter()
    final_model = ARIMA(pd.concat([y_train, y_val]).astype(float), order=best_order).fit()
    wrapper = StatsForecastModel(
        model=final_model,
        time_column=time_column,
        framework="statsmodels",
        learning_curve_=[
            {"epoch": index + 1, "validation_score": round(float(item["validation_score"]), 4), "candidate": item["candidate"]}
            for index, item in enumerate(history)
        ],
    )
    test_predictions = wrapper.predict(X_test_display)
    elapsed = round(time.perf_counter() - started_at, 3)
    test_metrics = regression_metrics(y_test, test_predictions)

    return [
        {
            "name": "ARIMA",
            "family": "timeseries_classical",
            "search": None,
            "pipeline": wrapper,
            "cv_score_mean": round(_score_result("regression", best_validation), 4),
            "cv_score_std": 0.0,
            "validation_metrics": best_validation,
            "test_metrics": test_metrics,
            "training_time_seconds": elapsed,
            "params": {"framework": "statsmodels", "order": list(best_order), "time_column": time_column},
            "evaluation": {
                "X_test": X_test_display.copy(),
                "y_test": y_test.copy(),
                "predictions": np.asarray(test_predictions),
                "probabilities": [],
            },
        }
    ]


def _statsmodels_exponential_smoothing_results(
    *,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    X_val_display: pd.DataFrame,
    X_test_display: pd.DataFrame,
    time_column: str,
) -> list[dict[str, Any]]:
    statsmodels = _optional_module("statsmodels")
    if statsmodels is None:
        return []

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal_period = 7 if len(y_train) >= 21 else 4 if len(y_train) >= 12 else None
    candidate_specs = [
        {"trend": "add", "seasonal": "add" if seasonal_period else None, "seasonal_periods": seasonal_period},
        {"trend": "add", "seasonal": None, "seasonal_periods": None},
        {"trend": None, "seasonal": None, "seasonal_periods": None},
    ]
    history = []
    best_validation = None
    best_spec = None

    for spec in candidate_specs:
        try:
            model = ExponentialSmoothing(
                y_train.astype(float),
                trend=spec["trend"],
                seasonal=spec["seasonal"],
                seasonal_periods=spec["seasonal_periods"],
            ).fit(optimized=True)
            validation_predictions = np.asarray(model.forecast(steps=len(y_val)), dtype=np.float32).reshape(-1)
            validation_metrics = regression_metrics(y_val, validation_predictions)
            history.append(
                {
                    "candidate": f"trend={spec['trend']}, seasonal={spec['seasonal']}, period={spec['seasonal_periods']}",
                    "validation_score": _score_result("regression", validation_metrics),
                }
            )
            if best_validation is None or _score_result("regression", validation_metrics) > _score_result("regression", best_validation):
                best_validation = validation_metrics
                best_spec = spec
        except Exception:
            continue

    if best_spec is None or best_validation is None:
        return []

    started_at = time.perf_counter()
    final_model = ExponentialSmoothing(
        pd.concat([y_train, y_val]).astype(float),
        trend=best_spec["trend"],
        seasonal=best_spec["seasonal"],
        seasonal_periods=best_spec["seasonal_periods"],
    ).fit(optimized=True)
    wrapper = StatsForecastModel(
        model=final_model,
        time_column=time_column,
        framework="statsmodels",
        learning_curve_=[
            {"epoch": index + 1, "validation_score": round(float(item["validation_score"]), 4), "candidate": item["candidate"]}
            for index, item in enumerate(history)
        ],
    )
    test_predictions = wrapper.predict(X_test_display)
    elapsed = round(time.perf_counter() - started_at, 3)
    test_metrics = regression_metrics(y_test, test_predictions)

    return [
        {
            "name": "Exponential Smoothing",
            "family": "timeseries_classical",
            "search": None,
            "pipeline": wrapper,
            "cv_score_mean": round(_score_result("regression", best_validation), 4),
            "cv_score_std": 0.0,
            "validation_metrics": best_validation,
            "test_metrics": test_metrics,
            "training_time_seconds": elapsed,
            "params": {
                "framework": "statsmodels",
                "trend": best_spec["trend"],
                "seasonal": best_spec["seasonal"],
                "seasonal_periods": best_spec["seasonal_periods"],
                "time_column": time_column,
            },
            "evaluation": {
                "X_test": X_test_display.copy(),
                "y_test": y_test.copy(),
                "predictions": np.asarray(test_predictions),
                "probabilities": [],
            },
        }
    ]


def _prophet_results(
    *,
    X_train_display: pd.DataFrame,
    X_val_display: pd.DataFrame,
    X_test_display: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    time_column: str,
) -> list[dict[str, Any]]:
    prophet_module = _optional_module("prophet")
    if prophet_module is None:
        return []

    from prophet import Prophet

    candidate_specs = [
        {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
        {"changepoint_prior_scale": 0.2, "seasonality_mode": "multiplicative"},
    ]
    history = []
    best_validation = None
    best_spec = None

    train_frame = pd.DataFrame({"ds": pd.to_datetime(X_train_display[time_column]), "y": y_train.to_numpy(dtype=np.float32)})
    validation_frame = pd.DataFrame({"ds": pd.to_datetime(X_val_display[time_column])})

    for spec in candidate_specs:
        try:
            model = Prophet(**spec)
            model.fit(train_frame)
            validation_predictions = model.predict(validation_frame)["yhat"].to_numpy(dtype=np.float32)
            validation_metrics = regression_metrics(y_val, validation_predictions)
            history.append(
                {
                    "candidate": f"changepoint_prior_scale={spec['changepoint_prior_scale']}, seasonality_mode={spec['seasonality_mode']}",
                    "validation_score": _score_result("regression", validation_metrics),
                }
            )
            if best_validation is None or _score_result("regression", validation_metrics) > _score_result("regression", best_validation):
                best_validation = validation_metrics
                best_spec = spec
        except Exception:
            continue

    if best_spec is None or best_validation is None:
        return []

    started_at = time.perf_counter()
    final_frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(pd.concat([X_train_display[time_column], X_val_display[time_column]], axis=0)),
            "y": pd.concat([y_train, y_val], axis=0).to_numpy(dtype=np.float32),
        }
    )
    final_model = Prophet(**best_spec)
    final_model.fit(final_frame)
    wrapper = ProphetForecastModel(
        model=final_model,
        time_column=time_column,
        learning_curve_=[
            {"epoch": index + 1, "validation_score": round(float(item["validation_score"]), 4), "candidate": item["candidate"]}
            for index, item in enumerate(history)
        ],
    )
    test_predictions = wrapper.predict(X_test_display)
    elapsed = round(time.perf_counter() - started_at, 3)
    test_metrics = regression_metrics(y_test, test_predictions)

    return [
        {
            "name": "Prophet",
            "family": "timeseries_classical",
            "search": None,
            "pipeline": wrapper,
            "cv_score_mean": round(_score_result("regression", best_validation), 4),
            "cv_score_std": 0.0,
            "validation_metrics": best_validation,
            "test_metrics": test_metrics,
            "training_time_seconds": elapsed,
            "params": {
                "framework": "prophet",
                "time_column": time_column,
                **best_spec,
            },
            "evaluation": {
                "X_test": X_test_display.copy(),
                "y_test": y_test.copy(),
                "predictions": np.asarray(test_predictions),
                "probabilities": [],
            },
        }
    ]


def _tensorflow_lstm_results(
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
) -> list[dict[str, Any]]:
    tensorflow = _optional_module("tensorflow")
    if tensorflow is None:
        return []

    lag_columns = _sorted_lag_columns(X_train)
    if len(lag_columns) < 2:
        return []

    tf = tensorflow
    keras = tf.keras

    def reshape(frame: pd.DataFrame) -> np.ndarray:
        values = frame[lag_columns].to_numpy(dtype=np.float32)
        return values.reshape(len(values), len(lag_columns), 1)

    X_train_seq = reshape(X_train)
    X_val_seq = reshape(X_val)
    X_test_seq = reshape(X_test)

    specs = [
        {"name": "LSTM", "units": [64], "dropout": 0.1},
        {"name": "Stacked LSTM", "units": [96, 48], "dropout": 0.15},
    ]
    results: list[dict[str, Any]] = []

    for spec in specs:
        started_at = time.perf_counter()
        layers: list[Any] = [keras.layers.Input(shape=(len(lag_columns), 1))]
        for index, units in enumerate(spec["units"]):
            layers.append(
                keras.layers.LSTM(
                    units,
                    return_sequences=index < len(spec["units"]) - 1,
                )
            )
            if spec["dropout"] > 0:
                layers.append(keras.layers.Dropout(spec["dropout"]))
        layers.append(keras.layers.Dense(1))
        model = keras.Sequential(layers)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        history = model.fit(
            X_train_seq,
            y_train.to_numpy(dtype=np.float32),
            validation_data=(X_val_seq, y_val.to_numpy(dtype=np.float32)),
            epochs=24,
            batch_size=min(64, max(8, len(X_train_seq) // 8 or 8)),
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        )

        validation_wrapper = TensorFlowLagSequenceModel(
            model=model,
            lag_columns=lag_columns,
            learning_curve_=[
                {"epoch": index + 1, "train_loss": round(float(loss), 4), "validation_loss": round(float(val_loss), 4)}
                for index, (loss, val_loss) in enumerate(zip(history.history.get("loss", []), history.history.get("val_loss", [])))
            ],
        )
        validation_metrics = regression_metrics(y_val, validation_wrapper.predict(X_val))

        final_model = keras.Sequential(layers)
        final_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        final_model.fit(
            reshape(train_val_X),
            train_val_y.to_numpy(dtype=np.float32),
            validation_split=0.1 if len(train_val_X) > 24 else 0.0,
            epochs=24,
            batch_size=min(64, max(8, len(train_val_X) // 8 or 8)),
            verbose=0,
        )
        final_wrapper = TensorFlowLagSequenceModel(
            model=final_model,
            lag_columns=lag_columns,
            learning_curve_=validation_wrapper.learning_curve_,
            feature_importance_summary=[
                {"feature": column, "importance": round(float(abs(train_val_X[column].corr(train_val_y))), 6)}
                for column in lag_columns[: min(len(lag_columns), 12)]
                if not np.isnan(train_val_X[column].corr(train_val_y))
            ],
        )
        test_predictions = final_wrapper.predict(X_test)
        elapsed = round(time.perf_counter() - started_at, 3)
        test_metrics = regression_metrics(y_test, test_predictions)

        results.append(
            {
                "name": f"{spec['name']} Forecaster",
                "family": "timeseries_deep_learning",
                "search": None,
                "pipeline": final_wrapper,
                "cv_score_mean": round(_score_result("regression", validation_metrics), 4),
                "cv_score_std": 0.0,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_time_seconds": elapsed,
                "params": {
                    "framework": "tensorflow",
                    "sequence_model": "lstm",
                    "lag_columns": lag_columns,
                    "units": spec["units"],
                    "dropout": spec["dropout"],
                },
                "evaluation": {
                    "X_test": X_test.copy(),
                    "y_test": y_test.copy(),
                    "predictions": np.asarray(test_predictions),
                    "probabilities": [],
                },
            }
        )
    return results


def train_optional_timeseries_specialists(
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train_display: pd.DataFrame,
    X_val_display: pd.DataFrame,
    X_test_display: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
    time_column: str,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if progress_callback:
        progress_callback(
            "Trying advanced time-series specialists",
            {
                "current_model": "ARIMA / Prophet / Exponential Smoothing / LSTM",
                "training_strategy": "Classical forecasting specialists plus recurrent deep learning",
            },
        )
    results.extend(
        _statsmodels_arima_results(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            X_val_display=X_val_display,
            X_test_display=X_test_display,
            time_column=time_column,
        )
    )
    results.extend(
        _statsmodels_exponential_smoothing_results(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            X_val_display=X_val_display,
            X_test_display=X_test_display,
            time_column=time_column,
        )
    )
    results.extend(
        _prophet_results(
            X_train_display=X_train_display,
            X_val_display=X_val_display,
            X_test_display=X_test_display,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            time_column=time_column,
        )
    )
    lstm_results = _tensorflow_lstm_results(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_val_X=train_val_X,
        train_val_y=train_val_y,
    )
    for item in lstm_results:
        item["evaluation"]["X_test"] = X_test_display.copy()
    results.extend(lstm_results)
    return results


def _transformer_embedding_results(
    *,
    model_label: str,
    model_id: str,
    task_type: str,
    text_column: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
) -> list[dict[str, Any]]:
    transformers = _optional_module("transformers")
    torch_module = _optional_module("torch")
    if transformers is None or torch_module is None:
        return []

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        backbone = transformers.AutoModel.from_pretrained(model_id, local_files_only=True)
        backbone.eval()
    except Exception:
        return []

    def embed(frame: pd.DataFrame) -> np.ndarray:
        texts = frame[text_column].fillna("").astype(str).tolist()
        vectors: list[np.ndarray] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch_module.no_grad():
                outputs = backbone(**encoded)
            hidden_state = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)

    started_at = time.perf_counter()
    X_train_embeddings = embed(X_train)
    X_val_embeddings = embed(X_val)
    X_test_embeddings = embed(X_test)
    X_train_val_embeddings = embed(train_val_X)

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        train_val_encoded = label_encoder.fit_transform(train_val_y)
        head = LogisticRegression(max_iter=1000, class_weight="balanced")
        param_values = [0.5, 1.0, 2.0]
    else:
        label_encoder = None
        y_train_encoded = y_train.to_numpy(dtype=np.float32)
        y_val_encoded = y_val.to_numpy(dtype=np.float32)
        train_val_encoded = train_val_y.to_numpy(dtype=np.float32)
        head = Ridge()
        param_values = [0.5, 1.0, 5.0]

    best_validation = None
    best_value = None
    best_head = None

    for value in param_values:
        current_head = clone(head)
        if task_type == "classification":
            current_head.set_params(C=value)
        else:
            current_head.set_params(alpha=value)
        current_head.fit(X_train_embeddings, y_train_encoded)

        temp_wrapper = TransformerEmbeddingModel(
            model_id=model_id,
            model_label=model_label,
            task_type=task_type,
            text_column=text_column,
            head_model=current_head,
            label_encoder=label_encoder,
        )
        validation_metrics = _evaluate_text_model(task_type, temp_wrapper, X_val, y_val)
        if best_validation is None or _score_result(task_type, validation_metrics) > _score_result(task_type, best_validation):
            best_validation = validation_metrics
            best_value = value
            best_head = current_head

    if best_validation is None or best_value is None or best_head is None:
        return []

    final_head = clone(head)
    if task_type == "classification":
        final_head.set_params(C=best_value)
    else:
        final_head.set_params(alpha=best_value)
    final_head.fit(X_train_val_embeddings, train_val_encoded)
    final_wrapper = TransformerEmbeddingModel(
        model_id=model_id,
        model_label=model_label,
        task_type=task_type,
        text_column=text_column,
        head_model=final_head,
        label_encoder=label_encoder if task_type == "classification" else None,
        learning_curve_=[
            {"epoch": index + 1, "validation_score": round(float(score), 4)}
            for index, score in enumerate(
                [_score_result(task_type, best_validation)]
            )
        ],
    )
    test_metrics = _evaluate_text_model(task_type, final_wrapper, X_test, y_test)
    elapsed = round(time.perf_counter() - started_at, 3)

    predictions = final_wrapper.predict(X_test)
    probabilities: list[float] = []
    if task_type == "classification" and len(np.unique(y_test)) == 2:
        probabilities = final_wrapper.predict_proba(X_test)[:, 1].tolist()

    return [
        {
            "name": f"{model_label} Embeddings + {'Logistic Regression' if task_type == 'classification' else 'Ridge'}",
            "family": "transformer_text",
            "search": None,
            "pipeline": final_wrapper,
            "cv_score_mean": round(_score_result(task_type, best_validation), 4),
            "cv_score_std": 0.0,
            "validation_metrics": best_validation,
            "test_metrics": test_metrics,
            "training_time_seconds": elapsed,
            "params": {
                "framework": "transformers",
                "model_id": model_id,
                "text_column": text_column,
                "head_regularization": best_value,
            },
            "evaluation": {
                "X_test": X_test.copy(),
                "y_test": y_test.copy(),
                "predictions": np.asarray(predictions),
                "probabilities": probabilities,
            },
        }
    ]


def train_optional_text_transformers(
    *,
    task_type: str,
    text_column: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    if progress_callback:
        progress_callback(
            "Trying transformer text encoders",
            {
                "current_model": "DistilBERT / BERT",
                "training_strategy": "Frozen transformer embeddings plus tuned linear head",
            },
        )

    results: list[dict[str, Any]] = []
    results.extend(
        _transformer_embedding_results(
            model_label="DistilBERT",
            model_id="distilbert-base-uncased",
            task_type=task_type,
            text_column=text_column,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            train_val_X=train_val_X,
            train_val_y=train_val_y,
        )
    )
    results.extend(
        _transformer_embedding_results(
            model_label="BERT",
            model_id="bert-base-uncased",
            task_type=task_type,
            text_column=text_column,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            train_val_X=train_val_X,
            train_val_y=train_val_y,
        )
    )
    return results
