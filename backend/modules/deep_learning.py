from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from ..utils.metrics import classification_metrics, primary_metric_name, regression_metrics


def _dense_matrix(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _optional_module(name: str) -> Any | None:
    try:
        return __import__(name)
    except Exception:
        return None


def _architecture_specs(dataset_type: str) -> list[dict[str, Any]]:
    if dataset_type == "text":
        return [
            {"suffix": "Wide ReLU", "hidden_layers": [512, 256], "dropout": 0.15, "activation": "relu"},
            {"suffix": "Deep GELU", "hidden_layers": [768, 512, 256, 128], "dropout": 0.2, "activation": "gelu"},
        ]
    if dataset_type == "timeseries":
        return [
            {"suffix": "Temporal Dense", "hidden_layers": [256, 128, 64], "dropout": 0.1, "activation": "relu"},
            {"suffix": "Deep Forecast", "hidden_layers": [384, 256, 128, 64], "dropout": 0.15, "activation": "gelu"},
        ]
    if dataset_type in {"image", "audio"}:
        return [
            {"suffix": "Feature Head", "hidden_layers": [256, 128], "dropout": 0.15, "activation": "relu"},
            {"suffix": "Deep Feature Head", "hidden_layers": [512, 256, 128], "dropout": 0.2, "activation": "gelu"},
        ]
    return [
        {"suffix": "Wide ReLU", "hidden_layers": [256, 128], "dropout": 0.15, "activation": "relu"},
        {"suffix": "Deep GELU", "hidden_layers": [384, 256, 128, 64], "dropout": 0.2, "activation": "gelu"},
    ]


@dataclass
class TensorFlowFeatureModel:
    preprocessor: Any
    model: Any
    task_type: str
    label_encoder: LabelEncoder | None = None
    learning_curve_: list[dict[str, Any]] = field(default_factory=list)
    feature_importance_summary: list[dict[str, Any]] = field(default_factory=list)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return _dense_matrix(self.preprocessor.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        outputs = self.model.predict(self._transform(X), verbose=0)
        if self.task_type == "classification":
            if outputs.ndim == 2 and outputs.shape[1] > 1:
                indices = outputs.argmax(axis=1)
            else:
                indices = (outputs.reshape(-1) >= 0.5).astype(int)
            return self.label_encoder.inverse_transform(indices) if self.label_encoder else indices
        return outputs.reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        outputs = self.model.predict(self._transform(X), verbose=0)
        if outputs.ndim == 1:
            outputs = outputs.reshape(-1, 1)
        if outputs.shape[1] == 1:
            return np.concatenate([1.0 - outputs, outputs], axis=1)
        return outputs


@dataclass
class TorchFeatureModel:
    preprocessor: Any
    model: Any
    torch_module: Any
    task_type: str
    label_encoder: LabelEncoder | None = None
    learning_curve_: list[dict[str, Any]] = field(default_factory=list)
    feature_importance_summary: list[dict[str, Any]] = field(default_factory=list)

    def _transform(self, X: pd.DataFrame) -> Any:
        return self.torch_module.tensor(_dense_matrix(self.preprocessor.transform(X)), dtype=self.torch_module.float32)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self.torch_module.no_grad():
            outputs = self.model(self._transform(X))
        if self.task_type == "classification":
            if outputs.shape[1] == 1:
                probs = self.torch_module.sigmoid(outputs).cpu().numpy().reshape(-1)
                indices = (probs >= 0.5).astype(int)
            else:
                probs = self.torch_module.softmax(outputs, dim=1).cpu().numpy()
                indices = probs.argmax(axis=1)
            return self.label_encoder.inverse_transform(indices) if self.label_encoder else indices
        return outputs.cpu().numpy().reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self.torch_module.no_grad():
            outputs = self.model(self._transform(X))
        if outputs.shape[1] == 1:
            probs = self.torch_module.sigmoid(outputs).cpu().numpy().reshape(-1, 1)
            return np.concatenate([1.0 - probs, probs], axis=1)
        return self.torch_module.softmax(outputs, dim=1).cpu().numpy()


def _evaluate_model(task_type: str, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    if task_type == "classification":
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") and len(np.unique(y)) == 2 else None
        return classification_metrics(y, predictions, probabilities)
    return regression_metrics(y, predictions)


def _tensorflow_results(
    *,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
    task_type: str,
    dataset_type: str,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    tensorflow = _optional_module("tensorflow")
    if tensorflow is None:
        return []

    tf = tensorflow
    keras = tf.keras
    fit_preprocessor = clone(preprocessor)
    X_train_array = _dense_matrix(fit_preprocessor.fit_transform(X_train))
    X_val_array = _dense_matrix(fit_preprocessor.transform(X_val))
    X_test_array = _dense_matrix(fit_preprocessor.transform(X_test))

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        class_count = len(label_encoder.classes_)
        output_units = 1 if class_count <= 2 else class_count
        output_activation = "sigmoid" if class_count <= 2 else "softmax"
        loss_name = "binary_crossentropy" if class_count <= 2 else "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
        final_y_train = y_train_encoded
        final_y_val = y_val_encoded
    else:
        label_encoder = None
        output_units = 1
        output_activation = "linear"
        loss_name = "mse"
        metrics = ["mae"]
        final_y_train = y_train.to_numpy(dtype=np.float32)
        final_y_val = y_val.to_numpy(dtype=np.float32)

    def build_model(input_dim: int, spec: dict[str, Any]) -> Any:
        layers = [keras.layers.Input(shape=(input_dim,))]
        for width in spec["hidden_layers"]:
            layers.append(keras.layers.Dense(width, activation=spec["activation"]))
            layers.append(keras.layers.BatchNormalization())
            if spec["dropout"] > 0:
                layers.append(keras.layers.Dropout(spec["dropout"]))
        layers.append(keras.layers.Dense(output_units, activation=output_activation))
        return keras.Sequential(layers)

    results: list[dict[str, Any]] = []
    specs = _architecture_specs(dataset_type)
    for index, spec in enumerate(specs, start=1):
        if progress_callback:
            progress_callback(
                f"Trying TensorFlow / Keras network: {spec['suffix']}",
                {
                    "current_model": f"TensorFlow / Keras Dense Network ({dataset_type})",
                    "architecture": spec["hidden_layers"],
                    "training_strategy": f"{spec['activation']} activation, dropout {spec['dropout']}",
                    "deep_model_index": index,
                    "deep_model_total": len(specs),
                },
            )
        started_at = time.perf_counter()
        model = build_model(X_train_array.shape[1], spec)
        model.compile(optimizer="adam", loss=loss_name, metrics=metrics)
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        history = model.fit(
            X_train_array,
            final_y_train,
            validation_data=(X_val_array, final_y_val),
            epochs=22,
            batch_size=min(64, max(8, len(X_train_array) // 8 or 8)),
            verbose=0,
            callbacks=callbacks,
        )

        validation_wrapper = TensorFlowFeatureModel(
            preprocessor=fit_preprocessor,
            model=model,
            task_type=task_type,
            label_encoder=label_encoder,
            learning_curve_=[
                {"epoch": index + 1, "train_loss": round(float(loss), 4), "validation_loss": round(float(val_loss), 4)}
                for index, (loss, val_loss) in enumerate(zip(history.history.get("loss", []), history.history.get("val_loss", [])))
            ],
        )
        validation_metrics = _evaluate_model(task_type, validation_wrapper, X_val, y_val)

        final_preprocessor = clone(preprocessor)
        X_train_val_array = _dense_matrix(final_preprocessor.fit_transform(train_val_X))
        final_model = build_model(X_train_val_array.shape[1], spec)
        final_model.compile(optimizer="adam", loss=loss_name, metrics=metrics)
        if task_type == "classification":
            final_targets = label_encoder.fit_transform(train_val_y)
        else:
            final_targets = train_val_y.to_numpy(dtype=np.float32)
        final_model.fit(
            X_train_val_array,
            final_targets,
            validation_split=0.1 if len(X_train_val_array) > 24 else 0.0,
            epochs=22,
            batch_size=min(64, max(8, len(X_train_val_array) // 8 or 8)),
            verbose=0,
        )
        final_wrapper = TensorFlowFeatureModel(
            preprocessor=final_preprocessor,
            model=final_model,
            task_type=task_type,
            label_encoder=label_encoder if task_type == "classification" else None,
            learning_curve_=validation_wrapper.learning_curve_,
        )
        test_metrics = _evaluate_model(task_type, final_wrapper, X_test, y_test)
        elapsed = round(time.perf_counter() - started_at, 3)

        results.append(
            {
                "name": f"TensorFlow / Keras Dense Network ({dataset_type} | {spec['suffix']})",
                "family": "deep_learning",
                "search": None,
                "pipeline": final_wrapper,
                "cv_score_mean": round(float(validation_metrics[primary_metric_name(task_type)]), 4),
                "cv_score_std": 0.0,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_time_seconds": elapsed,
                "params": {
                    "framework": "tensorflow",
                    "epochs": 22,
                    "hidden_layers": spec["hidden_layers"],
                    "activation": spec["activation"],
                    "dropout": spec["dropout"],
                },
                "evaluation": {
                    "X_test": X_test.copy(),
                    "y_test": y_test.copy(),
                    "predictions": final_wrapper.predict(X_test),
                    "probabilities": final_wrapper.predict_proba(X_test)[:, 1].tolist()
                    if task_type == "classification" and len(np.unique(y_test)) == 2
                    else [],
                },
            }
        )

    return results


def _pytorch_results(
    *,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
    task_type: str,
    dataset_type: str,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    torch = _optional_module("torch")
    if torch is None:
        return []

    fit_preprocessor = clone(preprocessor)
    X_train_array = _dense_matrix(fit_preprocessor.fit_transform(X_train))
    X_val_array = _dense_matrix(fit_preprocessor.transform(X_val))

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        class_count = len(label_encoder.classes_)
    else:
        label_encoder = None
        y_train_encoded = y_train.to_numpy(dtype=np.float32)
        y_val_encoded = y_val.to_numpy(dtype=np.float32)
        class_count = 1

    class DenseNet(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int, spec: dict[str, Any]) -> None:
            super().__init__()
            modules: list[Any] = []
            current_dim = input_dim
            activation_layer = torch.nn.GELU if spec["activation"] == "gelu" else torch.nn.ReLU
            for width in spec["hidden_layers"]:
                modules.extend(
                    [
                        torch.nn.Linear(current_dim, width),
                        activation_layer(),
                        torch.nn.BatchNorm1d(width),
                    ]
                )
                if spec["dropout"] > 0:
                    modules.append(torch.nn.Dropout(spec["dropout"]))
                current_dim = width
            modules.append(torch.nn.Linear(current_dim, output_dim))
            self.layers = torch.nn.Sequential(*modules)

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    def train_once(features: np.ndarray, targets: np.ndarray, output_dim: int, spec: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
        model = DenseNet(features.shape[1], output_dim, spec)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if task_type == "classification":
            criterion = torch.nn.BCEWithLogitsLoss() if output_dim == 1 else torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

        x_tensor = torch.tensor(features, dtype=torch.float32)
        if task_type == "classification":
            if output_dim == 1:
                y_tensor = torch.tensor(targets.reshape(-1, 1), dtype=torch.float32)
            else:
                y_tensor = torch.tensor(targets, dtype=torch.long)
        else:
            y_tensor = torch.tensor(targets.reshape(-1, 1), dtype=torch.float32)

        curve = []
        model.train()
        for epoch in range(18):
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            curve.append({"epoch": epoch + 1, "train_loss": round(float(loss.item()), 4)})
        return model, curve

    results: list[dict[str, Any]] = []
    specs = _architecture_specs(dataset_type)
    for index, spec in enumerate(specs, start=1):
        if progress_callback:
            progress_callback(
                f"Trying PyTorch network: {spec['suffix']}",
                {
                    "current_model": f"PyTorch Dense Network ({dataset_type})",
                    "architecture": spec["hidden_layers"],
                    "training_strategy": f"{spec['activation']} activation, dropout {spec['dropout']}",
                    "deep_model_index": index,
                    "deep_model_total": len(specs),
                },
            )
        started_at = time.perf_counter()
        validation_model, curve = train_once(X_train_array, y_train_encoded, 1 if class_count <= 2 else class_count, spec)
        validation_wrapper = TorchFeatureModel(
            preprocessor=fit_preprocessor,
            model=validation_model,
            torch_module=torch,
            task_type=task_type,
            label_encoder=label_encoder,
            learning_curve_=curve,
        )
        validation_metrics = _evaluate_model(task_type, validation_wrapper, X_val, y_val)

        final_preprocessor = clone(preprocessor)
        X_train_val_array = _dense_matrix(final_preprocessor.fit_transform(train_val_X))
        final_targets = (
            label_encoder.fit_transform(train_val_y)
            if task_type == "classification"
            else train_val_y.to_numpy(dtype=np.float32)
        )
        final_model, final_curve = train_once(
            X_train_val_array,
            final_targets,
            1 if task_type == "regression" or len(np.unique(final_targets)) <= 2 else len(np.unique(final_targets)),
            spec,
        )
        final_wrapper = TorchFeatureModel(
            preprocessor=final_preprocessor,
            model=final_model,
            torch_module=torch,
            task_type=task_type,
            label_encoder=label_encoder if task_type == "classification" else None,
            learning_curve_=final_curve,
        )
        test_metrics = _evaluate_model(task_type, final_wrapper, X_test, y_test)
        elapsed = round(time.perf_counter() - started_at, 3)

        results.append(
            {
                "name": f"PyTorch Dense Network ({dataset_type} | {spec['suffix']})",
                "family": "deep_learning",
                "search": None,
                "pipeline": final_wrapper,
                "cv_score_mean": round(float(validation_metrics[primary_metric_name(task_type)]), 4),
                "cv_score_std": 0.0,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_time_seconds": elapsed,
                "params": {
                    "framework": "pytorch",
                    "epochs": 18,
                    "hidden_layers": spec["hidden_layers"],
                    "activation": spec["activation"],
                    "dropout": spec["dropout"],
                },
                "evaluation": {
                    "X_test": X_test.copy(),
                    "y_test": y_test.copy(),
                    "predictions": final_wrapper.predict(X_test),
                    "probabilities": final_wrapper.predict_proba(X_test)[:, 1].tolist()
                    if task_type == "classification" and len(np.unique(y_test)) == 2
                    else [],
                },
            }
        )

    return results


def train_optional_deep_models(
    *,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    train_val_X: pd.DataFrame,
    train_val_y: pd.Series,
    task_type: str,
    dataset_type: str,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    results.extend(
        _tensorflow_results(
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
            progress_callback=progress_callback,
        )
    )
    results.extend(
        _pytorch_results(
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
            progress_callback=progress_callback,
        )
    )
    return results
