from __future__ import annotations

import json
from pathlib import Path
import pickle
import site
import sys

USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.config import BUILTIN_MODELS_DIR, ensure_directories


def write_manifest(include_paths: bool = True) -> None:
    manifest = {
        "models": [
            {
                "name": "fast-classifier",
                "label": "Fast Classifier",
                "path": str(BUILTIN_MODELS_DIR / "fast_classifier.pkl") if include_paths else "",
            },
            {
                "name": "accurate-classifier",
                "label": "Accurate Classifier",
                "path": str(BUILTIN_MODELS_DIR / "accurate_classifier.pkl") if include_paths else "",
            },
            {
                "name": "balanced-classifier",
                "label": "Balanced Classifier",
                "path": str(BUILTIN_MODELS_DIR / "balanced_classifier.pkl") if include_paths else "",
            },
        ]
    }
    with (BUILTIN_MODELS_DIR / "models_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def train_builtin_models() -> None:
    ensure_directories()
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError:
        write_manifest(include_paths=False)
        return

    data = load_breast_cancer()
    X, y = data.data, data.target

    models = {
        "fast_classifier.pkl": LogisticRegression(max_iter=1000).fit(X, y),
        "accurate_classifier.pkl": RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y),
        "balanced_classifier.pkl": ExtraTreesClassifier(n_estimators=200, random_state=42).fit(X, y),
    }
    for filename, model in models.items():
        with (BUILTIN_MODELS_DIR / filename).open("wb") as handle:
            pickle.dump(model, handle)
    write_manifest(include_paths=True)


if __name__ == "__main__":
    train_builtin_models()
    print("Built-in models are ready.")
