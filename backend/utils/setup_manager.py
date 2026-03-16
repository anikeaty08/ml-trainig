from __future__ import annotations

import json
import os
import platform
import shutil
import site
import sys
from pathlib import Path
from typing import Any

from ..config import BUILTIN_MODELS_DIR, DATA_DIR, ROOT_DIR, ensure_directories


PACKS_DIR = BUILTIN_MODELS_DIR / "packs"
SETUP_STATE_PATH = DATA_DIR / "setup_state.json"


PACK_DEFINITIONS = [
    {
        "id": "core-tabular",
        "label": "Core Tabular",
        "size_mb": 220,
        "description": "Classical tabular classification and regression baselines with local reports.",
        "models": [
            "Logistic Regression",
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "Histogram Gradient Boosting",
            "SVM",
            "KNN",
            "Naive Bayes",
            "Voting Ensemble",
        ],
        "features": ["tabular", "reports", "code export"],
    },
    {
        "id": "boosting-pack",
        "label": "Boosting Pack",
        "size_mb": 420,
        "description": "XGBoost and LightGBM model family support.",
        "models": ["XGBoost", "XGBoost Regressor", "LightGBM", "LightGBM Regressor"],
        "features": ["boosting", "tabular"],
    },
    {
        "id": "text-classic",
        "label": "Text Classic",
        "size_mb": 280,
        "description": "TF-IDF plus classical text classifiers and regressors.",
        "models": ["TF-IDF + Logistic Regression", "TF-IDF + Naive Bayes", "SVM + TF-IDF"],
        "features": ["text", "tfidf"],
    },
    {
        "id": "text-transformers",
        "label": "Text Transformers",
        "size_mb": 1450,
        "description": "DistilBERT and BERT-ready text pack.",
        "models": ["DistilBERT", "BERT"],
        "features": ["text", "transformers"],
    },
    {
        "id": "timeseries-classic",
        "label": "Time Series Classic",
        "size_mb": 320,
        "description": "Lag models, ARIMA, Prophet, and smoothing workflows.",
        "models": ["ARIMA", "Prophet", "XGBoost with Lag Features", "Exponential Smoothing"],
        "features": ["timeseries", "forecasting"],
    },
    {
        "id": "timeseries-deep",
        "label": "Time Series Deep",
        "size_mb": 980,
        "description": "LSTM-based forecasting runtime and templates.",
        "models": ["LSTM"],
        "features": ["timeseries", "deep_learning"],
    },
    {
        "id": "vision-transfer",
        "label": "Vision Transfer Learning",
        "size_mb": 1850,
        "description": "ResNet50, EfficientNet-B0, and MobileNetV2 transfer-learning assets.",
        "models": ["ResNet50", "EfficientNet-B0", "MobileNetV2"],
        "features": ["image", "transfer_learning"],
    },
    {
        "id": "vision-transformers",
        "label": "Vision Transformers",
        "size_mb": 1300,
        "description": "ViT-ready image modeling support.",
        "models": ["Vision Transformer (ViT)", "ResNet50 + EfficientNet Ensemble"],
        "features": ["image", "transformers"],
    },
    {
        "id": "audio-classic",
        "label": "Audio Classic",
        "size_mb": 540,
        "description": "Spectrogram features and CNN-style audio workflow assets.",
        "models": ["Spectrogram + CNN"],
        "features": ["audio", "cnn"],
    },
    {
        "id": "audio-transformers",
        "label": "Audio Transformers",
        "size_mb": 1620,
        "description": "Wav2Vec2-ready audio pack.",
        "models": ["Wav2Vec2", "CNN + Wav2Vec2 Ensemble"],
        "features": ["audio", "transformers"],
    },
    {
        "id": "deep-learning-runtime",
        "label": "Deep Learning Runtime",
        "size_mb": 2100,
        "description": "TensorFlow, Keras, and PyTorch runtime support for local deep models.",
        "models": ["TensorFlow / Keras", "PyTorch"],
        "features": ["deep_learning", "runtime"],
    },
]


INSTALL_PROFILES = [
    {
        "id": "core",
        "label": "Core",
        "description": "Fastest install focused on tabular ML.",
        "pack_ids": ["core-tabular"],
    },
    {
        "id": "analyst",
        "label": "Analyst",
        "description": "Tabular, time-series, and classic text support.",
        "pack_ids": ["core-tabular", "boosting-pack", "text-classic", "timeseries-classic"],
    },
    {
        "id": "vision",
        "label": "Vision",
        "description": "Tabular plus transfer-learning image workflows.",
        "pack_ids": ["core-tabular", "boosting-pack", "vision-transfer"],
    },
    {
        "id": "research",
        "label": "Research",
        "description": "Broad local ML/DL set without the heaviest transformer packs.",
        "pack_ids": [
            "core-tabular",
            "boosting-pack",
            "text-classic",
            "timeseries-classic",
            "vision-transfer",
            "audio-classic",
            "deep-learning-runtime",
        ],
    },
    {
        "id": "full",
        "label": "Full",
        "description": "All packs enabled for the widest local benchmark set.",
        "pack_ids": [pack["id"] for pack in PACK_DEFINITIONS],
    },
]


def _pack_lookup() -> dict[str, dict[str, Any]]:
    return {pack["id"]: pack for pack in PACK_DEFINITIONS}


def _profile_lookup() -> dict[str, dict[str, Any]]:
    return {profile["id"]: profile for profile in INSTALL_PROFILES}


def _default_state() -> dict[str, Any]:
    return {
        "setup_complete": False,
        "selected_profile": "",
        "selected_pack_ids": [],
        "installed_pack_ids": [],
        "deferred_pack_ids": [],
        "selected_provider_ids": ["ollama"],
        "download_now": False,
        "updated_at": "",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def load_setup_state() -> dict[str, Any]:
    ensure_directories()
    if not SETUP_STATE_PATH.exists():
        state = _default_state()
        state["updated_at"] = _utc_now()
        _write_json(SETUP_STATE_PATH, state)
        return state
    return json.loads(SETUP_STATE_PATH.read_text(encoding="utf-8"))


def save_setup_state(state: dict[str, Any]) -> dict[str, Any]:
    next_state = {**_default_state(), **state, "updated_at": _utc_now()}
    _write_json(SETUP_STATE_PATH, next_state)
    return next_state


def _marker_path(pack_id: str) -> Path:
    return PACKS_DIR / f"{pack_id}.json"


def _mark_pack_installed(pack_id: str) -> None:
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    pack = _pack_lookup()[pack_id]
    payload = {
        "id": pack_id,
        "label": pack["label"],
        "installed_at": _utc_now(),
        "models": pack["models"],
    }
    _write_json(_marker_path(pack_id), payload)


def _mark_pack_removed(pack_id: str) -> None:
    path = _marker_path(pack_id)
    if path.exists():
        path.unlink(missing_ok=True)


def installed_pack_ids() -> list[str]:
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.stem for path in PACKS_DIR.glob("*.json"))


def install_packs(pack_ids: list[str]) -> list[str]:
    installed: list[str] = []
    for pack_id in pack_ids:
        if pack_id not in _pack_lookup():
            continue
        _mark_pack_installed(pack_id)
        installed.append(pack_id)
    state = load_setup_state()
    state["installed_pack_ids"] = sorted(set(state.get("installed_pack_ids", []) + installed))
    state["deferred_pack_ids"] = [pack_id for pack_id in state.get("deferred_pack_ids", []) if pack_id not in installed]
    save_setup_state(state)
    return installed


def remove_pack(pack_id: str) -> dict[str, Any]:
    _mark_pack_removed(pack_id)
    state = load_setup_state()
    state["installed_pack_ids"] = [item for item in state.get("installed_pack_ids", []) if item != pack_id]
    if pack_id in state.get("selected_pack_ids", []):
        state["deferred_pack_ids"] = sorted(set(state.get("deferred_pack_ids", []) + [pack_id]))
    return save_setup_state(state)


def _memory_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024**3), 2)
        except Exception:
            return None
    return None


def runtime_snapshot() -> dict[str, Any]:
    ensure_directories()
    disk = shutil.disk_usage(ROOT_DIR)
    free_disk_gb = round(disk.free / (1024**3), 2)
    total_disk_gb = round(disk.total / (1024**3), 2)
    ram_gb = _memory_gb()

    if free_disk_gb >= 20 and (ram_gb or 0) >= 12:
        recommended_profile = "full"
    elif free_disk_gb >= 12 and (ram_gb or 0) >= 8:
        recommended_profile = "research"
    elif free_disk_gb >= 8:
        recommended_profile = "analyst"
    else:
        recommended_profile = "core"

    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "free_disk_gb": free_disk_gb,
        "total_disk_gb": total_disk_gb,
        "memory_gb": ram_gb,
        "recommended_profile": recommended_profile,
        "user_site": site.getusersitepackages(),
        "python_executable": sys.executable,
    }


def list_profiles() -> list[dict[str, Any]]:
    packs = _pack_lookup()
    return [
        {
            **profile,
            "estimated_size_mb": sum(packs[pack_id]["size_mb"] for pack_id in profile["pack_ids"] if pack_id in packs),
        }
        for profile in INSTALL_PROFILES
    ]


def list_packs() -> list[dict[str, Any]]:
    installed = set(installed_pack_ids())
    return [
        {
            **pack,
            "installed": pack["id"] in installed,
            "marker_path": str(_marker_path(pack["id"])),
        }
        for pack in PACK_DEFINITIONS
    ]


def initialize_setup(
    *,
    profile_id: str,
    pack_ids: list[str],
    provider_ids: list[str],
    download_now: bool,
) -> dict[str, Any]:
    profile = _profile_lookup().get(profile_id)
    selected_pack_ids = pack_ids or (profile["pack_ids"] if profile else [])
    selected_pack_ids = [pack_id for pack_id in selected_pack_ids if pack_id in _pack_lookup()]
    state = load_setup_state()
    state.update(
        {
            "setup_complete": True,
            "selected_profile": profile_id,
            "selected_pack_ids": selected_pack_ids,
            "selected_provider_ids": provider_ids or ["ollama"],
            "download_now": bool(download_now),
        }
    )
    if download_now:
        state["installed_pack_ids"] = sorted(set(state.get("installed_pack_ids", []) + install_packs(selected_pack_ids)))
        state["deferred_pack_ids"] = []
    else:
        state["deferred_pack_ids"] = selected_pack_ids
    return save_setup_state(state)


def setup_status() -> dict[str, Any]:
    state = load_setup_state()
    installed = installed_pack_ids()
    state["installed_pack_ids"] = installed
    state = save_setup_state(state)
    packs = list_packs()
    profiles = list_profiles()
    snapshot = runtime_snapshot()
    profile_lookup = _profile_lookup()
    selected_profile = profile_lookup.get(state.get("selected_profile", ""))
    estimated_size_mb = sum(_pack_lookup()[pack_id]["size_mb"] for pack_id in state.get("selected_pack_ids", []) if pack_id in _pack_lookup())
    return {
        "needs_setup": not state.get("setup_complete", False),
        "setup_state": state,
        "packs": packs,
        "profiles": profiles,
        "runtime": snapshot,
        "selected_profile": selected_profile,
        "estimated_download_mb": estimated_size_mb,
    }
