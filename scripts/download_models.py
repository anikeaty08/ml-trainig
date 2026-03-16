from __future__ import annotations

import argparse
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
from backend.utils.setup_manager import initialize_setup, install_packs, list_packs, list_profiles


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local builtin models and install model packs.")
    parser.add_argument("--profile", default="", help="Install a named setup profile such as core, analyst, research, or full.")
    parser.add_argument("--packs", default="", help="Comma-separated pack ids to mark as installed.")
    parser.add_argument("--interactive", action="store_true", help="Choose a profile interactively in the terminal.")
    return parser.parse_args()


def interactive_profile() -> str:
    profiles = list_profiles()
    print("Select a local setup profile:")
    for index, profile in enumerate(profiles, start=1):
        print(f"{index}. {profile['label']} ({profile['id']}) - {profile['estimated_size_mb']} MB")
    raw = input("Choice [default 1]: ").strip()
    try:
        selected = profiles[max(0, int(raw or "1") - 1)]
    except Exception:
        selected = profiles[0]
    return selected["id"]


if __name__ == "__main__":
    args = parse_args()
    train_builtin_models()
    selected_profile = args.profile
    if args.interactive:
        selected_profile = interactive_profile()
    if selected_profile:
        profile_lookup = {profile["id"]: profile for profile in list_profiles()}
        profile = profile_lookup.get(selected_profile)
        if profile:
            initialize_setup(
                profile_id=selected_profile,
                pack_ids=profile["pack_ids"],
                provider_ids=["ollama"],
                download_now=True,
            )
    if args.packs:
        install_packs([item.strip() for item in args.packs.split(",") if item.strip()])
    installed = [pack["id"] for pack in list_packs() if pack["installed"]]
    print(f"Built-in models are ready. Installed packs: {', '.join(installed) or 'none'}")
