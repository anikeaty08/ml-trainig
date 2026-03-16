from __future__ import annotations

from typing import Any


def provider_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": "ollama",
            "label": "Ollama",
            "provider_type": "local",
            "capabilities": ["chat", "image", "onboard"],
            "auth_modes": ["local"],
            "default_base_url": "http://127.0.0.1:11434",
            "onboarding_hint": "Install Ollama locally and pull models such as llama3.2 or llava:7b.",
            "suggested_models": ["ollama/llama3.2", "ollama/qwen2.5", "ollama/llava:7b"],
        },
        {
            "id": "lmstudio",
            "label": "LM Studio",
            "provider_type": "local",
            "capabilities": ["chat", "image", "onboard"],
            "auth_modes": ["local", "api_key"],
            "default_base_url": "http://127.0.0.1:1234/v1",
            "onboarding_hint": "Start the local OpenAI-compatible server inside LM Studio.",
            "suggested_models": ["lmstudio/qwen2.5-coder", "lmstudio/llama-3.1-8b-instruct"],
        },
        {
            "id": "openai_compatible",
            "label": "OpenAI-Compatible Endpoint",
            "provider_type": "custom",
            "capabilities": ["chat", "image", "profiles"],
            "auth_modes": ["api_key"],
            "default_base_url": "http://127.0.0.1:4000/v1",
            "onboarding_hint": "Point this at any local gateway or self-hosted OpenAI-compatible server.",
            "suggested_models": ["openai_compatible/local-model", "openai_compatible/vision-model"],
        },
        {
            "id": "openai",
            "label": "OpenAI API",
            "provider_type": "cloud_optional",
            "capabilities": ["chat", "image", "profiles", "browser_login"],
            "auth_modes": ["api_key", "browser_login", "oauth"],
            "default_base_url": "https://api.openai.com/v1",
            "onboarding_hint": "Use an API key profile or store a browser-login hint for future support.",
            "suggested_models": ["openai/gpt-4.1-mini", "openai/gpt-4.1", "openai/gpt-4o-mini"],
        },
        {
            "id": "openai-codex",
            "label": "OpenAI Codex",
            "provider_type": "cloud_optional",
            "capabilities": ["chat", "browser_login", "profiles", "onboard"],
            "auth_modes": ["browser_login", "oauth", "api_key"],
            "default_base_url": "https://api.openai.com/v1",
            "onboarding_hint": "Use a ChatGPT-style browser-login profile or an API-key-backed OpenAI endpoint if available.",
            "suggested_models": ["openai-codex/gpt-5.3-codex", "openai-codex/gpt-5.1-codex"],
        },
        {
            "id": "anthropic",
            "label": "Anthropic",
            "provider_type": "cloud_optional",
            "capabilities": ["chat", "profiles", "onboard"],
            "auth_modes": ["api_key", "browser_login", "oauth"],
            "default_base_url": "https://api.anthropic.com/v1",
            "onboarding_hint": "Add an Anthropic API key for Claude models. Browser-login/setup-token style onboarding can be tracked as a local profile hint.",
            "suggested_models": ["anthropic/claude-3-7-sonnet-latest", "anthropic/claude-3-5-haiku-latest"],
        },
        {
            "id": "google",
            "label": "Google Gemini",
            "provider_type": "cloud_optional",
            "capabilities": ["chat", "image", "profiles", "onboard"],
            "auth_modes": ["api_key", "browser_login", "oauth"],
            "default_base_url": "https://generativelanguage.googleapis.com",
            "onboarding_hint": "Add a Gemini API key or keep a browser-login hint if you use Google-hosted tools outside this app.",
            "suggested_models": ["google/gemini-2.5-flash", "google/gemini-2.5-pro"],
        },
        {
            "id": "kimi",
            "label": "Kimi / Moonshot",
            "provider_type": "cloud_optional",
            "capabilities": ["chat", "profiles", "onboard"],
            "auth_modes": ["api_key"],
            "default_base_url": "https://api.moonshot.ai/v1",
            "onboarding_hint": "Add your Moonshot API key and use Kimi models through the OpenAI-compatible endpoint.",
            "suggested_models": ["kimi/moonshot-v1-8k", "kimi/kimi-k2"],
        },
        {
            "id": "openrouter",
            "label": "OpenRouter",
            "provider_type": "cloud_optional",
            "capabilities": ["chat", "image", "profiles", "onboard"],
            "auth_modes": ["api_key"],
            "default_base_url": "https://openrouter.ai/api/v1",
            "onboarding_hint": "Add an OpenRouter API key to access many routed models with provider/model refs.",
            "suggested_models": ["openrouter/openai/gpt-4o-mini", "openrouter/moonshotai/kimi-k2"],
        },
    ]


def recommend_model_families(analysis: dict[str, Any], task_type: str) -> dict[str, Any]:
    rows = analysis["shape"]["rows"]
    categorical_count = analysis["categorical_feature_count"]
    dataset_type = analysis.get("dataset_type", "tabular")

    if dataset_type == "text":
        recommended = [
            {"name": "Logistic Regression + TF-IDF", "reason": "Strong text baseline with good interpretability"},
            {"name": "Linear SVM-style margin models", "reason": "Often competitive on sparse text features"},
            {"name": "Ridge / Linear Regression + TF-IDF", "reason": "Good fast baseline for text regression"},
            {"name": "Random Forest on text embeddings/features", "reason": "Useful non-linear comparison once text is vectorized"},
        ]
        return {
            "recommended": recommended,
            "not_recommended": [{"name": "KNN", "reason": "Sparse text spaces usually make it noisy and slow"}],
            "dataset_fit": analysis["dataset_fitness"],
        }

    if dataset_type == "timeseries":
        recommended = [
            {"name": "Lagged Linear/Ridge Regression", "reason": "Strong interpretable forecasting baseline"},
            {"name": "Random Forest Regressor", "reason": "Useful for non-linear lag interactions"},
            {"name": "Histogram Gradient Boosting Regressor", "reason": "Strong local forecasting candidate"},
            {"name": "Voting Ensemble", "reason": "Combines multiple lag-based forecasters"},
        ]
        return {
            "recommended": recommended,
            "not_recommended": [{"name": "Shuffle-based CV models", "reason": "Temporal ordering must be respected"}],
            "dataset_fit": analysis["dataset_fitness"],
        }

    if dataset_type == "image":
        recommended = [
            {"name": "Random Forest on visual features", "reason": "Robust default for handcrafted image features"},
            {"name": "Extra Trees on visual features", "reason": "Fast ensemble comparison for local image features"},
            {"name": "Histogram Gradient Boosting", "reason": "Useful boosted-tree comparison on derived image signals"},
            {"name": "SVM", "reason": "Can perform well on compact visual feature spaces"},
        ]
        return {
            "recommended": recommended,
            "not_recommended": [{"name": "Naive Bayes", "reason": "Usually too weak for rich visual patterns"}],
            "dataset_fit": analysis["dataset_fitness"],
        }

    if dataset_type == "audio":
        recommended = [
            {"name": "Spectrogram + CNN", "reason": "Natural deep-learning comparison if TensorFlow or PyTorch is installed"},
            {"name": "Random Forest on spectrogram features", "reason": "Strong local baseline on engineered audio features"},
            {"name": "Gradient Boosting on spectrogram features", "reason": "Useful non-linear comparison on audio descriptors"},
            {"name": "Wav2Vec2", "reason": "Transformer-style audio comparison when the full stack is installed"},
        ]
        return {
            "recommended": recommended,
            "not_recommended": [{"name": "Naive Bayes", "reason": "Audio patterns usually need richer decision boundaries"}],
            "dataset_fit": analysis["dataset_fitness"],
        }

    if task_type == "classification":
        recommended = [
            {"name": "Logistic Regression", "reason": "Strong baseline and sanity check"},
            {"name": "Random Forest", "reason": "Good mixed-feature default"},
            {"name": "Extra Trees", "reason": "Fast strong tabular ensemble"},
            {"name": "Histogram Gradient Boosting", "reason": "Strong boosted-tree candidate"},
            {"name": "SVM", "reason": "Useful comparison when the feature space is manageable"},
        ]
        not_recommended = []
        if rows > 20000:
            not_recommended.append({"name": "KNN", "reason": "Prediction gets slow on large datasets"})
        if categorical_count > 20:
            not_recommended.append({"name": "SVM", "reason": "Large encoded spaces can be slow"})
    else:
        recommended = [
            {"name": "Linear Regression", "reason": "Baseline for signal calibration"},
            {"name": "Ridge", "reason": "Regularized linear baseline"},
            {"name": "Random Forest Regressor", "reason": "Robust non-linear default"},
            {"name": "Histogram Gradient Boosting Regressor", "reason": "Strong larger-data candidate"},
        ]
        not_recommended = []

    return {
        "recommended": recommended,
        "not_recommended": not_recommended,
        "dataset_fit": analysis["dataset_fitness"],
    }
