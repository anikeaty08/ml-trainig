from backend.modules.recommender import provider_catalog, recommend_model_families


def test_recommender_returns_models():
    analysis = {
        "shape": {"rows": 1000, "columns": 8},
        "categorical_feature_count": 2,
        "dataset_fitness": ["Good fit for classification workflows"],
    }

    recommendation = recommend_model_families(analysis, task_type="classification")

    assert recommendation["recommended"]
    assert provider_catalog()
