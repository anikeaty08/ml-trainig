import pandas as pd

from backend.modules.analyzer import analyze_dataset


def test_analyzer_returns_shape_and_summary():
    dataframe = pd.DataFrame(
        {
            "age": [25, 30, 41, 52],
            "income": [50, 60, 80, 90],
            "target": [0, 1, 1, 0],
        }
    )

    summary = analyze_dataset(dataframe, target_column="target", task_type="classification")

    assert summary["shape"]["rows"] == 4
    assert summary["numeric_feature_count"] == 2
    assert "target_summary" in summary
