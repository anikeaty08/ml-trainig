import pandas as pd

from backend.modules.cleaner import clean_tabular_dataset


def test_cleaner_removes_duplicates_and_imputes():
    dataframe = pd.DataFrame(
        {
            "feature": [1, 1, None, 2],
            "category": ["a", "a", None, "b"],
            "target": [0, 0, 1, 1],
        }
    )

    cleaned = clean_tabular_dataset(dataframe, target_column="target")
    result = cleaned["dataframe"]

    assert len(result) == 3
    assert result["feature"].isna().sum() == 0
    assert result["category"].isna().sum() == 0
