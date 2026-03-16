from pathlib import Path

import pandas as pd

from backend.modules.detector import detect_dataset


def test_detector_identifies_text_dataset(tmp_path: Path):
    path = tmp_path / "text.csv"
    pd.DataFrame(
        {
            "review_text": [
                "This product is genuinely fantastic and works better than expected.",
                "Terrible purchase and the quality was disappointing.",
                "Solid overall with a few small issues.",
            ],
            "label": ["positive", "negative", "positive"],
        }
    ).to_csv(path, index=False)

    detection = detect_dataset(path)

    assert detection["dataset_type"] == "text"
    assert detection["dataset_context"]["text_column"] == "review_text"


def test_detector_identifies_timeseries_dataset(tmp_path: Path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "sales": range(40),
        }
    ).to_csv(path, index=False)

    detection = detect_dataset(path)

    assert detection["dataset_type"] == "timeseries"
    assert detection["dataset_context"]["time_column"] == "date"
