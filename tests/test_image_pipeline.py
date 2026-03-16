import io
import zipfile
from pathlib import Path

from PIL import Image

from backend.modules.image_pipeline import clean_image_dataset, detect_image_archive


def _make_png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (24, 24), color=color)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_archive_detection_and_cleaning(tmp_path: Path):
    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("cats/cat_1.png", _make_png_bytes((255, 0, 0)))
        archive.writestr("dogs/dog_1.png", _make_png_bytes((0, 255, 0)))

    detection = detect_image_archive(archive_path)
    cleaned = clean_image_dataset(archive_path)

    assert detection["dataset_type"] == "image"
    assert detection["problem_type"] == "classification"
    assert cleaned["dataframe"].shape[0] == 2
    assert "label" in cleaned["dataframe"].columns
