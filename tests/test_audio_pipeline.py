import io
import wave
import zipfile
from pathlib import Path

import numpy as np

from backend.modules.audio_pipeline import clean_audio_dataset
from backend.modules.detector import detect_dataset


def _wav_bytes(frequency: float = 220.0, sample_rate: int = 16000, duration_seconds: float = 0.25) -> bytes:
    sample_count = int(sample_rate * duration_seconds)
    timeline = np.arange(sample_count)
    waveform = (0.25 * np.sin(2 * np.pi * frequency * timeline / sample_rate) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())
    return buffer.getvalue()


def test_detector_identifies_audio_archive(tmp_path: Path):
    archive_path = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("yes/sample1.wav", _wav_bytes(220.0))
        archive.writestr("no/sample2.wav", _wav_bytes(440.0))

    detection = detect_dataset(archive_path)

    assert detection["dataset_type"] == "audio"
    assert detection["target_column"] == "label"
    assert detection["dataset_context"]["audio_count"] == 2


def test_clean_audio_dataset_extracts_features(tmp_path: Path):
    archive_path = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("yes/sample1.wav", _wav_bytes(220.0))
        archive.writestr("no/sample2.wav", _wav_bytes(440.0))

    cleaned = clean_audio_dataset(archive_path)

    dataframe = cleaned["dataframe"]
    assert len(dataframe) == 2
    assert "spectrogram_mean" in dataframe.columns
    assert cleaned["summary"]["rows_after"] == 2
