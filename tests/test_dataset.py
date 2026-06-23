from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.io import discover_sequences, parse_sequence_filename


def _record(idx: int) -> dict:
    return {
        "timestamp": idx / 25,
        "frame_idx": idx,
        "gaze_screen_tf_calibrate_xy_px": [float(idx), 0.0],
        "target_xy_px": [float(idx + 3), 4.0],
        "deviation_px_after_calibrate": float(idx) + 0.25,
        "bbox": [0, 0, 10, 10],
        "landmarks": [[float(i), float(i + 1)] for i in range(35)],
        "confidence": 0.99,
    }


def test_parse_sequence_filename() -> None:
    info = parse_sequence_filename("S01_easy_alert.jsonl")
    assert info.subject_id == "S01"
    assert info.task_type == "easy"
    assert info.label == 0


def test_dataset_windowing_and_label(tmp_path: Path) -> None:
    data_file = tmp_path / "S01_easy_sleepy.jsonl"
    with data_file.open("w", encoding="utf-8") as handle:
        for idx in range(5):
            handle.write(json.dumps(_record(idx), ensure_ascii=False) + "\n")
    sequences = discover_sequences(tmp_path)
    dataset = ADFWindowDataset(sequences=sequences, window_size=4, stride=1, local_mean_size=2)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["adf"].shape == (4, 3)
    assert sample["label"].item() == 1.0
    assert sample["landmarks"].shape[0] == 70
    assert sample["subject_id"] == "S01"
    assert sample["adf"][:, 0].tolist() == [0.25, 1.25, 2.25, 3.25]

