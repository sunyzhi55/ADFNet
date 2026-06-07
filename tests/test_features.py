from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.features import average_landmarks, compute_adf_features, nearest_target_distance, sliding_mean
from models.distribution import soft_dtw_distance


def test_nearest_target_distance_for_hard_targets() -> None:
    distance = nearest_target_distance([10, 10], [[20, 10], [13, 14], [100, 100]])
    assert distance == 5.0


def test_sliding_mean_uses_trailing_window() -> None:
    values = np.asarray([1, 2, 3, 4], dtype=np.float32)
    np.testing.assert_allclose(sliding_mean(values, 2), [1.0, 1.5, 2.5, 3.5])


def test_compute_adf_features_easy_shape_and_diff() -> None:
    records = [
        {
            "gaze_screen_tf_calibrate_xy_px": [0, 0],
            "target_xy_px": [3, 4],
        },
        {
            "gaze_screen_tf_calibrate_xy_px": [0, 0],
            "target_xy_px": [6, 8],
        },
    ]
    adf = compute_adf_features(records, "easy", local_mean_size=2)
    assert adf.shape == (2, 3)
    np.testing.assert_allclose(adf[:, 0], [5.0, 10.0])
    np.testing.assert_allclose(adf[:, 1], [0.0, 5.0])
    np.testing.assert_allclose(adf[:, 2], [5.0, 7.5])


def test_average_landmarks_handles_inconsistent_shapes() -> None:
    records = [
        {"landmarks": [[1.0, 2.0], [3.0, 4.0]]},
        {"landmarks": [[5.0, 6.0]]},
        {"landmarks": []},
    ]
    landmarks = average_landmarks(records, landmark_dim=6)
    assert landmarks.shape == (6,)
    np.testing.assert_allclose(landmarks, [3.0, 4.0, 1.5, 2.0, 0.0, 0.0])


def test_soft_dtw_distance_is_finite() -> None:
    value = soft_dtw_distance(np.arange(8, dtype=np.float32), np.arange(4, dtype=np.float32), gamma=1.0)
    assert np.isfinite(value)
