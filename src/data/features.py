from __future__ import annotations

import numpy as np

from data.io import parse_points


def nearest_target_distance(
    gaze_xy: np.ndarray,
    target_xy: np.ndarray | list[float] | list[list[float]],
) -> float:
    gaze = np.asarray(gaze_xy, dtype=np.float32).reshape(2)
    targets = np.asarray(parse_points(target_xy), dtype=np.float32)
    if targets.size == 0:
        return 0.0
    distances = np.linalg.norm(targets.reshape(-1, 2) - gaze[None, :], axis=1)
    return float(distances.min())


def sliding_mean(values: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("sliding_mean expects a 1-D array")
    if window_size <= 1:
        return values.copy()
    output = np.empty_like(values)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    for idx in range(len(values)):
        start = max(0, idx - window_size + 1)
        output[idx] = (cumsum[idx + 1] - cumsum[start]) / (idx - start + 1)
    return output


def compute_adf_features(
    records: list[dict],
    task_type: str,
    local_mean_size: int = 16,
) -> np.ndarray:
    distances: list[float] = []
    for record in records:
        gaze_points = parse_points(record.get("gaze_screen_tf_calibrate_xy_px"))
        if not gaze_points:
            gaze_points = parse_points(record.get("gaze_screen_xy_px"))
        if not gaze_points:
            distances.append(0.0)
            continue
        gaze_xy = np.asarray(gaze_points[0], dtype=np.float32)
        target = record.get("target_centers_xy_px") if task_type == "hard" else record.get("target_xy_px")
        distances.append(nearest_target_distance(gaze_xy, target))

    drift = np.asarray(distances, dtype=np.float32)
    diff = np.diff(drift, prepend=drift[:1])
    local_mean = sliding_mean(drift, local_mean_size)
    return np.stack([drift, diff.astype(np.float32), local_mean], axis=-1)


def average_landmarks(records: list[dict], landmark_dim: int = 70) -> np.ndarray:
    """Average landmarks after normalizing every frame to a fixed vector size.

    Some detectors emit fewer landmarks on low-confidence frames. Those frames
    are still usable: existing coordinates are copied, missing tail dimensions
    are padded with zeros, and completely invalid frames are skipped.
    """
    frames: list[np.ndarray] = []
    for record in records:
        points = parse_points(record.get("RetinaFace_landmarks"))
        if not points:
            continue
        vector = np.asarray(points, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            continue
        normalized = np.zeros(landmark_dim, dtype=np.float32)
        copy_len = min(vector.size, landmark_dim)
        normalized[:copy_len] = vector[:copy_len]
        frames.append(normalized)
    if not frames:
        return np.zeros(landmark_dim, dtype=np.float32)
    return np.mean(np.stack(frames, axis=0), axis=0).astype(np.float32)
