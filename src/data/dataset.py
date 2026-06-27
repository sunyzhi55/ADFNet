from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data.features import average_landmarks, compute_adf_features
from data.io import SequenceInfo, discover_sequences, filter_sequences_by_task, iter_jsonl


@dataclass(frozen=True)
class WindowSample:
    adf: np.ndarray
    label: int
    landmarks: np.ndarray
    subject_id: str
    task_type: str
    path: Path
    dist_stats: np.ndarray | None = None
    subject_label: int = -1  # GRL 对抗目标：train fold 内 subject_id->int，未知被试保持 -1


class ADFWindowDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        sequences: Sequence[SequenceInfo] | None = None,
        window_size: int = 256,
        stride: int = 64,
        local_mean_size: int = 16,
        landmark_dim: int = 70,
        min_confidence: float | None = None,
        task_mode: str = "all",
    ) -> None:
        if sequences is None:
            if root is None:
                raise ValueError("root and sequences cannot both be None")
            sequences = discover_sequences(root)
        sequences = filter_sequences_by_task(list(sequences), task_mode)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.window_size = window_size
        self.stride = stride
        self.samples: list[WindowSample] = []
        for info in sequences:
            records = list(iter_jsonl(info.path))
            if min_confidence is not None:
                records = [r for r in records if float(r.get("confidence", 1.0)) >= min_confidence]
            if len(records) < window_size:
                continue

            adf = compute_adf_features(records, info.task_type, local_mean_size)
            # The range upper bound intentionally drops the trailing fragment
            # when a sequence length is not divisible by stride/window size.
            for start in range(0, len(records) - window_size + 1, stride):
                end = start + window_size
                window_records = records[start:end]
                self.samples.append(
                    WindowSample(
                        adf=adf[start:end].astype(np.float32),
                        label=info.label,
                        landmarks=average_landmarks(window_records, landmark_dim),
                        subject_id=info.subject_id,
                        task_type=info.task_type,
                        path=info.path,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        return {
            "adf": torch.from_numpy(sample.adf),
            "label": torch.tensor([sample.label], dtype=torch.float32),
            "landmarks": torch.from_numpy(sample.landmarks),
            "dist_stats": torch.from_numpy(sample.dist_stats) if sample.dist_stats is not None else torch.empty(0),
            "subject_id": sample.subject_id,
            "subject_label": torch.tensor([sample.subject_label], dtype=torch.long),
            "task_type": sample.task_type,
        }

    def attach_distribution_stats(
        self,
        gamma_reference,
        feature_window: int | None = None,
        stats_mean: np.ndarray | None = None,
        stats_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.samples:
            mean = np.zeros(3, dtype=np.float32) if stats_mean is None else stats_mean.astype(np.float32)
            std = np.ones(3, dtype=np.float32) if stats_std is None else stats_std.astype(np.float32)
            return mean, std
        adf = np.stack([sample.adf for sample in self.samples], axis=0)
        stats = gamma_reference.features_numpy(adf, feature_window=feature_window)
        stats = np.nan_to_num(stats, nan=0.0, posinf=1.0e6, neginf=-1.0e6).astype(np.float32)
        if stats_mean is None or stats_std is None:
            stats_mean = stats.mean(axis=0).astype(np.float32)
            stats_std = stats.std(axis=0).astype(np.float32)
        stats_std = np.where(stats_std < 1.0e-6, 1.0, stats_std).astype(np.float32)
        stats = ((stats - stats_mean) / stats_std).astype(np.float32)
        self.samples = [
            WindowSample(
                adf=sample.adf,
                label=sample.label,
                landmarks=sample.landmarks,
                subject_id=sample.subject_id,
                task_type=sample.task_type,
                path=sample.path,
                dist_stats=stats[idx].astype(np.float32),
            )
            for idx, sample in enumerate(self.samples)
        ]
        return stats_mean.astype(np.float32), stats_std.astype(np.float32)

    def attach_subject_labels(self, mapping: dict[str, int]) -> None:
        """按 train fold 的 subject_id->int 映射，给每个窗口打上身份标签。

        未知被试（验证集/留出被试）映射不到则保持 -1，由损失里的
        CrossEntropyLoss(ignore_index=-1) 忽略，不参与对抗。
        """
        self.samples = [
            WindowSample(
                adf=sample.adf,
                label=sample.label,
                landmarks=sample.landmarks,
                subject_id=sample.subject_id,
                task_type=sample.task_type,
                path=sample.path,
                dist_stats=sample.dist_stats,
                subject_label=int(mapping.get(sample.subject_id, -1)),
            )
            for sample in self.samples
        ]


def collect_alert_distances(dataset: ADFWindowDataset) -> np.ndarray:
    values = [sample.adf[:, 0] for sample in dataset.samples if sample.label == 0]
    if not values:
        raise ValueError("No alert samples found in the training set; cannot fit Gamma reference")
    return np.concatenate(values, axis=0).astype(np.float32)
