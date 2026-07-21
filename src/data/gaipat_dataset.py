"""GAIPAT 公开数据集：数据加载、特征计算与 Dataset 类。

目录结构::

    <gaipat_root>/
    ├── release/
    │   ├── 5530740_house_4_release_21_0.jsonl
    │   └── ...
    └── grasp/
        ├── 5530740_house_4_grasp_21_1.jsonl
        └── ...

文件命名: ``{subject_id}_{task}_{step}_{event}_{block_id}_{label}.jsonl``

标签规则（已完成 relabel）:
    0 = 分心 (Distracted / Wandering for release)
    1 = 专注 (Focused)
    2, 3 = 丢弃

核心特征: ``deviation_cm``（视线偏差距离，厘米）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data.features import sliding_mean
from data.io import SequenceInfo, iter_jsonl

# ── 文件名解析 ─────────────────────────────────────────────────

GAIPAT_FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<task>\w+)_(?P<step>\d+)_"
    r"(?P<event>\w+)_(?P<block>\d+)_(?P<label>\d+)\.jsonl$",
    re.I,
)


def parse_gaipat_filename(path: str | Path) -> SequenceInfo | None:
    """解析 GAIPAT JSONL 文件名，返回 SequenceInfo。

    label 为 2 或 3 的文件返回 None（应丢弃）。
    """
    path = Path(path)
    m = GAIPAT_FILENAME_RE.match(path.name)
    if not m:
        return None
    label = int(m.group("label"))
    if label not in (0, 1):
        return None
    event = m.group("event").lower()
    return SequenceInfo(
        path=path,
        subject_id=m.group("subject"),
        task_type=event,       # release / grasp
        label_name="focused" if label == 1 else "distracted",
        label=label,
    )


# ── 数据发现 ────────────────────────────────────────────────────

def discover_gaipat_sequences(root: str | Path) -> list[SequenceInfo]:
    """扫描 GAIPAT 根目录（release/ + grasp/）下所有有效 JSONL。"""
    root = Path(root)
    if not root.exists():
        return []
    seqs: list[SequenceInfo] = []
    for p in sorted(root.rglob("*.jsonl")):
        info = parse_gaipat_filename(p)
        if info is not None:
            seqs.append(info)
    return sorted(seqs, key=lambda s: (s.subject_id, s.task_type, str(s.path)))


# ── 特征计算 ────────────────────────────────────────────────────

def compute_gaipat_features(
    records: list[dict],
    local_mean_size: int = 16,
    per_sample_norm: bool = True,
) -> np.ndarray:
    """从 GAIPAT JSONL 记录计算 ADF 三通道特征。

    直接用 ``deviation_cm`` 作为 drift（无需 gaze-target 坐标计算）。
    当 ``per_sample_norm=True`` 时对 drift 做 Min-Max 归一化到 [0, 1]，
    消除 px / cm 单位差异。
    """
    drift = np.array(
        [float(r.get("deviation_cm", 0.0)) for r in records],
        dtype=np.float32,
    )
    if per_sample_norm:
        lo, hi = float(drift.min()), float(drift.max())
        drift = ((drift - lo) / (hi - lo + 1e-8)).astype(np.float32)
    diff = np.diff(drift, prepend=drift[:1])
    local_mean = sliding_mean(drift, local_mean_size)
    return np.stack([drift, diff.astype(np.float32), local_mean], axis=-1)


# ── WindowSample ────────────────────────────────────────────────

@dataclass(frozen=True)
class GaipatWindowSample:
    adf: np.ndarray
    label: int
    subject_id: str
    event_type: str
    path: Path
    dist_stats: np.ndarray | None = None
    subject_label: int = -1


# ── Dataset ─────────────────────────────────────────────────────

class GaipatWindowDataset(Dataset):
    """GAIPAT 窗口数据集。

    每个 JSONL 文件恰好 ``window_size`` 条记录 → 1 个窗口样本。
    接口与 :class:`ADFWindowDataset` 对齐，可无缝接入 ``train_fold`` /
    ``evaluate_checkpoint``。
    """

    def __init__(
        self,
        sequences: Sequence[SequenceInfo],
        window_size: int = 256,
        local_mean_size: int = 16,
        per_sample_norm: bool = True,
        **_ignored,
    ) -> None:
        self.window_size = window_size
        self.samples: list[GaipatWindowSample] = []
        for info in sequences:
            records = list(iter_jsonl(info.path))
            records = [r for r in records if r.get("deviation_cm") is not None]
            if len(records) < window_size:
                continue
            adf = compute_gaipat_features(
                records[:window_size],
                local_mean_size=local_mean_size,
                per_sample_norm=per_sample_norm,
            )
            self.samples.append(
                GaipatWindowSample(
                    adf=adf.astype(np.float32),
                    label=(info.label + 1) % 2,  # GAIPAT: 0=分心,1=专注 → ADF: 0=专注,1=分心
                    subject_id=info.subject_id,
                    event_type=info.task_type,
                    path=info.path,
                )
            )

    # ── 标准 Dataset 接口 ──

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "adf": torch.from_numpy(s.adf),
            "label": torch.tensor([s.label], dtype=torch.float32),
            "landmarks": torch.zeros(70, dtype=torch.float32),
            "dist_stats": (
                torch.from_numpy(s.dist_stats)
                if s.dist_stats is not None
                else torch.empty(0)
            ),
            "subject_id": s.subject_id,
            "subject_label": torch.tensor([s.subject_label], dtype=torch.long),
            "task_type": s.event_type,
        }

    # ── 分布统计（与 ADFWindowDataset 接口一致） ──

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

        adf = np.stack([s.adf for s in self.samples], axis=0)
        stats = gamma_reference.features_numpy(adf, feature_window=feature_window)
        stats = np.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

        if stats_mean is None or stats_std is None:
            stats_mean = stats.mean(axis=0).astype(np.float32)
            stats_std = stats.std(axis=0).astype(np.float32)
        stats_std = np.where(stats_std < 1e-6, 1.0, stats_std).astype(np.float32)
        stats = ((stats - stats_mean) / stats_std).astype(np.float32)

        self.samples = [
            GaipatWindowSample(
                adf=s.adf,
                label=s.label,
                subject_id=s.subject_id,
                event_type=s.event_type,
                path=s.path,
                dist_stats=stats[i].astype(np.float32),
                subject_label=s.subject_label,
            )
            for i, s in enumerate(self.samples)
        ]
        return stats_mean.astype(np.float32), stats_std.astype(np.float32)

    def attach_subject_labels(self, mapping: dict[str, int]) -> None:
        self.samples = [
            GaipatWindowSample(
                adf=s.adf,
                label=s.label,
                subject_id=s.subject_id,
                event_type=s.event_type,
                path=s.path,
                dist_stats=s.dist_stats,
                subject_label=int(mapping.get(s.subject_id, -1)),
            )
            for s in self.samples
        ]


# ── 工具函数 ────────────────────────────────────────────────────

def collect_gaipat_alert_distances(dataset: GaipatWindowDataset) -> np.ndarray:
    """收集 label=0（focused/alert）样本的 drift 值，用于拟合 Gamma 参考分布。"""
    values = [s.adf[:, 0] for s in dataset.samples if s.label == 0]
    if not values:
        raise ValueError("No focused (label=0) samples in GAIPAT dataset; cannot fit Gamma reference")
    return np.concatenate(values, axis=0).astype(np.float32)
