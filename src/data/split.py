from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from data.io import SequenceInfo


@dataclass(frozen=True)
class SubjectFold:
    """单个 fold 的划分。

    ``train`` / ``val`` 仅来自「非测试」被试池：测试被试由配置 ``split.test_subjects``
    全局 hold-out，绝不进入任何 fold 的训练或验证，仅供训练结束后用
    ``evaluate.py`` 做最终评估，从而保证 val（选轮用）与 test（报告用）严格分离。
    """

    name: str
    train: list[SequenceInfo]
    val: list[SequenceInfo]
    val_subjects: tuple[str, ...] = ()


def subject_ids(sequences: Iterable[SequenceInfo]) -> list[str]:
    return sorted({seq.subject_id for seq in sequences})


def filter_by_subjects(sequences: Iterable[SequenceInfo], subjects: Sequence[str | int]) -> list[SequenceInfo]:
    """返回 subject_id ∈ subjects 的序列。"""
    keep = {str(s) for s in subjects}
    return [seq for seq in sequences if seq.subject_id in keep]


def exclude_subjects(sequences: Iterable[SequenceInfo], subjects: Sequence[str | int]) -> list[SequenceInfo]:
    """返回 subject_id ∉ subjects 的序列（用于剔除测试被试）。"""
    drop = {str(s) for s in subjects}
    return [seq for seq in sequences if seq.subject_id not in drop]


def _train_val_pool(sequences: list[SequenceInfo], test_subjects: Sequence[str | int]) -> list[SequenceInfo]:
    """排除测试被试后的 train+val 池。"""
    if not test_subjects:
        return list(sequences)
    return exclude_subjects(sequences, test_subjects)


def loso_folds(
    sequences: list[SequenceInfo],
    test_subjects: Sequence[str | int] = (),
) -> list[SubjectFold]:
    """留一被试法：在「非测试」池上，每个被试轮流当 val，其余当 train。"""
    pool = _train_val_pool(sequences, test_subjects)
    folds: list[SubjectFold] = []
    for subject in subject_ids(pool):
        val = [seq for seq in pool if seq.subject_id == subject]
        train = [seq for seq in pool if seq.subject_id != subject]
        folds.append(SubjectFold(name=f"loso_{subject}", train=train, val=val, val_subjects=(subject,)))
    return folds


def group_kfold_folds(
    sequences: list[SequenceInfo],
    n_splits: int = 5,
    seed: int = 42,
    test_subjects: Sequence[str | int] = (),
) -> list[SubjectFold]:
    """按被试分组的 K 折：在「非测试」池上把被试随机均分为 K 组，每组轮流当 val。"""
    pool = _train_val_pool(sequences, test_subjects)
    subjects = np.asarray(subject_ids(pool), dtype=object)
    if n_splits < 2 or n_splits > len(subjects):
        raise ValueError(f"n_splits must be in [2, subject_count]; got {n_splits} vs {len(subjects)}")
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    groups = np.array_split(shuffled, n_splits)
    folds: list[SubjectFold] = []
    for idx, group in enumerate(groups, start=1):
        val_subs = set(group.tolist())
        val = [seq for seq in pool if seq.subject_id in val_subs]
        train = [seq for seq in pool if seq.subject_id not in val_subs]
        folds.append(
            SubjectFold(
                name=f"groupkfold_{idx}",
                train=train,
                val=val,
                val_subjects=tuple(sorted(val_subs)),
            )
        )
    return folds


def explicit_folds(
    sequences: list[SequenceInfo],
    fold_val_subjects: Sequence[Sequence[str | int]],
    test_subjects: Sequence[str | int] = (),
) -> list[SubjectFold]:
    """显式指定每 fold 的验证被试 id（保证跨实验完全一致）。

    ``fold_val_subjects`` 形如 ``[["01","02"], ["03"], ...]``；每个子列表是该 fold
    的 val 被试，池中其余（已排除测试被试）作为该 fold 的 train。
    """
    pool = _train_val_pool(sequences, test_subjects)
    folds: list[SubjectFold] = []
    for idx, val_list in enumerate(fold_val_subjects, start=1):
        val_set = {str(s) for s in val_list}
        val = [seq for seq in pool if seq.subject_id in val_set]
        train = [seq for seq in pool if seq.subject_id not in val_set]
        name = f"fold_{idx:02d}_" + "-".join(sorted(val_set))
        folds.append(
            SubjectFold(name=name, train=train, val=val, val_subjects=tuple(sorted(val_set)))
        )
    return folds


def test_sequences(sequences: Iterable[SequenceInfo], test_subjects: Sequence[str | int]) -> list[SequenceInfo]:
    """返回测试被试的序列，仅供 ``evaluate.py`` 最终评估。"""
    if not test_subjects:
        return []
    return filter_by_subjects(sequences, test_subjects)


def summarize_by_subject(sequences: list[SequenceInfo]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for seq in sequences:
        counts[seq.subject_id] += 1
    return dict(counts)
