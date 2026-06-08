from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from data.io import SequenceInfo


@dataclass(frozen=True)
class SubjectFold:
    name: str
    train: list[SequenceInfo]
    test: list[SequenceInfo]


def subject_ids(sequences: Iterable[SequenceInfo]) -> list[str]:
    return sorted({seq.subject_id for seq in sequences})


def loso_folds(sequences: list[SequenceInfo]) -> list[SubjectFold]:
    subjects = subject_ids(sequences)
    folds: list[SubjectFold] = []
    for subject in subjects:
        train = [seq for seq in sequences if seq.subject_id != subject]
        test = [seq for seq in sequences if seq.subject_id == subject]
        folds.append(SubjectFold(name=f"loso_{subject}", train=train, test=test))
    return folds


def group_kfold_folds(
    sequences: list[SequenceInfo],
    n_splits: int = 5,
    seed: int = 42,
) -> list[SubjectFold]:
    subjects = np.asarray(subject_ids(sequences), dtype=object)
    if n_splits < 2 or n_splits > len(subjects):
        raise ValueError("n_splits must be in [2, subject_count]")
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    groups = np.array_split(shuffled, n_splits)
    folds: list[SubjectFold] = []
    for idx, group in enumerate(groups, start=1):
        test_subjects = set(group.tolist())
        train = [seq for seq in sequences if seq.subject_id not in test_subjects]
        test = [seq for seq in sequences if seq.subject_id in test_subjects]
        folds.append(SubjectFold(name=f"groupkfold_{idx}", train=train, test=test))
    return folds


def summarize_by_subject(sequences: list[SequenceInfo]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for seq in sequences:
        counts[seq.subject_id] += 1
    return dict(counts)
