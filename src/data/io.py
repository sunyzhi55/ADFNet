from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FILENAME_RE = re.compile(
    r"^(?P<subject>.+)_(?P<task>easy|hard)_(?P<label>alert|sleep|sleepy)\.jsonl$",
    re.I,
)
LABEL_MAP = {"alert": 0, "sleep": 1, "sleepy": 1}


@dataclass(frozen=True)
class SequenceInfo:
    path: Path
    subject_id: str
    task_type: str
    label_name: str
    label: int


def parse_sequence_filename(path: str | Path) -> SequenceInfo:
    path = Path(path)
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(
            f"Invalid file name, expected [subject_id]_[easy|hard]_[alert|sleep].jsonl: {path.name}"
        )
    task_type = match.group("task").lower()
    label_name = match.group("label").lower()
    return SequenceInfo(
        path=path,
        subject_id=match.group("subject"),
        task_type=task_type,
        label_name=label_name,
        label=LABEL_MAP[label_name],
    )


def discover_sequences(root: str | Path) -> list[SequenceInfo]:
    root = Path(root)
    if not root.exists():
        return []
    sequences = [parse_sequence_filename(path) for path in root.rglob("*.jsonl")]
    return sorted(
        sequences,
        key=lambda item: (item.subject_id, item.task_type, item.label_name, str(item.path)),
    )


def filter_sequences_by_task(sequences: list[SequenceInfo], task_mode: str = "all") -> list[SequenceInfo]:
    task_mode = task_mode.lower()
    if task_mode in {"all", "both", "easy+hard"}:
        return list(sequences)
    if task_mode not in {"easy", "hard"}:
        raise ValueError("task_mode must be one of: all, easy, hard")
    return [seq for seq in sequences if seq.task_type == task_mode]


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not a valid JSONL line") from exc


def parse_points(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    if all(isinstance(v, (int, float)) for v in value):
        if len(value) < 2:
            return []
        if len(value) % 2 != 0:
            value = value[:-1]
        return [[float(value[i]), float(value[i + 1])] for i in range(0, len(value), 2)]

    points: list[list[float]] = []
    for item in value:
        if isinstance(item, str):
            try:
                item = ast.literal_eval(item)
            except (SyntaxError, ValueError):
                continue
        if isinstance(item, tuple):
            item = list(item)
        if isinstance(item, list) and len(item) >= 2:
            try:
                points.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
    return points
