from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _to_jsonable(obj: Any) -> Any:
    """递归把对象转成 JSON 可序列化类型（Path→str、tuple→list，其余原样）。"""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return out or None
    except Exception:
        return None


def save_hparams(
    cfg: dict,
    output_dir: str | Path,
    *,
    script: str,
    task_mode: str,
    timestamp: str,
    extra: dict | None = None,
) -> Path:
    """把本次训练的有效超参数以 JSON 写入 ``<output_dir>/hparams.json``。

    内容包括：脚本名、task_mode、时间戳、seed、git commit、Python 版本、
    完整 config（含运行时已生效的 output_dir 等），以及调用方通过 ``extra``
    传入的运行级元信息（如 n_folds、total_subjects、test_subjects、各 fold 的
    val 被试等）。LOSO 与 GroupKFold 实验均会在各自输出目录写一份，便于后续
    对比/消融实验时查看每次用了哪些超参数。
    """
    record: dict[str, Any] = {
        "script": script,
        "task_mode": task_mode,
        "timestamp": timestamp,
        "seed": cfg.get("seed"),
        "exp_name": cfg.get("exp_name"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "config": _to_jsonable(cfg),
    }
    if extra:
        for key, value in extra.items():
            if str(key).startswith("_"):
                continue
            record[key] = _to_jsonable(value)
    out = Path(output_dir) / "hparams.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    return out
