from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.io import discover_sequences, filter_sequences_by_task
from data.split import test_sequences
from training.trainer import evaluate_checkpoint
from utils.config import load_config
from utils.logging import setup_logger
from datetime import datetime
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ADFNet checkpoints")
    parser.add_argument("--config", default="configs/default.yaml")
    # 单 checkpoint 模式：
    parser.add_argument("--checkpoint", default=None, help="单个 checkpoint 路径（单模型评估）")
    # 批量模式：
    parser.add_argument("--run-dir", default=None,
                        help="训练输出目录（含各 fold 子目录的 best.pt）；批量在测试集上评估")
    parser.add_argument("--data-root", default=None, help="override test data root")
    parser.add_argument("--task-mode", choices=["all", "easy", "hard"], default=None)
    parser.add_argument("--output-csv", default=None, help="where to save evaluation metrics")
    parser.add_argument("--fold-glob", default="*/best.pt", help="批量模式下匹配 checkpoint 的 glob（默认 */best.pt）")
    return parser.parse_args()


def task_mode_from_args(cfg: dict, task_mode: str | None) -> str:
    return task_mode or cfg.get("data", {}).get("task_mode", "all")


def dataset_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg["data"])
    kwargs.pop("root", None)
    kwargs.pop("task_mode", None)
    return kwargs


def split_cfg(cfg: dict) -> dict:
    return cfg.get("split", {}) or {}


def save_fold_metrics(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        numeric = frame.select_dtypes(include="number")
        mean = {"fold": "mean", **numeric.mean(numeric_only=True).to_dict()}
        std = {"fold": "std", **numeric.std(numeric_only=True).to_dict()}
        frame = pd.concat([frame, pd.DataFrame([mean, std])], ignore_index=True)
    frame.to_csv(output, index=False)


def run_single(cfg: dict, args, task_mode: str, data_root: str, logger) -> None:
    if not args.checkpoint:
        raise SystemExit("Single mode requires --checkpoint")
    sequences = filter_sequences_by_task(discover_sequences(data_root), task_mode)
    dataset = ADFWindowDataset(sequences=sequences, **dataset_kwargs(cfg))
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty. Check data_root, task_mode, window_size and stride.")
    metrics = evaluate_checkpoint(cfg, dataset, args.checkpoint)
    row = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(data_root),
        "task_mode": task_mode,
        "windows": len(dataset),
        **metrics,
    }
    output_csv = Path(args.output_csv) if args.output_csv else Path(cfg["training"]["output_dir"]) / f"eval_metrics_{task_mode}.csv"
    save_fold_metrics([row], output_csv)
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Evaluation metrics saved to %s", output_csv)


def run_batch(cfg: dict, args, task_mode: str, data_root: str, logger) -> None:
    """读取 run_dir 下每个 fold 的 best.pt，在测试集上批量评估。

    要求配置 ``split.test_subjects`` 非空：测试被试在训练时已被 hold-out，
    此处构建一次测试集，对每个 fold 的模型（含其各自的 gamma 参考/归一化）分别评估。
    """
    test_subjects = split_cfg(cfg).get("test_subjects", []) or []
    if not test_subjects:
        raise SystemExit(
            "Batch mode (--run-dir) requires split.test_subjects in config. "
            "Add a 'split: {test_subjects: [..]}' section or use single mode (--checkpoint)."
        )
    run_dir = Path(args.run_dir)
    ckpts = sorted(run_dir.glob(args.fold_glob))
    if not ckpts:
        raise SystemExit(f"No checkpoints matching '{args.fold_glob}' under {run_dir}")

    sequences = filter_sequences_by_task(discover_sequences(data_root), task_mode)
    test_seqs = test_sequences(sequences, test_subjects)
    if not test_seqs:
        raise SystemExit(f"No test sequences for test_subjects={test_subjects} (task_mode={task_mode})")
    # 测试集只构建一次；evaluate_checkpoint 每次会按该 fold 的 gamma/归一化重新 attach。
    test_dataset = ADFWindowDataset(sequences=test_seqs, **dataset_kwargs(cfg))
    logger.info("Batch eval: %d checkpoints, test_subjects=%s, test windows=%d",
                len(ckpts), list(test_subjects), len(test_dataset))

    rows = []
    for ckpt in ckpts:
        fold_name = ckpt.parent.name
        try:
            metrics = evaluate_checkpoint(cfg, test_dataset, ckpt)
        except Exception as exc:  # 单 fold 失败不阻断其余
            logger.warning("Evaluate %s failed: %s", fold_name, exc)
            continue
        logger.info("%s: %s", fold_name, metrics)
        rows.append({"fold": fold_name, "checkpoint": str(ckpt), "task_mode": task_mode,
                     "windows": len(test_dataset), **metrics})

    output_csv = Path(args.output_csv) if args.output_csv else run_dir / f"test_metrics_{task_mode}.csv"
    save_fold_metrics(rows, output_csv)
    logger.info("Batch test metrics saved to %s", output_csv)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    timestamp = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    cfg["training"]["output_dir"] = f"{cfg['training']['output_dir']}_{timestamp}_{cfg['exp_name']}/"
    Path(cfg["training"]["output_dir"]).mkdir(exist_ok=True, parents=True)
    cfg["exp_name"] = f"{cfg['exp_name']}_eval"
    logger = setup_logger(cfg["training"]["output_dir"], name=cfg["exp_name"])
    task_mode = task_mode_from_args(cfg, args.task_mode)
    data_root = args.data_root or cfg["data"]["root"]

    if args.run_dir:
        run_batch(cfg, args, task_mode, data_root, logger)
    else:
        run_single(cfg, args, task_mode, data_root, logger)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Total evaluation time: {hours}h {minutes}m {seconds}s")
