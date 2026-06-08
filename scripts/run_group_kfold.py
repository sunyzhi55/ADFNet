from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.io import discover_sequences, filter_sequences_by_task
from data.split import group_kfold_folds
from training.seed import set_seed
from training.trainer import train_fold
from utils.config import load_config
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-wise GroupKFold training/evaluation for ADFNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-mode", choices=["all", "easy", "hard"], default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def task_mode_from_args(cfg: dict, task_mode: str | None) -> str:
    return task_mode or cfg.get("data", {}).get("task_mode", "all")


def dataset_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg["data"])
    kwargs.pop("root", None)
    kwargs.pop("task_mode", None)
    return kwargs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = setup_logger(cfg["training"]["output_dir"], name="adfnet_groupkfold")
    task_mode = task_mode_from_args(cfg, args.task_mode)
    sequences = filter_sequences_by_task(discover_sequences(cfg["data"]["root"]), task_mode)
    logger.info("Found %d JSONL sequences for task_mode=%s", len(sequences), task_mode)
    folds = group_kfold_folds(sequences, n_splits=args.n_splits, seed=cfg["seed"])
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    rows = []
    data_kwargs = dataset_kwargs(cfg)
    for fold in folds:
        train_dataset = ADFWindowDataset(sequences=fold.train, **data_kwargs)
        test_dataset = ADFWindowDataset(sequences=fold.test, **data_kwargs)
        logger.info("%s: train windows=%d, test windows=%d", fold.name, len(train_dataset), len(test_dataset))
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            logger.warning("%s has empty samples, skipped", fold.name)
            continue
        metrics = train_fold(cfg, train_dataset, test_dataset, f"{fold.name}_{task_mode}")
        rows.append({"fold": fold.name, "task_mode": task_mode, **metrics})
    output = Path(cfg["training"]["output_dir"]) / f"group_kfold_metrics_{task_mode}.csv"
    save_fold_metrics(rows, output)
    logger.info("GroupKFold metrics saved to %s", output)


def save_fold_metrics(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        numeric = frame.select_dtypes(include="number")
        mean = {"fold": "mean", **numeric.mean(numeric_only=True).to_dict()}
        std = {"fold": "std", **numeric.std(numeric_only=True).to_dict()}
        frame = pd.concat([frame, pd.DataFrame([mean, std])], ignore_index=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    main()
