from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Train ADFNet")
    parser.add_argument("--config", default="configs/default.yaml", help="config path")
    parser.add_argument("--task-mode", choices=["all", "easy", "hard"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="check data/config only")
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
    logger = setup_logger(cfg["training"]["output_dir"])
    task_mode = task_mode_from_args(cfg, args.task_mode)
    sequences = filter_sequences_by_task(discover_sequences(cfg["data"]["root"]), task_mode)
    logger.info("Found %d JSONL sequences for task_mode=%s", len(sequences), task_mode)
    if not sequences:
        logger.warning("No data found. Check data.root and task_mode.")
        return
    data_kwargs = dataset_kwargs(cfg)
    if args.dry_run:
        dataset = ADFWindowDataset(sequences=sequences, **data_kwargs)
        logger.info("dry-run done: windows=%d", len(dataset))
        return
    folds = group_kfold_folds(sequences, n_splits=2, seed=cfg["seed"])
    fold = folds[0]
    train_dataset = ADFWindowDataset(sequences=fold.train, **data_kwargs)
    val_dataset = ADFWindowDataset(sequences=fold.test, **data_kwargs)
    logger.info("Training fold=%s, train=%d, val=%d", fold.name, len(train_dataset), len(val_dataset))
    metrics = train_fold(cfg, train_dataset, val_dataset, f"{fold.name}_{task_mode}")
    logger.info("Training finished: %s", metrics)


if __name__ == "__main__":
    main()
