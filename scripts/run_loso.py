from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.io import discover_sequences, filter_sequences_by_task
from data.split import loso_folds
from training.seed import set_seed
from training.trainer import train_fold
from utils.config import load_config, save_hparams
from utils.logging import setup_logger
from datetime import datetime
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-wise LOSO training/evaluation for ADFNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-mode", choices=["all", "easy", "hard"], default=None)
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

    timestamp = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    cfg["training"]["output_dir"] = f"{cfg['training']['output_dir']}_{timestamp}_{cfg['exp_name']}/"
    Path(cfg["training"]["output_dir"]).mkdir(exist_ok=True, parents=True)
    cfg["exp_name"] = f"{cfg['exp_name']}_loso"

    logger = setup_logger(cfg["training"]["output_dir"], name=cfg["exp_name"])
    
    task_mode = task_mode_from_args(cfg, args.task_mode)
    sequences = filter_sequences_by_task(discover_sequences(cfg["data"]["root"]), task_mode)
    logger.info("Found %d JSONL sequences for task_mode=%s", len(sequences), task_mode)
    folds = loso_folds(sequences)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    total_subjects = len({seq.subject_id for seq in sequences})
    save_hparams(
        cfg,
        cfg["training"]["output_dir"],
        script="run_loso.py",
        task_mode=task_mode,
        timestamp=timestamp,
        extra={
            "total_subjects": total_subjects,
            "n_folds": len(folds),
            "max_folds": args.max_folds,
            "val_subjects_per_fold": [list(f.val_subjects) for f in folds],
        },
    )
    logger.info("Hyperparameters saved to %s", Path(cfg["training"]["output_dir"]) / "hparams.json")
    rows = []
    data_kwargs = dataset_kwargs(cfg)
    for fold in folds:
        train_dataset = ADFWindowDataset(sequences=fold.train, **data_kwargs)
        val_dataset = ADFWindowDataset(sequences=fold.val, **data_kwargs)
        logger.info("%s: train windows=%d, val windows=%d", fold.name, len(train_dataset), len(val_dataset))
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.warning("%s has empty samples, skipped", fold.name)
            continue
        metrics = train_fold(cfg, train_dataset, val_dataset, f"{fold.name}_{task_mode}")
        rows.append({"fold": fold.name, "val_subjects": ",".join(fold.val_subjects),
                     "task_mode": task_mode, **metrics})
    output = Path(cfg["training"]["output_dir"]) / f"loso_metrics_{task_mode}.csv"
    save_fold_metrics(rows, output)
    logger.info("LOSO metrics saved to %s", output)


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
    start_time = time.time()
    main()
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
