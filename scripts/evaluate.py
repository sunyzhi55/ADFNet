from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.io import discover_sequences, filter_sequences_by_task
from training.trainer import evaluate_checkpoint
from utils.config import load_config
from utils.logging import setup_logger
from datetime import datetime
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ADFNet checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None, help="override test data root")
    parser.add_argument("--task-mode", choices=["all", "easy", "hard"], default=None)
    parser.add_argument("--output-csv", default=None, help="where to save evaluation metrics")
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
    timestamp = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    cfg["training"]["output_dir"] = f"{cfg['training']['output_dir']}_{timestamp}_{cfg['exp_name']}/"
    Path(cfg["training"]["output_dir"]).mkdir(exist_ok=True, parents=True)
    cfg["exp_name"] = f"{cfg['exp_name']}_eval"
    logger = setup_logger(cfg["training"]["output_dir"], name=cfg["exp_name"])
    task_mode = task_mode_from_args(cfg, args.task_mode)
    data_root = args.data_root or cfg["data"]["root"]
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
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_csv, index=False)
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Evaluation metrics saved to %s", output_csv)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Total evaluation time: {hours}h {minutes}m {seconds}s")
