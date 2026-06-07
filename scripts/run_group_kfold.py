from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.io import discover_sequences
from data.split import group_kfold_folds
from training.seed import set_seed
from training.trainer import train_fold
from utils.config import load_config
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按受试者分组 K 折训练/评估 ADFNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = setup_logger(cfg["training"]["output_dir"], name="adfnet_groupkfold")
    sequences = discover_sequences(cfg["data"]["root"])
    folds = group_kfold_folds(sequences, n_splits=args.n_splits, seed=cfg["seed"])
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    rows = []
    for fold in folds:
        train_dataset = ADFWindowDataset(sequences=fold.train, **cfg["data"])
        test_dataset = ADFWindowDataset(sequences=fold.test, **cfg["data"])
        logger.info("%s: train windows=%d, test windows=%d", fold.name, len(train_dataset), len(test_dataset))
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            logger.warning("%s 样本为空，跳过", fold.name)
            continue
        metrics = train_fold(cfg, train_dataset, test_dataset, fold.name)
        rows.append({"fold": fold.name, **metrics})
    output = Path(cfg["training"]["output_dir"]) / "group_kfold_metrics.csv"
    save_fold_metrics(rows, output)
    logger.info("GroupKFold 结果已保存：%s", output)


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
