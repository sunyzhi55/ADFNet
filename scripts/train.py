from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="训练 ADFNet")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据与配置，不启动训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = setup_logger(cfg["training"]["output_dir"])
    sequences = discover_sequences(cfg["data"]["root"])
    logger.info("发现 %d 个 JSONL 序列", len(sequences))
    if not sequences:
        logger.warning("未发现数据。请将 JSONL 放入 data.root 指定目录后再训练。")
        return
    if args.dry_run:
        dataset = ADFWindowDataset(sequences=sequences, **cfg["data"])
        logger.info("dry-run 完成：窗口样本数=%d", len(dataset))
        return
    folds = group_kfold_folds(sequences, n_splits=2, seed=cfg["seed"])
    fold = folds[0]
    train_dataset = ADFWindowDataset(sequences=fold.train, **cfg["data"])
    val_dataset = ADFWindowDataset(sequences=fold.test, **cfg["data"])
    logger.info("训练 fold=%s, train=%d, val=%d", fold.name, len(train_dataset), len(val_dataset))
    metrics = train_fold(cfg, train_dataset, val_dataset, fold.name)
    logger.info("完成训练：%s", metrics)


if __name__ == "__main__":
    main()
