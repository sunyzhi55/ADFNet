from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from training.trainer import evaluate_checkpoint
from utils.config import load_config
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 ADFNet checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger(cfg["training"]["output_dir"], name="adfnet_eval")
    data_cfg = dict(cfg["data"])
    data_root = data_cfg.pop("root")
    dataset = ADFWindowDataset(root=data_root, **data_cfg)
    if len(dataset) == 0:
        raise RuntimeError("评估集为空，请检查 data.root、window_size 和 stride")
    metrics = evaluate_checkpoint(cfg, dataset, args.checkpoint)
    logger.info("评估结果：%s", metrics)


if __name__ == "__main__":
    main()
