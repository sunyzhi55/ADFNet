from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.io import discover_sequences, filter_sequences_by_task
from data.split import explicit_folds, group_kfold_folds, test_sequences
from training.seed import set_seed
from training.trainer import train_fold
from utils.config import load_config, save_hparams
from utils.logging import setup_logger
from datetime import datetime
import time


def _parse_ablation_arg(values: list[str] | None) -> dict:
    """将 CLI '--ablation key=value ...' 解析为 dict。

    布尔值自动识别：true/false/yes/no/1/0。
    """
    if not values:
        return {}
    result: dict = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--ablation 参数格式错误，期望 key=value: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.lower() in ("true", "yes", "1"):
            result[key] = True
        elif raw.lower() in ("false", "no", "0"):
            result[key] = False
        else:
            try:
                result[key] = int(raw)
            except ValueError:
                result[key] = raw
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-wise GroupKFold training/evaluation for ADFNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-mode", choices=["all", "easy", "hard"], default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--ablation", nargs="+", default=None,
                        help="消融覆盖，形如 key=value (例: enable_grl=false enable_diff=false)")
    parser.add_argument("--exp-name", default=None, help="覆盖 exp_name")
    parser.add_argument("--output-dir", default=None, help="覆盖 training.output_dir")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    # CLI 覆盖
    if args.exp_name:
        cfg["exp_name"] = args.exp_name
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.ablation:
        abl = dict(cfg.get("ablation") or {})
        abl.update(_parse_ablation_arg(args.ablation))
        cfg["ablation"] = abl

    timestamp = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    cfg["training"]["output_dir"] = f"{cfg['training']['output_dir']}_{timestamp}_{cfg['exp_name']}/"
    Path(cfg["training"]["output_dir"]).mkdir(exist_ok=True, parents=True)
    cfg["exp_name"] = f"{cfg['exp_name']}_groupkfold"
    logger = setup_logger(cfg["training"]["output_dir"], name=cfg["exp_name"])
    
    task_mode = task_mode_from_args(cfg, args.task_mode)
    sequences = filter_sequences_by_task(discover_sequences(cfg["data"]["root"]), task_mode)
    logger.info("Found %d JSONL sequences for task_mode=%s", len(sequences), task_mode)

    test_subjects = split_cfg(cfg).get("test_subjects", []) or []
    explicit = split_cfg(cfg).get("explicit_folds")
    if test_subjects:
        test_seqs = test_sequences(sequences, test_subjects)
        logger.info("Test subjects %s held out: %d sequences (evaluate later with evaluate.py --run-dir)",
                    list(test_subjects), len(test_seqs))

    if explicit:
        folds = explicit_folds(sequences, explicit, test_subjects)
        logger.info("Using explicit_folds from config: %d folds", len(folds))
    else:
        folds = group_kfold_folds(sequences, n_splits=args.n_splits, seed=cfg["seed"], test_subjects=test_subjects)
        logger.info("GroupKFold(n_splits=%d) over non-test subjects: %d folds", args.n_splits, len(folds))

    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    total_subjects = len({seq.subject_id for seq in sequences})
    save_hparams(
        cfg,
        cfg["training"]["output_dir"],
        script="run_group_kfold.py",
        task_mode=task_mode,
        timestamp=timestamp,
        extra={
            "total_subjects": total_subjects,
            "n_folds": len(folds),
            "n_splits": args.n_splits,
            "max_folds": args.max_folds,
            "test_subjects": list(test_subjects),
            "explicit_folds": explicit,
            "val_subjects_per_fold": [list(f.val_subjects) for f in folds],
        },
    )
    logger.info("Hyperparameters saved to %s", Path(cfg["training"]["output_dir"]) / "hparams.json")
    rows = []
    data_kwargs = dataset_kwargs(cfg)
    for fold in folds:
        train_dataset = ADFWindowDataset(sequences=fold.train, **data_kwargs)
        val_dataset = ADFWindowDataset(sequences=fold.val, **data_kwargs)
        logger.info("%s: val_subjects=%s  train windows=%d, val windows=%d",
                    fold.name, list(fold.val_subjects), len(train_dataset), len(val_dataset))
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.warning("%s has empty samples, skipped", fold.name)
            continue
        metrics = train_fold(cfg, train_dataset, val_dataset, f"{fold.name}_{task_mode}")
        rows.append({"fold": fold.name, "val_subjects": ",".join(fold.val_subjects),
                     "task_mode": task_mode, **metrics})
    output = Path(cfg["training"]["output_dir"]) / f"group_kfold_metrics_{task_mode}.csv"
    save_fold_metrics(rows, output)
    logger.info("GroupKFold metrics saved to %s", output)
    if test_subjects:
        logger.info("Test set held out. Run batch evaluation:\n  python scripts/evaluate.py "
                    "--config %s --run-dir %s --task-mode %s", args.config,
                    cfg["training"]["output_dir"], task_mode)


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
