from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.gaipat_dataset import GaipatWindowDataset, discover_gaipat_sequences
from data.io import discover_sequences, filter_sequences_by_task
from data.split import explicit_folds, group_kfold_folds, test_sequences
from training.seed import set_seed
from training.trainer import train_fold, evaluate_checkpoint
from utils.config import load_config, save_hparams
from utils.logging import setup_logger
from datetime import datetime
import time

# ── eval_mode 常量 ──────────────────────────────────────────────
EVAL_FATIGUE = "fatigue"
EVAL_FATIGUE_TO_GAIPAT = "fatigue_to_gaipat"
EVAL_GAIPAT = "gaipat"
EVAL_GAIPAT_TO_FATIGUE = "gaipat_to_fatigue"
EVAL_MODES = [EVAL_FATIGUE, EVAL_FATIGUE_TO_GAIPAT, EVAL_GAIPAT, EVAL_GAIPAT_TO_FATIGUE]


def _parse_ablation_arg(values: list[str] | None) -> dict:
    if not values:
        return {}
    result: dict = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--ablation format error, expected key=value: {item}")
        key, raw = item.split("=", 1)
        key, raw = key.strip(), raw.strip()
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
                        help="Ablation overrides (key=value)")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--output-dir", default=None)
    # ── 跨数据集实验 ──
    parser.add_argument("--eval-mode", choices=EVAL_MODES, default=EVAL_FATIGUE,
                        help="fatigue: FG→FG | fatigue_to_gaipat: FG→GAIPAT | "
                             "gaipat: GAIPAT→GAIPAT | gaipat_to_fatigue: GAIPAT→FG")
    parser.add_argument("--gaipat-dir", default=None,
                        help="GAIPAT data root (overrides gaipat.root in config)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Skip training; load fold checkpoints from this dir for evaluation")
    return parser.parse_args()


def task_mode_from_args(cfg: dict, task_mode: str | None) -> str:
    return task_mode or cfg.get("data", {}).get("task_mode", "all")


def dataset_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg["data"])
    kwargs.pop("root", None)
    kwargs.pop("task_mode", None)
    return kwargs


def gaipat_dataset_kwargs(cfg: dict) -> dict:
    gaipat_cfg = cfg.get("gaipat", {})
    return {
        "window_size": gaipat_cfg.get("window_size", 256),
        "local_mean_size": gaipat_cfg.get("local_mean_size", cfg["data"].get("local_mean_size", 16)),
        "per_sample_norm": gaipat_cfg.get("per_sample_norm", True),
    }


def build_gaipat_dataset(cfg: dict, gaipat_dir: str) -> GaipatWindowDataset:
    sequences = discover_gaipat_sequences(gaipat_dir)
    if not sequences:
        raise RuntimeError(f"No valid GAIPAT sequences found under {gaipat_dir}")
    return GaipatWindowDataset(sequences=sequences, **gaipat_dataset_kwargs(cfg))


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    eval_mode = args.eval_mode

    # CLI 覆盖
    if args.exp_name:
        cfg["exp_name"] = args.exp_name
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.ablation:
        abl = dict(cfg.get("ablation") or {})
        abl.update(_parse_ablation_arg(args.ablation))
        cfg["ablation"] = abl

    gaipat_dir = args.gaipat_dir or cfg.get("gaipat", {}).get("root", "")

    # ── 确定数据源 ──
    train_on_gaipat = eval_mode in (EVAL_GAIPAT, EVAL_GAIPAT_TO_FATIGUE)
    eval_on_gaipat = eval_mode in (EVAL_FATIGUE_TO_GAIPAT, EVAL_GAIPAT)
    is_cross_dataset = eval_mode in (EVAL_FATIGUE_TO_GAIPAT, EVAL_GAIPAT_TO_FATIGUE)

    if train_on_gaipat:
        task_mode = "all"
    else:
        task_mode = task_mode_from_args(cfg, args.task_mode)

    timestamp = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    cfg["training"]["output_dir"] = f"{cfg['training']['output_dir']}_{timestamp}_{cfg['exp_name']}/"
    Path(cfg["training"]["output_dir"]).mkdir(exist_ok=True, parents=True)
    cfg["exp_name"] = f"{cfg['exp_name']}_groupkfold_{eval_mode}"
    logger = setup_logger(cfg["training"]["output_dir"], name=cfg["exp_name"])
    logger.info("eval_mode=%s | train_on_gaipat=%s | eval_on_gaipat=%s | task_mode=%s",
                eval_mode, train_on_gaipat, eval_on_gaipat, task_mode)

    # ── 准备数据 ──
    if train_on_gaipat:
        sequences = discover_gaipat_sequences(gaipat_dir)
        logger.info("GAIPAT: found %d sequences", len(sequences))
        if not sequences:
            raise RuntimeError(f"No GAIPAT sequences under {gaipat_dir}")
        # GAIPAT 无 explicit_folds / test_subjects
        folds = group_kfold_folds(sequences, n_splits=args.n_splits, seed=cfg["seed"])
    else:
        sequences = filter_sequences_by_task(discover_sequences(cfg["data"]["root"]), task_mode)
        logger.info("FatigueGuard: found %d sequences for task_mode=%s", len(sequences), task_mode)
        test_subjects = split_cfg(cfg).get("test_subjects", []) or []
        explicit = split_cfg(cfg).get("explicit_folds")
        if explicit:
            folds = explicit_folds(sequences, explicit, test_subjects)
            logger.info("Using explicit_folds: %d folds", len(folds))
        else:
            folds = group_kfold_folds(sequences, n_splits=args.n_splits,
                                      seed=cfg["seed"], test_subjects=test_subjects)
            logger.info("GroupKFold(n_splits=%d): %d folds", args.n_splits, len(folds))

    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    total_subjects = len({seq.subject_id for seq in sequences})
    save_hparams(
        cfg, cfg["training"]["output_dir"],
        script="run_group_kfold.py", task_mode=task_mode, timestamp=timestamp,
        extra={
            "eval_mode": eval_mode,
            "train_on_gaipat": train_on_gaipat,
            "eval_on_gaipat": eval_on_gaipat,
            "gaipat_dir": gaipat_dir,
            "total_subjects": total_subjects,
            "n_folds": len(folds),
            "n_splits": args.n_splits,
            "max_folds": args.max_folds,
            "val_subjects_per_fold": [list(f.val_subjects) for f in folds],
        },
    )
    logger.info("Hyperparameters saved to %s", Path(cfg["training"]["output_dir"]) / "hparams.json")

    # ── 训练 + 同源评估 ──
    rows = []
    data_kwargs = dataset_kwargs(cfg)
    skip_training = args.checkpoint_dir is not None

    for fold in folds:
        fold_output_name = f"{fold.name}_{task_mode}"

        if not skip_training:
            if train_on_gaipat:
                g_kwargs = gaipat_dataset_kwargs(cfg)
                train_dataset = GaipatWindowDataset(sequences=fold.train, **g_kwargs)
                val_dataset = GaipatWindowDataset(sequences=fold.val, **g_kwargs)
            else:
                train_dataset = ADFWindowDataset(sequences=fold.train, **data_kwargs)
                val_dataset = ADFWindowDataset(sequences=fold.val, **data_kwargs)

            logger.info("%s: val_subjects=%s  train=%d, val=%d",
                        fold.name, list(fold.val_subjects), len(train_dataset), len(val_dataset))
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                logger.warning("%s has empty samples, skipped", fold.name)
                continue

            metrics = train_fold(cfg, train_dataset, val_dataset, fold_output_name)
            rows.append({"fold": fold.name, "val_subjects": ",".join(fold.val_subjects),
                         "task_mode": task_mode, **metrics})
        else:
            logger.info("%s: skipping training, will load checkpoint from %s",
                        fold.name, args.checkpoint_dir)

    if rows:
        output = Path(cfg["training"]["output_dir"]) / f"group_kfold_metrics_{task_mode}.csv"
        save_fold_metrics(rows, output)
        logger.info("GroupKFold metrics saved to %s", output)

    # ── 跨数据集评估 ──
    if is_cross_dataset:
        cross_rows = _run_cross_eval(
            cfg=cfg,
            folds=folds,
            task_mode=task_mode,
            eval_on_gaipat=eval_on_gaipat,
            gaipat_dir=gaipat_dir,
            output_dir=cfg["training"]["output_dir"],
            checkpoint_dir=args.checkpoint_dir,
            logger=logger,
        )
        if cross_rows:
            target_name = "gaipat" if eval_on_gaipat else "fatigue"
            cross_output = Path(cfg["training"]["output_dir"]) / f"kfold_cross_to_{target_name}_{task_mode}.csv"
            save_fold_metrics(cross_rows, cross_output)
            logger.info("Cross-dataset metrics saved to %s", cross_output)


def _run_cross_eval(
    cfg: dict,
    folds: list,
    task_mode: str,
    eval_on_gaipat: bool,
    gaipat_dir: str,
    output_dir: str,
    checkpoint_dir: str | None,
    logger,
) -> list[dict]:
    """跨数据集评估：加载每个 fold 的 best.pt，在目标数据集上评估。"""
    if eval_on_gaipat:
        test_dataset = build_gaipat_dataset(cfg, gaipat_dir)
        logger.info("Cross-eval target: GAIPAT (%d windows)", len(test_dataset))
    else:
        data_kwargs = dataset_kwargs(cfg)
        target_sequences = filter_sequences_by_task(
            discover_sequences(cfg["data"]["root"]), task_mode
        )
        test_dataset = ADFWindowDataset(sequences=target_sequences, **data_kwargs)
        logger.info("Cross-eval target: FatigueGuard (%d windows, task_mode=%s)",
                    len(test_dataset), task_mode)

    if len(test_dataset) == 0:
        logger.warning("Cross-eval target dataset is empty, skipping")
        return []

    ckpt_base = Path(checkpoint_dir) if checkpoint_dir else Path(output_dir)
    cross_rows = []
    for fold in folds:
        ckpt_path = ckpt_base / f"{fold.name}_{task_mode}" / "best.pt"
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s, skipping", ckpt_path)
            continue
        try:
            metrics = evaluate_checkpoint(cfg, test_dataset, ckpt_path)
            logger.info("Cross-eval %s: %s", fold.name, metrics)
            cross_rows.append({
                "fold": fold.name,
                "val_subjects": ",".join(fold.val_subjects),
                "task_mode": task_mode,
                "target": "gaipat" if eval_on_gaipat else "fatigue",
                **metrics,
            })
        except Exception as exc:
            logger.warning("Cross-eval %s failed: %s", fold.name, exc)

    return cross_rows


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
