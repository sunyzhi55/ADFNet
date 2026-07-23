"""
消融实验启动器 - ADFNet Ablation Study Launcher
================================================

以 ``configs/default.yaml`` 为基线，系统性地禁用/替换各组件来衡量其贡献。

消融维度（6 个二值开关 -> 2^6 = 64 种组合）::

    enable_gamma        - Gamma 分布对齐流（DistributionBranch + GammaReference）
    enable_grl          - GRL 梯度反转 + 身份对抗判别器
    enable_diff         - 一阶差分通道
    enable_sliding_mean - 滑动均值通道
    enable_soft_dtw     - Soft-DTW 距离（分布特征 3->2）
    enable_mamba        - Mamba-MLA 时序编码器

替换实验（独立运行，不参与组合遍历）::

    temporal_encoder: lstm        - 用 LSTM 替换 Mamba-MLA
    temporal_encoder: transformer - 用 Transformer 替换 Mamba-MLA
    reference_distribution: gaussian  - 用高斯分布替换 Gamma 分布
    reference_distribution: kde       - 用核密度估计替换 Gamma 分布
    reference_distribution: lognormal - 用对数正态分布替换 Gamma 分布
    reference_distribution: weibull   - 用 Weibull 分布替换 Gamma 分布
    reference_distribution: rayleigh  - 用 Rayleigh 分布替换 Gamma 分布

用法示例::

    # 完整模型基线（kfold + loso, easy + hard）
    python scripts/run_ablation.py --preset full

    # 单个消融：去掉 GRL
    python scripts/run_ablation.py --preset no_grl

    # 全部 64 种组合
    python scripts/run_ablation.py --preset all_combinations

    # LSTM 替换（仅 kfold + easy）
    python scripts/run_ablation.py --preset lstm --cv kfold --task-mode easy

    # 指定 GPU
    python scripts/run_ablation.py --preset all_combinations --device cuda:1
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset
from data.gaipat_dataset import GaipatWindowDataset, discover_gaipat_sequences
from data.io import discover_sequences, filter_sequences_by_task
from data.split import explicit_folds, group_kfold_folds, loso_folds
from training.seed import set_seed
from training.trainer import train_fold, evaluate_checkpoint
from utils.config import load_config, save_hparams
from utils.logging import setup_logger

# ── eval_mode 常量 ──
EVAL_FATIGUE = "fatigue"
EVAL_FATIGUE_TO_GAIPAT = "fatigue_to_gaipat"
EVAL_GAIPAT = "gaipat"
EVAL_GAIPAT_TO_FATIGUE = "gaipat_to_fatigue"
EVAL_MODES = [EVAL_FATIGUE, EVAL_FATIGUE_TO_GAIPAT, EVAL_GAIPAT, EVAL_GAIPAT_TO_FATIGUE]

# ══════════════════════════════════════════════════════════════
# 消融组件定义
# ══════════════════════════════════════════════════════════════

#: 参与组合遍历的 6 个二值开关
COMPONENTS = (
    "enable_gamma",
    "enable_grl",
    "enable_diff",
    "enable_sliding_mean",
    "enable_soft_dtw",
    "enable_mamba",
)

#: 预设名称 → ablation 覆盖值；None 表示"全组合"或"替换实验"
PRESETS: dict[str, dict | None] = {
    # ── 基线 ──
    "full": {},
    # ── 单独消融（每次只去掉一个组件）──
    "no_gamma":        {"enable_gamma": False},
    "no_grl":          {"enable_grl": False},
    "no_diff":         {"enable_diff": False},
    "no_sliding_mean": {"enable_sliding_mean": False},
    "no_soft_dtw":     {"enable_soft_dtw": False},
    "no_mamba":        {"enable_mamba": False},
    # ── 特殊 ──
    "all_combinations": None,
    "lstm":            {"temporal_encoder": "lstm"},
    "transformer":     {"temporal_encoder": "transformer"},
    "gaussian":        {"reference_distribution": "gaussian"},
    "kde":             {"reference_distribution": "kde"},
    "lognormal":       {"reference_distribution": "lognormal"},
    "weibull":         {"reference_distribution": "weibull"},
    "rayleigh":        {"reference_distribution": "rayleigh"},
}


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def ablation_label(overrides: dict) -> str:
    """把 ablation 覆盖值转成人类可读的标签（用于目录名和 CSV）。"""
    if not overrides:
        return "full"
    parts: list[str] = []
    for key in COMPONENTS:
        if key in overrides and not overrides[key]:
            short = key.replace("enable_", "")
            parts.append(f"no_{short}")
    enc = overrides.get("temporal_encoder")
    if enc and enc != "mamba":
        parts.append(enc)
    dist = overrides.get("reference_distribution")
    if dist and dist != "gamma":
        parts.append(dist)
    return "_".join(parts) if parts else "full"


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """把消融覆盖写入 cfg["ablation"]，返回 cfg 自身以便链式调用。"""
    ablation = dict(cfg.get("ablation") or {})
    ablation.update(overrides)
    cfg["ablation"] = ablation
    return cfg


def generate_all_combinations() -> list[dict]:
    """生成 2^6 = 64 种二值组合。"""
    combos: list[dict] = []
    for bits in itertools.product([True, False], repeat=len(COMPONENTS)):
        combos.append(dict(zip(COMPONENTS, bits)))
    return combos


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


# ══════════════════════════════════════════════════════════════
# 核心训练流水线
# ══════════════════════════════════════════════════════════════

def run_loso(cfg: dict, sequences, task_mode: str, logger,
             max_folds: int | None = None, train_on_gaipat: bool = False) -> list[dict]:
    folds = loso_folds(sequences)
    if max_folds is not None:
        folds = folds[:max_folds]
    logger.info("LOSO: %d folds for task_mode=%s", len(folds), task_mode)
    d_kwargs = gaipat_dataset_kwargs(cfg) if train_on_gaipat else dataset_kwargs(cfg)
    DS = GaipatWindowDataset if train_on_gaipat else ADFWindowDataset
    rows: list[dict] = []
    for fold in folds:
        train_ds = DS(sequences=fold.train, **d_kwargs)
        val_ds = DS(sequences=fold.val, **d_kwargs)
        logger.info("%s: train=%d, val=%d", fold.name, len(train_ds), len(val_ds))
        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning("%s has empty samples, skipped", fold.name)
            continue
        metrics = train_fold(cfg, train_ds, val_ds, f"{fold.name}_{task_mode}")
        rows.append({"fold": fold.name, "val_subjects": ",".join(fold.val_subjects),
                      "task_mode": task_mode, **metrics})
    output = Path(cfg["training"]["output_dir"]) / f"loso_metrics_{task_mode}.csv"
    save_fold_metrics(rows, output)
    logger.info("LOSO metrics saved to %s", output)
    return rows


def run_kfold(cfg: dict, sequences, task_mode: str, logger,
              n_splits: int = 5, max_folds: int | None = None,
              train_on_gaipat: bool = False) -> list[dict]:
    if train_on_gaipat:
        folds = group_kfold_folds(sequences, n_splits=n_splits, seed=cfg["seed"])
        logger.info("GroupKFold(n_splits=%d) on GAIPAT: %d folds", n_splits, len(folds))
    else:
        test_subjects = split_cfg(cfg).get("test_subjects", []) or []
        explicit = split_cfg(cfg).get("explicit_folds")
        if explicit:
            folds = explicit_folds(sequences, explicit, test_subjects)
            logger.info("Using explicit_folds: %d folds", len(folds))
        else:
            folds = group_kfold_folds(sequences, n_splits=n_splits, seed=cfg["seed"],
                                      test_subjects=test_subjects)
            logger.info("GroupKFold(n_splits=%d): %d folds", n_splits, len(folds))
    if max_folds is not None:
        folds = folds[:max_folds]
    d_kwargs = gaipat_dataset_kwargs(cfg) if train_on_gaipat else dataset_kwargs(cfg)
    DS = GaipatWindowDataset if train_on_gaipat else ADFWindowDataset
    rows: list[dict] = []
    for fold in folds:
        train_ds = DS(sequences=fold.train, **d_kwargs)
        val_ds = DS(sequences=fold.val, **d_kwargs)
        logger.info("%s: val_subjects=%s  train=%d, val=%d",
                    fold.name, list(fold.val_subjects), len(train_ds), len(val_ds))
        if len(train_ds) == 0 or len(val_ds) == 0:
            logger.warning("%s has empty samples, skipped", fold.name)
            continue
        metrics = train_fold(cfg, train_ds, val_ds, f"{fold.name}_{task_mode}")
        rows.append({"fold": fold.name, "val_subjects": ",".join(fold.val_subjects),
                      "task_mode": task_mode, **metrics})
    output = Path(cfg["training"]["output_dir"]) / f"group_kfold_metrics_{task_mode}.csv"
    save_fold_metrics(rows, output)
    logger.info("GroupKFold metrics saved to %s", output)
    return rows


def run_cross_eval(cfg: dict, folds, task_mode: str, eval_on_gaipat: bool,
                   gaipat_dir: str, output_dir: str, checkpoint_dir: str | None,
                   logger) -> list[dict]:
    """跨数据集评估：加载每个 fold 的 best.pt，在目标数据集上评估。"""
    if eval_on_gaipat:
        sequences = discover_gaipat_sequences(gaipat_dir)
        test_dataset = GaipatWindowDataset(sequences=sequences, **gaipat_dataset_kwargs(cfg))
        logger.info("Cross-eval target: GAIPAT (%d windows)", len(test_dataset))
    else:
        d_kwargs = dataset_kwargs(cfg)
        target_seqs = filter_sequences_by_task(discover_sequences(cfg["data"]["root"]), task_mode)
        test_dataset = ADFWindowDataset(sequences=target_seqs, **d_kwargs)
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
            logger.warning("Checkpoint not found: %s", ckpt_path)
            continue
        try:
            metrics = evaluate_checkpoint(cfg, test_dataset, ckpt_path)
            logger.info("Cross-eval %s: %s", fold.name, metrics)
            cross_rows.append({
                "fold": fold.name, "task_mode": task_mode,
                "target": "gaipat" if eval_on_gaipat else "fatigue", **metrics,
            })
        except Exception as exc:
            logger.warning("Cross-eval %s failed: %s", fold.name, exc)
    return cross_rows


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADFNet Ablation Study Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--preset", default="full", choices=list(PRESETS.keys()),
                   help="Ablation preset (default: full)")
    p.add_argument("--cv", default="both", choices=["kfold", "loso", "both"],
                   help="Cross-validation: kfold, loso, both (default: both)")
    p.add_argument("--task-mode", nargs="+", default=["easy", "hard"],
                   choices=["easy", "hard"],
                   help="Task difficulty (default: easy hard)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output-base", default="./outputs/ablation")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--device", default=None)
    # ── cross-dataset ──
    p.add_argument("--eval-mode", choices=EVAL_MODES, default=EVAL_FATIGUE,
                   help="fatigue | fatigue_to_gaipat | gaipat | gaipat_to_fatigue")
    p.add_argument("--gaipat-dir", default=None,
                   help="GAIPAT data root (overrides gaipat.root in config)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Skip training; load fold checkpoints for cross-dataset evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    set_seed(base_cfg["seed"])

    eval_mode = args.eval_mode
    train_on_gaipat = eval_mode in (EVAL_GAIPAT, EVAL_GAIPAT_TO_FATIGUE)
    eval_on_gaipat = eval_mode in (EVAL_FATIGUE_TO_GAIPAT, EVAL_GAIPAT)
    is_cross = eval_mode in (EVAL_FATIGUE_TO_GAIPAT, EVAL_GAIPAT_TO_FATIGUE)
    gaipat_dir = args.gaipat_dir or base_cfg.get("gaipat", {}).get("root", "")
    skip_training = args.checkpoint_dir is not None

    preset_val = PRESETS[args.preset]
    overrides_list = generate_all_combinations() if preset_val is None else [preset_val]

    # GAIPAT 无 easy/hard → task_mode 强制 all
    task_modes = ["all"] if train_on_gaipat else args.task_mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 发现数据序列
    if train_on_gaipat:
        sequences_all = discover_gaipat_sequences(gaipat_dir)
    else:
        sequences_all = discover_sequences(base_cfg["data"]["root"])
    logger_root = setup_logger(args.output_base, name="ablation_launcher")
    logger_root.info("eval_mode=%s | preset=%s -> %d config(s) | cv=%s | task_modes=%s",
                     eval_mode, args.preset, len(overrides_list), args.cv, task_modes)

    all_results: list[dict] = []
    all_cross_results: list[dict] = []
    t_start = time.time()

    for idx, overrides in enumerate(overrides_list):
        label = ablation_label(overrides)
        run_dir = f"{args.output_base}/{timestamp}_{label}"
        logger_root.info("[%d/%d] Ablation: %s -> %s", idx + 1, len(overrides_list), label, run_dir)

        for task_mode in task_modes:
            sequences = filter_sequences_by_task(list(sequences_all), task_mode)
            if not sequences:
                logger_root.warning("No sequences for task_mode=%s", task_mode)
                continue

            # ── KFold ──
            if args.cv in ("kfold", "both"):
                cfg = copy.deepcopy(base_cfg)
                apply_overrides(cfg, overrides)
                cfg["training"]["output_dir"] = f"{run_dir}/kfold_{task_mode}"
                cfg["exp_name"] = f"ADFNet_ablation_{label}_kfold_{task_mode}"
                Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)
                if args.device:
                    cfg["training"]["device"] = args.device
                set_seed(cfg["seed"])

                logger = setup_logger(cfg["training"]["output_dir"],
                                     name=f"kfold_{label}_{task_mode}")
                logger.info("Ablation: %s | KFold | task_mode=%s | eval_mode=%s",
                            label, task_mode, eval_mode)
                save_hparams(
                    cfg, cfg["training"]["output_dir"],
                    script="run_ablation.py", task_mode=task_mode, timestamp=timestamp,
                    extra={"ablation_label": label, "cv": "kfold",
                           "eval_mode": eval_mode, "ablation_overrides": overrides,
                           "preset": args.preset},
                )

                if not skip_training:
                    rows = run_kfold(cfg, sequences, task_mode, logger,
                                    n_splits=args.n_splits, max_folds=args.max_folds,
                                    train_on_gaipat=train_on_gaipat)
                    for r in rows:
                        r["ablation"] = label
                        r["cv"] = "kfold"
                    all_results.extend(rows)

                # 跨数据集评估
                if is_cross:
                    folds = group_kfold_folds(sequences, n_splits=args.n_splits,
                                              seed=cfg["seed"])
                    if args.max_folds is not None:
                        folds = folds[:args.max_folds]
                    cross_rows = run_cross_eval(
                        cfg, folds, task_mode, eval_on_gaipat, gaipat_dir,
                        cfg["training"]["output_dir"], args.checkpoint_dir, logger,
                    )
                    for r in cross_rows:
                        r["ablation"] = label
                        r["cv"] = "kfold"
                    all_cross_results.extend(cross_rows)
                    if cross_rows:
                        tgt = "gaipat" if eval_on_gaipat else "fatigue"
                        out = Path(cfg["training"]["output_dir"]) / f"kfold_cross_to_{tgt}_{task_mode}.csv"
                        save_fold_metrics(cross_rows, out)

            # ── LOSO ──
            if args.cv in ("loso", "both"):
                cfg = copy.deepcopy(base_cfg)
                apply_overrides(cfg, overrides)
                cfg["training"]["output_dir"] = f"{run_dir}/loso_{task_mode}"
                cfg["exp_name"] = f"ADFNet_ablation_{label}_loso_{task_mode}"
                Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)
                if args.device:
                    cfg["training"]["device"] = args.device
                set_seed(cfg["seed"])

                logger = setup_logger(cfg["training"]["output_dir"],
                                     name=f"loso_{label}_{task_mode}")
                logger.info("Ablation: %s | LOSO | task_mode=%s | eval_mode=%s",
                            label, task_mode, eval_mode)
                save_hparams(
                    cfg, cfg["training"]["output_dir"],
                    script="run_ablation.py", task_mode=task_mode, timestamp=timestamp,
                    extra={"ablation_label": label, "cv": "loso",
                           "eval_mode": eval_mode, "ablation_overrides": overrides,
                           "preset": args.preset},
                )

                if not skip_training:
                    rows = run_loso(cfg, sequences, task_mode, logger,
                                   max_folds=args.max_folds,
                                   train_on_gaipat=train_on_gaipat)
                    for r in rows:
                        r["ablation"] = label
                        r["cv"] = "loso"
                    all_results.extend(rows)

                if is_cross:
                    folds = loso_folds(sequences)
                    if args.max_folds is not None:
                        folds = folds[:args.max_folds]
                    cross_rows = run_cross_eval(
                        cfg, folds, task_mode, eval_on_gaipat, gaipat_dir,
                        cfg["training"]["output_dir"], args.checkpoint_dir, logger,
                    )
                    for r in cross_rows:
                        r["ablation"] = label
                        r["cv"] = "loso"
                    all_cross_results.extend(cross_rows)
                    if cross_rows:
                        tgt = "gaipat" if eval_on_gaipat else "fatigue"
                        out = Path(cfg["training"]["output_dir"]) / f"loso_cross_to_{tgt}_{task_mode}.csv"
                        save_fold_metrics(cross_rows, out)

    # ── 汇总 ──
    if all_results:
        summary_path = Path(args.output_base) / f"{timestamp}_ablation_summary.csv"
        save_fold_metrics(all_results, summary_path)
        logger_root.info("Training results saved to %s", summary_path)
    if all_cross_results:
        cross_path = Path(args.output_base) / f"{timestamp}_cross_eval_summary.csv"
        save_fold_metrics(all_cross_results, cross_path)
        logger_root.info("Cross-eval results saved to %s", cross_path)

    elapsed = time.time() - t_start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    logger_root.info("Total ablation time: %dh %dm %ds", h, m, s)


if __name__ == "__main__":
    main()
