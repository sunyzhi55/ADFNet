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
from data.io import discover_sequences, filter_sequences_by_task
from data.split import explicit_folds, group_kfold_folds, loso_folds
from training.seed import set_seed
from training.trainer import train_fold
from utils.config import load_config, save_hparams
from utils.logging import setup_logger

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

def run_loso(cfg: dict, sequences, task_mode: str, logger, max_folds: int | None = None) -> list[dict]:
    """在给定 config + 序列上跑完整 LOSO，返回每 fold 的指标行。"""
    folds = loso_folds(sequences)
    if max_folds is not None:
        folds = folds[:max_folds]
    logger.info("LOSO: %d folds for task_mode=%s", len(folds), task_mode)
    data_kwargs = dataset_kwargs(cfg)
    rows: list[dict] = []
    for fold in folds:
        train_ds = ADFWindowDataset(sequences=fold.train, **data_kwargs)
        val_ds = ADFWindowDataset(sequences=fold.val, **data_kwargs)
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
              n_splits: int = 5, max_folds: int | None = None) -> list[dict]:
    """在给定 config + 序列上跑 GroupKFold / explicit_folds，返回每 fold 的指标行。"""
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
    data_kwargs = dataset_kwargs(cfg)
    rows: list[dict] = []
    for fold in folds:
        train_ds = ADFWindowDataset(sequences=fold.train, **data_kwargs)
        val_ds = ADFWindowDataset(sequences=fold.val, **data_kwargs)
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


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADFNet 消融实验启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--preset", default="full", choices=list(PRESETS.keys()),
                   help="消融预设（默认 full = 完整模型基线）")
    p.add_argument("--cv", default="both", choices=["kfold", "loso", "both"],
                   help="交叉验证方式（默认 both = kfold + loso 都跑）")
    p.add_argument("--task-mode", nargs="+", default=["easy", "hard"],
                   choices=["easy", "hard"],
                   help="任务难度，可多选（默认 easy hard 都跑）")
    p.add_argument("--config", default="configs/default.yaml", help="基线配置文件")
    p.add_argument("--output-base", default="./outputs/ablation",
                   help="消融实验输出根目录")
    p.add_argument("--n-splits", type=int, default=5, help="GroupKFold 折数")
    p.add_argument("--max-folds", type=int, default=None,
                   help="每种 CV 最多跑几个 fold（调试用）")
    p.add_argument("--device", default=None,
                   help="覆盖 training.device（如 cuda:1）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    set_seed(base_cfg["seed"])

    # 确定本次要跑的消融覆盖列表
    preset_val = PRESETS[args.preset]
    if preset_val is None:
        # all_combinations
        overrides_list = generate_all_combinations()
    else:
        overrides_list = [preset_val]

    task_modes = args.task_mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 发现数据序列（只需做一次）
    sequences_all = discover_sequences(base_cfg["data"]["root"])
    logger_root = setup_logger(args.output_base, name="ablation_launcher")
    logger_root.info("Preset: %s → %d ablation config(s), cv=%s, task_modes=%s",
                     args.preset, len(overrides_list), args.cv, task_modes)

    all_results: list[dict] = []
    t_start = time.time()

    for idx, overrides in enumerate(overrides_list):
        label = ablation_label(overrides)
        run_dir = f"{args.output_base}/{timestamp}_{label}"

        logger_root.info("[%d/%d] Ablation: %s  →  %s", idx + 1, len(overrides_list), label, run_dir)

        for task_mode in task_modes:
            sequences = filter_sequences_by_task(list(sequences_all), task_mode)
            if not sequences:
                logger_root.warning("No sequences for task_mode=%s, skipping", task_mode)
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
                logger.info("Ablation: %s | KFold | task_mode=%s", label, task_mode)
                save_hparams(
                    cfg, cfg["training"]["output_dir"],
                    script="run_ablation.py", task_mode=task_mode, timestamp=timestamp,
                    extra={"ablation_label": label, "cv": "kfold",
                           "ablation_overrides": overrides, "preset": args.preset},
                )

                rows = run_kfold(cfg, sequences, task_mode, logger,
                                n_splits=args.n_splits, max_folds=args.max_folds)
                for r in rows:
                    r["ablation"] = label
                    r["cv"] = "kfold"
                all_results.extend(rows)

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
                logger.info("Ablation: %s | LOSO | task_mode=%s", label, task_mode)
                save_hparams(
                    cfg, cfg["training"]["output_dir"],
                    script="run_ablation.py", task_mode=task_mode, timestamp=timestamp,
                    extra={"ablation_label": label, "cv": "loso",
                           "ablation_overrides": overrides, "preset": args.preset},
                )

                rows = run_loso(cfg, sequences, task_mode, logger,
                               max_folds=args.max_folds)
                for r in rows:
                    r["ablation"] = label
                    r["cv"] = "loso"
                all_results.extend(rows)

    # ── 汇总所有结果 ──
    if all_results:
        summary_path = Path(args.output_base) / f"{timestamp}_ablation_summary.csv"
        save_fold_metrics(all_results, summary_path)
        logger_root.info("All ablation results saved to %s", summary_path)

    elapsed = time.time() - t_start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    logger_root.info("Total ablation time: %dh %dm %ds", h, m, s)


if __name__ == "__main__":
    main()
