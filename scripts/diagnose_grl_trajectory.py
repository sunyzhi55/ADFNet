"""诊断 GRL 训练轨迹：视线是否编码身份？GRL 是否伤 val？

读取 result/loso_{easy,hard}/loso_*_<mode>/history.csv，回答两个决定性问题：

  Q1. 视线特征是否编码 subject 身份？
      看 λ≈0 的极早期（epoch<=early_window，此时 GRL 几乎没介入）判别器
      train_subject_acc 能否明显高于随机 1/K。若 ≈随机 → 视线不含身份 → GRL 无的之矢。
      若明显升高 → 视线含身份，GRL 有工作基础。

  Q2. GRL 是帮还是伤 val？
      按 λ 分箱看 val_f1/val_auc 均值，并算 val_f1 与 λ 的相关。
      若 val 随 λ 增大而下降 → GRL 净伤；若持平/上升 → GRL 有益或中性。

跑法：
  python scripts/diagnose_grl_trajectory.py --result-dir result --n-train-subjects 19
  python scripts/diagnose_grl_trajectory.py --result-dir result --n-train-subjects 19 --mode hard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose GRL trajectory from history.csv files")
    p.add_argument("--result-dir", default="result")
    p.add_argument("--mode", choices=["easy", "hard", "both"], default="both")
    p.add_argument("--n-train-subjects", type=int, default=19,
                   help="训练 fold 被试数，用于算随机基线 1/K（LOSO=19）")
    p.add_argument("--early-window", type=int, default=10,
                   help="前 N 个 epoch 视为 λ≈0 的极早期窗口")
    p.add_argument("--output-csv", default=None)
    return p.parse_args()


def load_histories(result_dir: Path, mode: str) -> list[tuple[str, pd.DataFrame]]:
    base = result_dir / f"loso_{mode}"
    files = sorted(base.glob(f"loso_*_{mode}/history.csv"))
    out = []
    for f in files:
        df = pd.read_csv(f)
        df = df.sort_values("epoch").reset_index(drop=True)
        out.append((f.parent.name, df))
    return out


def per_fold_stats(name: str, df: pd.DataFrame, early_window: int, random_baseline: float) -> dict:
    early = df[df["epoch"] <= early_window]
    s = {
        "fold": name,
        "n_epochs": len(df),
        "best_epoch": int(df.loc[df["val_f1"].idxmax(), "epoch"]) if df["val_f1"].notna().any() else -1,
        "best_val_f1": float(df["val_f1"].max()),
        "best_val_auc": float(df.loc[df["val_f1"].idxmax(), "val_auc"]) if df["val_f1"].notna().any() else float("nan"),
        # Q1: 早期（λ≈0）判别器准确率
        "early_max_subj_acc": float(early["train_subject_acc"].max()) if len(early) else float("nan"),
        "early_mean_subj_acc": float(early["train_subject_acc"].mean()) if len(early) else float("nan"),
        "early_peak_epoch": int(early.loc[early["train_subject_acc"].idxmax(), "epoch"]) if len(early) and early["train_subject_acc"].notna().any() else -1,
        "early_lambda_max": float(early["grl_lambda"].max()) if len(early) else float("nan"),
        # 全程判别器峰值
        "peak_subj_acc": float(df["train_subject_acc"].max()),
        "peak_subj_epoch": int(df.loc[df["train_subject_acc"].idxmax(), "epoch"]),
        "subj_acc_at_best": float(df.loc[df["val_f1"].idxmax(), "train_subject_acc"]) if df["val_f1"].notna().any() else float("nan"),
        # Q2: 早期 vs 全程 val 峰值（早期≈无GRL天花板）
        "early_best_val_f1": float(early["val_f1"].max()) if len(early) else float("nan"),
        "late_best_val_f1": float(df[df["epoch"] > early_window]["val_f1"].max()) if (df["epoch"] > early_window).any() else float("nan"),
    }
    s["early_vs_late_gap"] = s["late_best_val_f1"] - s["early_best_val_f1"]
    s["random_baseline"] = random_baseline
    s["early_lift_over_random"] = s["early_max_subj_acc"] - random_baseline
    return s


def lambda_bin_analysis(histories: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """按 λ 分箱，跨 fold 平均 val_f1/val_auc/train_subject_acc。"""
    bins = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.01)]
    labels = ["λ∈[0,.05)", "λ∈[.05,.2)", "λ∈[.2,.5)", "λ∈[.5,.9)", "λ∈[.9,1]"]
    rows = []
    for (lo, hi), lab in zip(bins, labels):
        vf, va, sa, ce = [], [], [], []
        for _, df in histories:
            m = (df["grl_lambda"] >= lo) & (df["grl_lambda"] < hi)
            sub = df[m]
            if len(sub):
                vf.append(sub["val_f1"].mean())
                va.append(sub["val_auc"].mean())
                sa.append(sub["train_subject_acc"].mean())
                ce.append(sub["train_adv_ce"].mean())
        rows.append({
            "lambda_bin": lab,
            "n_folds_covered": len(vf),
            "mean_val_f1": float(np.mean(vf)) if vf else float("nan"),
            "mean_val_auc": float(np.mean(va)) if va else float("nan"),
            "mean_train_subject_acc": float(np.mean(sa)) if sa else float("nan"),
            "mean_train_adv_ce": float(np.mean(ce)) if ce else float("nan"),
        })
    return pd.DataFrame(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """不依赖 scipy 的 Spearman 相关：对两列 rank 后算 Pearson。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy().astype(np.float64).copy()
    ry = pd.Series(y).rank().to_numpy().astype(np.float64).copy()
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def correlation_analysis(histories: list[tuple[str, pd.DataFrame]]) -> dict:
    """每 fold 算 val_f1 与 λ 的 Spearman 相关，再平均。负相关=GRL 越强 val 越差。"""
    corrs = []
    for _, df in histories:
        sub = df[["grl_lambda", "val_f1"]].dropna()
        if len(sub) > 5 and sub["grl_lambda"].nunique() > 1:
            corrs.append(_spearman(sub["grl_lambda"].to_numpy(), sub["val_f1"].to_numpy()))
    arr = np.asarray([c for c in corrs if not np.isnan(c)], dtype=np.float64)
    return {
        "per_fold_corr_mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        "per_fold_corr_std": float(np.nanstd(arr)) if arr.size else float("nan"),
        "frac_negative_corr": float(np.nanmean(arr < 0)) if arr.size else float("nan"),
        "n_folds": int(arr.size),
    }


def analyze_mode(result_dir: Path, mode: str, args) -> dict:
    histories = load_histories(result_dir, mode)
    if not histories:
        print(f"\n[{mode}] no history.csv found under {result_dir}/loso_{mode}")
        return {}
    random_base = 1.0 / args.n_train_subjects
    print(f"\n==================== mode={mode}  folds={len(histories)}  random_baseline={random_base:.4f} ====================")

    stats = [per_fold_stats(name, df, args.early_window, random_base) for name, df in histories]
    sdf = pd.DataFrame(stats)

    print("\n--- Q1: 视线是否编码身份？(λ≈0 早期判别器准确率) ---")
    print(f"  early_max_subj_acc  : mean={sdf['early_max_subj_acc'].mean():.3f}  (random={random_base:.3f})")
    print(f"  early_mean_subj_acc : mean={sdf['early_mean_subj_acc'].mean():.3f}")
    print(f"  early_lift_over_random: mean={sdf['early_lift_over_random'].mean():+.3f}  "
          f"(>0.05 表示视线含身份信号)")
    print(f"  peak_subj_acc (全程): mean={sdf['peak_subj_acc'].mean():.3f} @ mean epoch={sdf['peak_subj_epoch'].mean():.0f}")

    print("\n--- Q2: GRL 是帮还是伤 val？(按 λ 分箱) ---")
    bins = lambda_bin_analysis(histories)
    print(bins.to_string(index=False))

    print("\n--- 早期(λ≈0) vs 后期 val_f1 峰值（早期≈无GRL天花板）---")
    print(f"  early_best_val_f1 : mean={sdf['early_best_val_f1'].mean():.3f}")
    print(f"  late_best_val_f1  : mean={sdf['late_best_val_f1'].mean():.3f}")
    print(f"  gap (late-early)  : mean={sdf['early_vs_late_gap'].mean():+.3f}  "
          f"(>0 表示 GRL 后期还能涨；<0 表示 GRL 伤 val)")

    corr = correlation_analysis(histories)
    print("\n--- val_f1 与 λ 的 Spearman 相关（负=GRL 越强 val 越差）---")
    print(f"  per_fold corr mean={corr['per_fold_corr_mean']:+.3f}  std={corr['per_fold_corr_std']:.3f}")
    print(f"  负相关 fold 占比={corr['frac_negative_corr']*100:.0f}%  (n={corr['n_folds']})")

    # 判定
    print("\n--- 判定 ---")
    identity_signal = sdf["early_lift_over_random"].mean()
    late_gap = sdf["early_vs_late_gap"].mean()
    if identity_signal < 0.03:
        v = ("视线特征几乎不编码 subject 身份（早期判别器 ≈ 随机）。"
             "GRL 无的之矢 → 放弃 GRL，集中攻过拟合。")
    elif corr["per_fold_corr_mean"] < -0.2 or late_gap < -0.02:
        v = ("视线含身份信号，但 GRL 越强 val 越差（后期不如早期）。"
             "→ λ 太猛：调小 max_lambda(0.1~0.3)+warmup+缓 slope。")
    else:
        v = ("视线含身份且 GRL 对 val 中性/略益，但收益不显著。"
             "→ 可保留 GRL 微调，或转向过拟合为主攻方向。")
    print("  " + v)

    return {"mode": mode, "per_fold": sdf, "lambda_bins": bins, "corr": corr, "verdict": v}


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    modes = ["easy", "hard"] if args.mode == "both" else [args.mode]
    summary = []
    for mode in modes:
        r = analyze_mode(result_dir, mode, args)
        if r:
            summary.append(r)
    if args.output_csv and summary:
        out = []
        for r in summary:
            d = r["per_fold"].copy()
            d["mode"] = r["mode"]
            out.append(d)
        pd.concat(out, ignore_index=True).to_csv(args.output_csv, index=False)
        print(f"\n[saved] per-fold stats -> {args.output_csv}")


if __name__ == "__main__":
    main()
