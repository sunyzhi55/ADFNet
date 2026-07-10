"""计算 LOSO 实验结果的置信区间。

用法:
    python scripts/compute_ci.py outputs/xxx/loso_metrics_easy.csv
    python scripts/compute_ci.py outputs/xxx/loso_metrics_easy.csv outputs/xxx/loso_metrics_hard.csv
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson Score 置信区间（推荐用于准确率）。

    输入:
        k: 预测正确的样本数
        n: 测试样本总数
        confidence: 置信水平，默认 0.95

    输出:
        (lower, upper): 置信区间上下界
    """
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # 1.96 for 95%
    p_hat = k / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
    return max(0, center - margin), min(1, center + margin)


def process_csv(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    # 去掉 mean/std 汇总行，只保留每个 fold 的数据
    folds_df = df[df["fold"] != "mean"][df["fold"] != "std"].copy()

    print(f"\n{'='*60}")
    print(f"文件: {csv_path}")
    print(f"{'='*60}")

    # 汇总所有 fold 的混淆矩阵
    total_tn = folds_df["val_cm_tn"].sum()
    total_fp = folds_df["val_cm_fp"].sum()
    total_fn = folds_df["val_cm_fn"].sum()
    total_tp = folds_df["val_cm_tp"].sum()
    total_n = total_tn + total_fp + total_fn + total_tp
    total_correct = total_tn + total_tp

    # 计算总体置信区间
    acc = total_correct / total_n
    ci_low, ci_high = wilson_ci(total_correct, total_n)

    print(f"\n【汇总结果】（所有 fold 合并）")
    print(f"  总样本数:   {total_n}")
    print(f"  正确数:     {total_correct}")
    print(f"  准确率:     {acc:.4f}")
    print(f"  95% CI:     [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  区间宽度:   {ci_high - ci_low:.4f}")

    # 每个 fold 单独计算
    print(f"\n【每个 fold 的置信区间】")
    print(f"{'Fold':<10} {'样本数':>8} {'正确数':>8} {'ACC':>8} {'95% CI':>20}")
    print(f"{'-'*60}")
    for _, row in folds_df.iterrows():
        n = int(row["val_cm_tn"] + row["val_cm_fp"] + row["val_cm_fn"] + row["val_cm_tp"])
        k = int(row["val_cm_tn"] + row["val_cm_tp"])
        fold_acc = k / n if n > 0 else 0
        lo, hi = wilson_ci(k, n)
        print(f"{row['fold']:<10} {n:>8} {k:>8} {fold_acc:>8.4f} [{lo:.4f}, {hi:.4f}]")

    # 如果 CSV 中有多个 task_mode，按 task_mode 分组
    if "task_mode" in folds_df.columns:
        modes = folds_df["task_mode"].unique()
        if len(modes) > 1:
            print(f"\n【按 task_mode 分组】")
            for mode in modes:
                mode_df = folds_df[folds_df["task_mode"] == mode]
                m_tn = mode_df["val_cm_tn"].sum()
                m_tp = mode_df["val_cm_tp"].sum()
                m_n = (m_tn + mode_df["val_cm_fp"].sum() +
                       mode_df["val_cm_fn"].sum() + m_tp)
                m_acc = (m_tn + m_tp) / m_n if m_n > 0 else 0
                lo, hi = wilson_ci(m_tn + m_tp, m_n)
                print(f"  {mode}: ACC={m_acc:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], n={m_n}")


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/compute_ci.py <csv_path1> [csv_path2] ...")
        sys.exit(1)

    for path in sys.argv[1:]:
        process_csv(path)


if __name__ == "__main__":
    main()
