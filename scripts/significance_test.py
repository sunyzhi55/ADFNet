"""
显著性检验脚本 — ADFNet vs 对比方法
====================================

对 ADFNet 和最佳对比方法的逐折指标做 **配对 t 检验**（paired t-test），
并施加 **FDR (Benjamini-Hochberg) 校正**控制 false discovery rate。
同时输出校正前后的 p 值便于对比。

支持 4 组实验一次性计算:
    - LOSO easy (20 折)
    - LOSO hard (20 折)
    - KFold easy (5 折)
    - KFold hard (5 折)

使用方法:
    1. 在下方 DATA 区域填入 ADFNet 和对比方法的逐折指标列表
    2. 运行: python scripts/significance_test.py
    3. 结果输出到 --output 指定的 CSV（默认 result/significance_test.csv）

指标包括: auc, acc, f1, precision, recall（按需增删）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

# ══════════════════════════════════════════════════════════════
# DATA — 在这里填入逐折指标（每个列表长度 = fold 数）
# ══════════════════════════════════════════════════════════════

# ── 指标名称（可增删，只计算 ours 和 baseline 中都有的指标）──
METRICS = ["auc", "acc", "f1", "precision", "recall"]

# ── 对比方法名称（写入 CSV 便于溯源）──
BASELINE_NAME = "BestBaseline"  # 替换为实际方法名，如 "MIGCN", "STAFNet"

# ═══════════════════════════════════════════════════
# LOSO-Easy 实验（20 折）
# ═══════════════════════════════════════════════════

LOSO_EASY_OURS = {
    "auc":[0.999893, 0.831605, 0.850551, 0.949491, 0.995791, 0.913884, 0.865548, 0.922443, 0.97469, 0.856076, 0.8755666, 0.858171, 0.944182, 0.934783, 0.879227, 0.998261, 0.875499, 0.867428, 0.827158, 0.852148542],
    "acc":[0.992701, 0.828467, 0.791971, 0.916058, 0.908759, 0.771739, 0.80173913, 0.815217, 0.949275, 0.778985507, 0.79710144, 0.815217, 0.880435, 0.931159, 0.851449, 0.949275, 0.826087, 0.761733, 0.75, 0.747292],
    "f1":[0.992701, 0.853582, 0.8, 0.914498, 0.916388, 0.806154, 0.81081, 0.793522, 0.947368, 0.808777, 0.80689655, 0.817204, 0.869565, 0.929368, 0.851986, 0.948529, 0.832168, 0.795031, 0.769, 0.722222],
    # "precision": [0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],
    # "recall":[0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],
}

LOSO_EASY_BASELINE = {
    "auc": [0.99984, 0.777665, 0.929405, 0.955139, 0.999787, 0.975215, 0.721251, 0.81632, 0.988028, 0.807708, 0.897973, 0.928901, 0.995747, 0.956732, 0.911258, 0.939456, 0.869303, 0.93593, 0.846776, 0.778542],
    "acc": [0.992701, 0.693431, 0.835766, 0.890511, 0.989051, 0.913043, 0.695652, 0.768116, 0.974638, 0.746377, 0.818841, 0.887681, 0.967391, 0.905797, 0.84058, 0.884058, 0.800725, 0.884477, 0.731884, 0.747292],
    "precision": [0.985611, 0.819277, 0.802632, 0.928, 0.985507, 0.907143, 0.66875, 0.756944, 0.985185, 0.820755, 0.73913, 0.849673, 0.970803, 0.9375, 0.879032, 0.990741, 0.927835, 0.927419, 0.863636, 0.80531],
    "recall": [1.0, 0.49635, 0.890511, 0.846715, 0.992701, 0.92029, 0.775362, 0.789855, 0.963768, 0.630435, 0.985507, 0.942029, 0.963768, 0.869565, 0.789855, 0.775362, 0.652174, 0.833333, 0.550725, 0.654676],
    "specificity": [0.985401, 0.890511, 0.781022, 0.934307, 0.985401, 0.905797, 0.615942, 0.746377, 0.985507, 0.862319, 0.652174, 0.833333, 0.971014, 0.942029, 0.891304, 0.992754, 0.949275, 0.935252, 0.913043, 0.84058],
    "f1": [0.992754, 0.618182, 0.844291, 0.885496, 0.989091, 0.913669, 0.718121, 0.77305, 0.974359, 0.713115, 0.84472, 0.893471, 0.967273, 0.902256, 0.832061, 0.869919, 0.765957, 0.877863, 0.672566, 0.722222],
    "kappa": [0.985401, 0.386861, 0.671533, 0.781022, 0.978102, 0.826087, 0.391304, 0.536232, 0.949275, 0.492754, 0.637681, 0.775362, 0.934783, 0.811594, 0.681159, 0.768116, 0.601449, 0.768866, 0.463768, 0.494921],
    "balance_acc": [0.992701, 0.693431, 0.835766, 0.890511, 0.989051, 0.913043, 0.695652, 0.768116, 0.974638, 0.746377, 0.818841, 0.887681, 0.967391, 0.905797, 0.84058, 0.884058, 0.800725, 0.884293, 0.731884, 0.747628],
}

# ═══════════════════════════════════════════════════
# LOSO-Hard 实验（20 折）
# ═══════════════════════════════════════════════════

LOSO_HARD_OURS = {
    # "auc":[0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],
    # "acc":[0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],
    # "f1":[0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],
    "auc":[0.992115, 0.802174, 0.940647, 0.912662, 0.955405, 0.831837, 0.847857, 0.900719, 0.998571, 0.853255465, 0.847755, 0.661146, 0.975488, 0.982477, 0.947415, 0.867398, 0.945427, 0.8637092, 0.779694, 0.8545243],
    "acc":[0.978102, 0.686131, 0.905109, 0.817518248, 0.864964, 0.782143, 0.785714, 0.842294, 0.992857, 0.764285714, 0.8393857, 0.665468, 0.9319, 0.9319, 0.913669, 0.805755, 0.903226, 0.782142, 0.728571, 0.775],
    "f1":[0.978102, 0.739394, 0.905797, 0.845679, 0.846473, 0.801303, 0.72973, 0.826772, 0.992857, 0.798780488, 0.847457, 0.739496, 0.932862, 0.934256, 0.918367, 0.804348, 0.897338, 0.7797833, 0.763975, 0.81305638],
    # "precision": [0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],
    # "recall":[0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0],
}

LOSO_HARD_BASELINE = {
    "auc": [0.988279, 0.821408, 0.996377, 0.535191, 0.983697, 0.847857, 0.59801, 0.922508, 0.999082, 0.784694, 0.855, 0.745096, 0.988952, 0.976516, 0.962994, 0.867812, 0.950874, 0.729031, 0.820102, 0.779694],
    "acc": [0.956204, 0.79562, 0.941606, 0.529197, 0.927007, 0.785714, 0.575, 0.885305, 0.992857, 0.728571, 0.821429, 0.723022, 0.953405, 0.939068, 0.946043, 0.791367, 0.878136, 0.714286, 0.710714, 0.728571],
    "precision": [0.984496, 0.917526, 1.0, 0.519417, 0.933333, 0.987805, 0.578947, 0.94958, 1.0, 0.725352, 0.894737, 0.668478, 0.950355, 0.955556, 0.969697, 0.840336, 0.981818, 0.772727, 0.915493, 0.675824],
    "recall": [0.927007, 0.649635, 0.883212, 0.781022, 0.919708, 0.578571, 0.55, 0.81295, 0.985714, 0.735714, 0.728571, 0.884892, 0.957143, 0.921429, 0.920863, 0.719424, 0.771429, 0.607143, 0.464286, 0.878571],
    "specificity": [0.985401, 0.941606, 1.0, 0.277372, 0.934307, 0.992857, 0.6, 0.957143, 1.0, 0.721429, 0.914286, 0.561151, 0.94964, 0.956835, 0.971223, 0.863309, 0.985611, 0.821429, 0.957143, 0.578571],
    "f1": [0.954887, 0.760684, 0.937984, 0.623907, 0.926471, 0.72973, 0.564103, 0.875969, 0.992806, 0.730496, 0.80315, 0.76161, 0.953737, 0.938182, 0.944649, 0.775194, 0.864, 0.68, 0.616114, 0.763975],
    "kappa": [0.912409, 0.591241, 0.883212, 0.058394, 0.854015, 0.571429, 0.15, 0.770488, 0.985714, 0.457143, 0.642857, 0.446043, 0.906806, 0.87815, 0.892086, 0.582734, 0.756457, 0.428571, 0.421429, 0.457143],
    "balance_acc": [0.956204, 0.79562, 0.941606, 0.529197, 0.927007, 0.785714, 0.575, 0.885046, 0.992857, 0.728571, 0.821429, 0.723022, 0.953392, 0.939132, 0.946043, 0.791367, 0.87852, 0.714286, 0.710714, 0.728571],
}

# ═══════════════════════════════════════════════════
# KFold-Easy 实验（5 折）
# ═══════════════════════════════════════════════════

KFOLD_EASY_OURS = {
    "auc":       [0.86561, 0.80545648, 0.68837, 0.858703, 0.886034],
    "acc":       [0.817273, 0.7495463, 0.8045249, 0.781307, 0.844062],
    "f1":        [0.80236, 0.7162457, 0.8208955, 0.792777, 0.846702],
    # "precision": [0.0, 0.0, 0.0, 0.0, 0.0],
    # "recall":    [0.0, 0.0, 0.0, 0.0, 0.0],
}

KFOLD_EASY_BASELINE = {
    "auc": [0.896939, 0.789711, 0.635316, 0.804976, 0.943265],
    "acc": [0.817273, 0.721416, 0.653394, 0.753176, 0.867634],
    "precision": [0.908665, 0.840782, 0.608974, 0.813483, 0.899408],
    "recall": [0.705455, 0.546279, 0.858951, 0.656987, 0.827586],
    "specificity": [0.929091, 0.896552, 0.447464, 0.849365, 0.907609],
    "f1": [0.794268, 0.662266, 0.712678, 0.726908, 0.862004],
    "kappa": [0.634545, 0.442831, 0.306529, 0.506352, 0.735248],
    "balance_acc": [0.817273, 0.721416, 0.653207, 0.753176, 0.867597],
}

# ═══════════════════════════════════════════════════
# KFold-Hard 实验（5 折）
# ═══════════════════════════════════════════════════

KFOLD_HARD_OURS = {
    "auc":[0.816624, 0.788326, 0.8758275, 0.859989, 0.789087],
    "acc":[0.764228, 0.741906, 0.8103757, 0.798198, 0.766397],
    "f1":[0.763373, 0.747581, 0.8212479, 0.80789, 0.779661],
    # "precision": [0.0, 0.0, 0.0, 0.0, 0.0],
    # "recall":    [0.0, 0.0, 0.0, 0.0, 0.0],
}

KFOLD_HARD_BASELINE = {
    "auc": [0.860322, 0.808008, 0.720681, 0.851665, 0.807554],
    "acc": [0.785908, 0.747302, 0.667263, 0.800901, 0.763702],
    "precision": [0.807767, 0.810384, 0.647244, 0.77649, 0.722727],
    "recall": [0.750903, 0.645683, 0.735242, 0.845045, 0.856373],
    "specificity": [0.820976, 0.848921, 0.599284, 0.756757, 0.670863],
    # "f1": [0.778297, 0.718719, 0.688442, 0.809318, 0.783895], # AFM-CIR
    "f1": [0.791367, 0.80376, 0.705954, 0.720805, 0.830835], # HM-LSTM
    "kappa": [0.571843, 0.494604, 0.334526, 0.601802, 0.527324],
    "balance_acc": [0.78594, 0.747302, 0.667263, 0.800901, 0.763618],
}

# ═══════════════════════════════════════════════════
# 实验注册表（控制运行顺序与显示名称）
# ═══════════════════════════════════════════════════

EXPERIMENTS: dict[str, dict] = {
    "LOSO_easy":  {"ours": LOSO_EASY_OURS,  "baseline": LOSO_EASY_BASELINE,  "n_folds": 20},
    "LOSO_hard":  {"ours": LOSO_HARD_OURS,  "baseline": LOSO_HARD_BASELINE,  "n_folds": 20},
    "KFold_easy": {"ours": KFOLD_EASY_OURS, "baseline": KFOLD_EASY_BASELINE, "n_folds": 5},
    "KFold_hard": {"ours": KFOLD_HARD_OURS, "baseline": KFOLD_HARD_BASELINE, "n_folds": 5},
}


# ══════════════════════════════════════════════════════════════
# 核心检验逻辑
# ══════════════════════════════════════════════════════════════

def paired_ttest_fdr(
    ours: dict[str, list[float]],
    baseline: dict[str, list[float]],
    metrics: list[str],
    alpha: float = 0.05,
) -> list[dict]:
    """对每个指标做配对 t 检验，施加 FDR (Benjamini-Hochberg) 校正。

    BH 流程:
        1. 对每个指标做 paired t-test → p_raw
        2. 按 p_raw 升序排列，计算 p_bh = p_raw * m / rank
        3. 从最大 rank 开始强制单调: p_bh[i] = min(p_bh[i], p_bh[i+1])
        4. 上限 clip 到 1.0

    Returns:
        每个指标一行结果 dict，包含 p_raw 和 p_fdr 两组 p 值及显著性判定。
    """
    n_tests = len(metrics)

    # ── 第一轮: 逐指标配对 t 检验 ──
    raw_rows: list[dict] = []
    for m in metrics:
        a = np.asarray(ours[m], dtype=np.float64)
        b = np.asarray(baseline[m], dtype=np.float64)
        assert len(a) == len(b), f"{m}: 两组 fold 数不一致 ({len(a)} vs {len(b)})"
        n = len(a)
        diff = a - b

        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        mean_d, std_d = float(np.mean(diff)), float(np.std(diff, ddof=1))

        if std_d < 1e-15:
            t_stat, p_raw = 0.0, 1.0
        else:
            t_stat, p_raw = stats.ttest_rel(a, b)
            t_stat, p_raw = float(t_stat), float(p_raw)

        cohen_d = mean_d / std_d if std_d > 1e-15 else 0.0
        se = std_d / np.sqrt(n) if n > 0 else 0.0
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 0.0

        raw_rows.append({
            "metric": m, "n_folds": n, "df": df,
            "mean_a": mean_a, "mean_b": mean_b,
            "mean_d": mean_d, "std_d": std_d,
            "t_stat": t_stat, "p_raw": p_raw,
            "cohen_d": cohen_d,
            "ci_lo": mean_d - t_crit * se,
            "ci_hi": mean_d + t_crit * se,
        })

    # ── 第二轮: Benjamini-Hochberg FDR 校正 ──
    p_values = [r["p_raw"] for r in raw_rows]
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values, dtype=np.float64)[sorted_indices]

    # p_bh = p_raw * m / rank,  rank 从 1 开始
    ranks = np.arange(1, n_tests + 1, dtype=np.float64)
    p_bh = sorted_p * n_tests / ranks

    # 强制单调（从最大 rank 向回扫描）
    for i in range(n_tests - 2, -1, -1):
        p_bh[i] = min(p_bh[i], p_bh[i + 1])

    # clip 到 [0, 1]
    p_bh = np.clip(p_bh, 0.0, 1.0)

    # 还原到原始顺序
    p_fdr = np.empty(n_tests, dtype=np.float64)
    p_fdr[sorted_indices] = p_bh

    # ── 组装结果 ──
    results: list[dict] = []
    for i, r in enumerate(raw_rows):
        sig_raw = "Yes" if r["p_raw"] < alpha else "No"
        sig_fdr = "Yes" if p_fdr[i] < alpha else "No"

        results.append({
            "metric": r["metric"],
            "n_folds": r["n_folds"],
            "mean_ADFNet": round(r["mean_a"], 6),
            f"mean_{BASELINE_NAME}": round(r["mean_b"], 6),
            "mean_diff": round(r["mean_d"], 6),
            "std_diff": round(r["std_d"], 6),
            "t_stat": round(r["t_stat"], 4),
            "df": r["df"],
            "p_raw": f"{r['p_raw']:.6e}",
            "sig_raw": sig_raw,
            "p_fdr": f"{p_fdr[i]:.6e}",
            "sig_fdr": sig_fdr,
            "cohen_d": round(r["cohen_d"], 4),
            "ci_95_lower": round(r["ci_lo"], 6),
            "ci_95_upper": round(r["ci_hi"], 6),
        })

    return results


def _is_filled(data: dict[str, list[float]], expected_len: int) -> bool:
    """检查数据是否已填入（至少有一个指标不全为 0）。"""
    for vals in data.values():
        if len(vals) != expected_len:
            return False
        if any(v != 0.0 for v in vals):
            return True
    return False


def write_csv_segment(rows: list[dict], output: Path, segment_label: str) -> None:
    """追加写入一个实验段到 CSV。"""
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        with open(output, "a", newline="", encoding="utf-8") as f:
            f.write("\n")

    fieldnames = list(rows[0].keys())
    with open(output, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        f.write(f"# experiment={segment_label}, method=ADFNet vs {BASELINE_NAME}\n")
        w.writeheader()
        w.writerows(rows)


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="ADFNet 显著性检验（配对 t + FDR Benjamini-Hochberg）")
    p.add_argument("--output", default="result/significance_test.csv",
                   help="输出 CSV 路径 (默认 result/significance_test.csv)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="显著性水平 (默认 0.05)")
    args = p.parse_args()

    output_path = Path(args.output)
    # 如果文件已存在，先清空重写
    if output_path.exists():
        output_path.unlink()

    ran_any = False

    for exp_name, exp_cfg in EXPERIMENTS.items():
        ours = exp_cfg["ours"]
        baseline = exp_cfg["baseline"]
        expected_n = exp_cfg["n_folds"]

        # 提取两组都有的指标
        active_metrics = [m for m in METRICS if m in ours and m in baseline]

        # 检查数据是否已填入
        if not active_metrics or not _is_filled(ours, expected_n):
            print(f"\n[SKIP] {exp_name}: 数据未填入（全 0 或长度不匹配），跳过")
            continue

        # 也检查 baseline
        if not _is_filled(baseline, expected_n):
            print(f"\n[WARN] {exp_name}: baseline 数据全 0，仍执行检验（请确认是否已填入）")

        ran_any = True
        results = paired_ttest_fdr(ours, baseline, active_metrics, args.alpha)
        write_csv_segment(results, output_path, exp_name)

        # 控制台输出
        cv_part, task_part = exp_name.split("_")
        print(f"\n{'='*80}")
        print(f"  {cv_part} {task_part.capitalize()} — 配对 t 检验 "
              f"(ADFNet vs {BASELINE_NAME})")
        print(f"  n_folds={expected_n}, FDR (BH) 校正: "
              f"{len(active_metrics)} 个指标, alpha={args.alpha}")
        print(f"{'='*80}")
        print(f"  {'metric':12s}  {'ours':>8s}  {'base':>8s}  {'diff':>8s}  "
              f"{'t':>7s}  {'p_raw':>12s}  {'sig':>3s}  "
              f"{'p_fdr':>12s}  {'sig':>3s}  {'d':>6s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*7}  {'-'*12}  {'-'*3}  "
              f"{'-'*12}  {'-'*3}  {'-'*6}")
        for r in results:
            raw_mark = "*" if r["sig_raw"] == "Yes" else " "
            fdr_mark = "*" if r["sig_fdr"] == "Yes" else " "
            print(f"  {r['metric']:12s}  "
                  f"{r['mean_ADFNet']:8.4f}  "
                  f"{r[f'mean_{BASELINE_NAME}']:8.4f}  "
                  f"{r['mean_diff']:+8.4f}  "
                  f"{r['t_stat']:+7.3f}  "
                  f"{float(r['p_raw']):12.4e}  {raw_mark:>3s}  "
                  f"{float(r['p_fdr']):12.4e}  {fdr_mark:>3s}  "
                  f"{r['cohen_d']:+6.3f}")

    # ── 汇总 ──
    print(f"\n{'='*80}")
    if ran_any:
        print(f"  结果已写入: {output_path.resolve()}")
    else:
        print("  未执行任何检验，请在 DATA 区域填入数据。")
    print(f"  sig_raw = 未校正 p < {args.alpha}")
    print(f"  sig_fdr = FDR (BH) 校正后 p < {args.alpha}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
