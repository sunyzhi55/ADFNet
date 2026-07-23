"""
显著性检验脚本 — ADFNet vs 对比方法（三方法对比版）
====================================================

对 ADFNet 和最佳消融实验的结果的逐折指标同时执行三种配对检验:
    1. 配对 t 检验 (Paired t-test)          — 参数方法，假设差值正态
    2. Wilcoxon 符号秩检验 (Signed-rank)     — 非参数，假设差值对称
    3. 置换检验 (Permutation test)           — 无分布假设，精确/Monte-Carlo

三种方法均施加 **FDR (Benjamini-Hochberg) 校正**控制 false discovery rate。
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
from itertools import product
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
    "auc": [0.999893, 0.831605, 0.850551, 0.949491, 0.995791, 0.913884, 0.865548, 0.922443, 0.97469, 0.856076, 0.8755666, 0.858171, 0.944182, 0.934783, 0.879227, 0.998261, 0.875499, 0.867428, 0.827158, 0.852148542],
    "acc": [0.992701, 0.828467, 0.791971, 0.916058, 0.908759, 0.771739, 0.80173913, 0.815217, 0.949275, 0.778985507, 0.79710144, 0.815217, 0.880435, 0.931159, 0.851449, 0.949275, 0.826087, 0.761733, 0.75, 0.747292],
    "f1": [0.992701, 0.853582, 0.8, 0.914498, 0.916388, 0.806154, 0.81081, 0.793522, 0.947368, 0.808777, 0.80689655, 0.817204, 0.869565, 0.929368, 0.851986, 0.948529, 0.832168, 0.795031, 0.769, 0.722222],
    "precision": [0.992701, 0.744565, 0.77027, 0.931818, 0.845679, 0.700535, 0.6923, 0.899083, 0.984375, 0.7127071, 0.769736842, 0.808511, 0.956522, 0.954198, 0.848921, 0.962687, 0.804054, 0.695652, 0.714285, 0.80531],
    "recall": [0.992701, 1, 0.832117, 0.89781, 1, 0.949275, 0.97826, 0.710145, 0.913043, 0.9347826, 0.847826087, 0.826087, 0.797101, 0.905797, 0.855072, 0.934783, 0.862319, 0.927536, 0.833333, 0.654676]
}

LOSO_EASY_BASELINE = {
    "acc":[0.930657,0.631387,0.748175,0.864964,0.864964,0.768116,0.789855,0.717391,0.927536,0.641304,0.775362,0.778986,0.902174,0.811594,0.836957,0.858696,0.833333,0.707581,0.692029,0.700361],
    "f1": [0.97491,0.674419,0.745645,0.765343,0.815534,0.826498,0.746667,0.691358,0.962406,0.75841,0.793103,0.7,0.925373,0.888889,0.772152,0.9,0.747967,0.769231,0.764045,0.681481],
    "auc": [0.998082,0.72412,0.778891,0.809526,0.893441,0.908685,0.78421,0.783659,0.974533,0.78702,0.882903,0.697438,0.980361,0.950483,0.830288,0.929952,0.807183,0.831144,0.834751,0.529507]
}

# ═══════════════════════════════════════════════════
# LOSO-Hard 实验（20 折）
# ═══════════════════════════════════════════════════

LOSO_HARD_OURS = {
    "auc": [0.992115, 0.802174, 0.940647, 0.912662, 0.955405, 0.831837, 0.847857, 0.900719, 0.998571, 0.853255465, 0.847755, 0.661146, 0.975488, 0.982477, 0.947415, 0.867398, 0.945427, 0.8637092, 0.779694, 0.8545243],
    "acc": [0.978102, 0.686131, 0.905109, 0.817518248, 0.864964, 0.782143, 0.785714, 0.842294, 0.992857, 0.764285714, 0.8393857, 0.665468, 0.9319, 0.9319, 0.913669, 0.805755, 0.903226, 0.782142, 0.728571, 0.775],
    "f1": [0.978102, 0.739394, 0.905797, 0.845679, 0.846473, 0.801303, 0.72973, 0.826772, 0.992857, 0.798780488, 0.847457, 0.739496, 0.932862, 0.934256, 0.918367, 0.804348, 0.897338, 0.7797833, 0.763975, 0.81305638],
    "precision": [0.978102, 0.632124, 0.899281, 0.73262, 0.980769, 0.736527, 0.987805, 0.913043, 0.992857, 0.696808511, 0.8064516, 0.605505, 0.923077, 0.90604, 0.870968, 0.810219, 0.95935, 0.7883211, 0.675824, 0.695431472],
    "recall": [0.978102, 0.890511, 0.912409, 1, 0.744526, 0.878571, 0.578571, 0.755396, 0.992857, 0.935714286, 0.8928571, 0.94964, 0.942857, 0.964286, 0.971223, 0.798561, 0.842857, 0.7714285, 0.878571, 0.978571429]
}

LOSO_HARD_BASELINE = {
    "acc": [0.934307,0.50365,0.737226,0.532847,0.69708,0.789286,0.717857,0.709677,0.885714,0.785714,0.617857,0.622302,0.777778,0.767025,0.895683,0.816547,0.777778,0.646429,0.653571,0.728571],
    
    
    
    "f1": [0.925267,0.750789,0.814229,0.668317,0.791809,0.818533,0.744479,0.724638,0.826923,0.671463,0.690608,0.690411,0.786164,0.823129,0.921429,0.787645,0.782857,0.697297,0.708609,0.761628],
    
    
    "auc": [0.975119,0.791198,0.89765,0.409505,0.84834,0.862398,0.788622,0.736742,0.844337,0.631122,0.68,0.61886,0.759301,0.850976,0.945655,0.84817,0.891316,0.680663,0.732347,0.768163]
}

# ═══════════════════════════════════════════════════
# KFold-Easy 实验（5 折）
# ═══════════════════════════════════════════════════

# KFOLD_EASY_OURS = {
#     "auc": [0.86561, 0.80545648, 0.78525621851, 0.858703, 0.886034],
#     "acc": [0.817273, 0.7495463, 0.8045249, 0.781307, 0.844062],
#     "f1": [0.80236, 0.7612457, 0.8208955, 0.792777, 0.846702],
#     "precision": [0.873662, 0.7272727, 0.7580398, 0.753268, 0.831874],
#     "recall": [0.741818, 0.798541, 0.8951175, 0.836661, 0.862069]
# }

# KFOLD_EASY_BASELINE = {
#     "auc": [0.88076, 0.736414, 0.680298, 0.674029, 0.844798],
#     "acc": [0.796364, 0.683303, 0.652489, 0.636116, 0.786038],
#     "precision": [0.905473, 0.694231, 0.638752, 0.625418, 0.734027],
#     "recall": [0.661818, 0.655172, 0.703436, 0.678766, 0.896552],
#     "specificity": [0.930909, 0.711434, 0.601449, 0.593466, 0.675725],
#     "f1": [0.764706, 0.674136, 0.669535, 0.651001, 0.80719],
#     "kappa": [0.592727, 0.366606, 0.304913, 0.272232, 0.572162],
#     "balance_accuracy": [0.796364, 0.683303, 0.652443, 0.636116, 0.786138]
# }

# # ═══════════════════════════════════════════════════
# # KFold-Hard 实验（5 折）
# # ═══════════════════════════════════════════════════

# KFOLD_HARD_OURS = {
#     "auc": [0.816624, 0.788326, 0.8758275, 0.859989, 0.789087],
#     "acc": [0.764228, 0.741906, 0.8103757, 0.798198, 0.766397],
#     "f1": [0.763373, 0.747581, 0.8212479, 0.80789, 0.779661],
#     "precision": [0.766849, 0.731497, 0.7767145, 0.770867, 0.738363],
#     "recall": [0.759928, 0.764388, 0.871199, 0.848649, 0.825853]
# }

# KFOLD_HARD_BASELINE = {
#     # "acc": [0.7592, 0.628, 0.8286, 0.6764, 0.6251], # MIGCN
#     # "precision": [0.7545, 0.6024, 0.7843, 0.6301, 0.6004], # MIGCN
#     # "recall": [0.7688, 0.7531, 0.9068, 0.8551, 0.7481], # MIGCN
#     # "specificity": [0.7496, 0.5029, 0.7502, 0.4975, 0.5021], # MIGCN
#     # "f1": [0.7616, 0.6693, 0.8411, 0.7256, 0.6662], # MIGCN
#     # # "auc": [0.7204, 0.5657, 0.722, 0.7557, 0.4916], # MIGCN
#     # "auc": [0.794377, 0.803287, 0.719474, 0.75053, 0.639227], # AFM-CIR
#     # "kappa": [0.5184, 0.256, 0.6571, 0.3527, 0.2502], # MIGCN


    
#     "acc": [0.759190417, 0.627983539, 0.788128931, 0.676410045, 0.625102881],
#     "precision": [0.75445705, 0.60236998, 0.784299859, 0.630078836, 0.600396301],
#     "recall": [0.768786127, 0.75308642, 0.90678659, 0.855144033, 0.748148148],
#     "f1": [0.761554192, 0.669348939, 0.803069054, 0.725558659, 0.666178087],
#     # "auc": [0.7204, 0.5657, 0.722, 0.7557, 0.4916],
#     "auc": [0.794377, 0.803287, 0.719474, 0.75053, 0.639227], # AFM-CIR

#     "kappa": [0.5184, 0.256, 0.6571, 0.3527, 0.2502]


# }

# ═══════════════════════════════════════════════════
# 实验注册表（控制运行顺序与显示名称）
# ═══════════════════════════════════════════════════

EXPERIMENTS: dict[str, dict] = {
    "LOSO_easy":  {"ours": LOSO_EASY_OURS,  "baseline": LOSO_EASY_BASELINE,  "n_folds": 20},
    "LOSO_hard":  {"ours": LOSO_HARD_OURS,  "baseline": LOSO_HARD_BASELINE,  "n_folds": 20},
    # "KFold_easy": {"ours": KFOLD_EASY_OURS, "baseline": KFOLD_EASY_BASELINE, "n_folds": 5},
    # "KFold_hard": {"ours": KFOLD_HARD_OURS, "baseline": KFOLD_HARD_BASELINE, "n_folds": 5},
}


# ══════════════════════════════════════════════════════════════
# 核心检验逻辑
# ══════════════════════════════════════════════════════════════

# ── 置换检验配置 ──
PERM_EXACT_MAX_N = 20          # n <= 此值时精确枚举 (2^n)
PERM_MONTE_CARLO_N = 100_000   # n > 此值时 Monte-Carlo 迭代次数
PERM_SEED = 42                 # Monte-Carlo 可复现种子


def _permutation_test(diff: np.ndarray, n_perm: int = PERM_MONTE_CARLO_N,
                      seed: int = PERM_SEED) -> tuple[float, float]:
    """配对置换检验（双侧）。

    H0 下每对差值的符号等概率翻转。
    统计量: T = mean(diff)（等价于 sum）。
    p = P(|T_perm| >= |T_obs|)。

    n <= PERM_EXACT_MAX_N 时精确枚举 2^n 种符号组合;
    否则 Monte-Carlo 随机采样。
    """
    n = len(diff)
    t_obs = np.abs(np.mean(diff))

    if t_obs < 1e-15:
        return 0.0, 1.0

    if n <= PERM_EXACT_MAX_N:
        # 精确枚举: 2^n 种 ±1 组合
        count = 0
        total = 2 ** n
        for signs in product([-1, 1], repeat=n):
            t_perm = np.abs(np.mean(diff * np.array(signs)))
            if t_perm >= t_obs - 1e-12:
                count += 1
        p_val = count / total
    else:
        # Monte-Carlo
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(n_perm, n))
        t_perms = np.abs((signs * diff).mean(axis=1))
        p_val = float(np.mean(t_perms >= t_obs - 1e-12))

    return t_obs, p_val


def _bh_correction(p_values: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正，返回校正后 p 值数组。"""
    m = len(p_values)
    if m == 0:
        return np.array([])
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values, dtype=np.float64)[sorted_indices]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    p_bh = sorted_p * m / ranks
    # 强制单调
    for i in range(m - 2, -1, -1):
        p_bh[i] = min(p_bh[i], p_bh[i + 1])
    p_bh = np.clip(p_bh, 0.0, 1.0)
    # 还原顺序
    out = np.empty(m, dtype=np.float64)
    out[sorted_indices] = p_bh
    return out


def three_method_test(
    ours: dict[str, list[float]],
    baseline: dict[str, list[float]],
    metrics: list[str],
    alpha: float = 0.05,
) -> list[dict]:
    """对每个指标同时做三种配对检验，各施加 BH FDR 校正。

    Returns:
        每个指标一行结果 dict。
    """
    n_tests = len(metrics)

    # ── 第一轮: 逐指标计算三种检验的 raw p 值 ──
    raw_rows: list[dict] = []
    for m in metrics:
        a = np.asarray(ours[m], dtype=np.float64)
        b = np.asarray(baseline[m], dtype=np.float64)
        assert len(a) == len(b), f"{m}: 两组 fold 数不一致 ({len(a)} vs {len(b)})"
        n = len(a)
        diff = a - b

        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        mean_d, std_d = float(np.mean(diff)), float(np.std(diff, ddof=1))

        # --- 1. Paired t-test ---
        if std_d < 1e-15:
            t_stat, p_ttest = 0.0, 1.0
        else:
            t_stat, p_ttest = stats.ttest_rel(a, b)
            t_stat, p_ttest = float(t_stat), float(p_ttest)

        # --- 2. Wilcoxon signed-rank test ---
        # 当所有差值为 0 时 wilcoxon 会报错，需特判
        if np.all(np.abs(diff) < 1e-15):
            w_stat, p_wilcoxon = 0.0, 1.0
        else:
            # method='auto': n<=50 用精确分布, 否则正态近似
            w_result = stats.wilcoxon(a, b, alternative='two-sided', method='auto')
            w_stat, p_wilcoxon = float(w_result.statistic), float(w_result.pvalue)

        # --- 3. Permutation test ---
        perm_t, p_perm = _permutation_test(diff)

        # --- 效应量 ---
        cohen_d = mean_d / std_d if std_d > 1e-15 else 0.0
        # rank-biserial correlation (Wilcoxon 效应量): r = 1 - 2W/(n(n+1)/2)
        # 其中 W = sum of positive ranks; scipy 返回的 statistic 就是 W+
        max_w = n * (n + 1) / 2
        rank_biserial = 1.0 - 2.0 * w_stat / max_w if max_w > 0 else 0.0

        se = std_d / np.sqrt(n) if n > 0 else 0.0
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 0.0

        raw_rows.append({
            "metric": m, "n_folds": n, "df": df,
            "mean_a": mean_a, "mean_b": mean_b,
            "mean_d": mean_d, "std_d": std_d,
            # t-test
            "t_stat": t_stat, "p_ttest": p_ttest,
            # wilcoxon
            "w_stat": w_stat, "p_wilcoxon": p_wilcoxon,
            # permutation
            "perm_t": perm_t, "p_perm": p_perm,
            # effect sizes
            "cohen_d": cohen_d, "rank_biserial": rank_biserial,
            # CI (基于 t 分布)
            "ci_lo": mean_d - t_crit * se,
            "ci_hi": mean_d + t_crit * se,
        })

    # ── 第二轮: 分别对三种方法的 p 值做 BH FDR 校正 ──
    p_ttest_all = [r["p_ttest"] for r in raw_rows]
    p_wilc_all = [r["p_wilcoxon"] for r in raw_rows]
    p_perm_all = [r["p_perm"] for r in raw_rows]

    fdr_ttest = _bh_correction(p_ttest_all)
    fdr_wilc = _bh_correction(p_wilc_all)
    fdr_perm = _bh_correction(p_perm_all)

    # ── 组装结果 ──
    results: list[dict] = []
    for i, r in enumerate(raw_rows):
        results.append({
            "metric": r["metric"],
            "n_folds": r["n_folds"],
            "mean_ADFNet": round(r["mean_a"], 6),
            f"mean_{BASELINE_NAME}": round(r["mean_b"], 6),
            "mean_diff": round(r["mean_d"], 6),
            "std_diff": round(r["std_d"], 6),
            # t-test
            "t_stat": round(r["t_stat"], 4),
            "p_ttest_raw": f"{r['p_ttest']:.6e}",
            "sig_ttest_raw": "Yes" if r["p_ttest"] < alpha else "No",
            "p_ttest_fdr": f"{fdr_ttest[i]:.6e}",
            "sig_ttest_fdr": "Yes" if fdr_ttest[i] < alpha else "No",
            # wilcoxon
            "w_stat": round(r["w_stat"], 4),
            "p_wilc_raw": f"{r['p_wilcoxon']:.6e}",
            "sig_wilc_raw": "Yes" if r["p_wilcoxon"] < alpha else "No",
            "p_wilc_fdr": f"{fdr_wilc[i]:.6e}",
            "sig_wilc_fdr": "Yes" if fdr_wilc[i] < alpha else "No",
            # permutation
            "p_perm_raw": f"{r['p_perm']:.6e}",
            "sig_perm_raw": "Yes" if r["p_perm"] < alpha else "No",
            "p_perm_fdr": f"{fdr_perm[i]:.6e}",
            "sig_perm_fdr": "Yes" if fdr_perm[i] < alpha else "No",
            # effect sizes & CI
            "cohen_d": round(r["cohen_d"], 4),
            "rank_biserial": round(r["rank_biserial"], 4),
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
    p = argparse.ArgumentParser(
        description="ADFNet 显著性检验（配对 t / Wilcoxon / Permutation + FDR BH 校正）")
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
        results = three_method_test(ours, baseline, active_metrics, args.alpha)
        write_csv_segment(results, output_path, exp_name)

        # 控制台输出
        cv_part, task_part = exp_name.split("_")
        print(f"\n{'='*100}")
        print(f"  {cv_part} {task_part.capitalize()} — 三方法对比检验 "
              f"(ADFNet vs {BASELINE_NAME})")
        print(f"  n_folds={expected_n}, FDR (BH) 校正: "
              f"{len(active_metrics)} 个指标, alpha={args.alpha}")
        print(f"{'='*100}")

        # 表头
        hdr = (f"  {'metric':10s}  {'diff':>8s}  "
               f"{'p_t':>10s} {'sig':>3s}  "
               f"{'p_t_fdr':>10s} {'sig':>3s}  "
               f"{'p_wilc':>10s} {'sig':>3s}  "
               f"{'p_wilc_fdr':>10s} {'sig':>3s}  "
               f"{'p_perm':>10s} {'sig':>3s}  "
               f"{'p_perm_fdr':>10s} {'sig':>3s}  "
               f"{'d':>6s} {'r_rb':>6s}")
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")

        for r in results:
            t_mark = "*" if r["sig_ttest_raw"] == "Yes" else " "
            t_fdr_mark = "*" if r["sig_ttest_fdr"] == "Yes" else " "
            w_mark = "*" if r["sig_wilc_raw"] == "Yes" else " "
            w_fdr_mark = "*" if r["sig_wilc_fdr"] == "Yes" else " "
            p_mark = "*" if r["sig_perm_raw"] == "Yes" else " "
            p_fdr_mark = "*" if r["sig_perm_fdr"] == "Yes" else " "

            print(f"  {r['metric']:10s}  {r['mean_diff']:+8.4f}  "
                  f"{float(r['p_ttest_raw']):10.4e} {t_mark:>3s}  "
                  f"{float(r['p_ttest_fdr']):10.4e} {t_fdr_mark:>3s}  "
                  f"{float(r['p_wilc_raw']):10.4e} {w_mark:>3s}  "
                  f"{float(r['p_wilc_fdr']):10.4e} {w_fdr_mark:>3s}  "
                  f"{float(r['p_perm_raw']):10.4e} {p_mark:>3s}  "
                  f"{float(r['p_perm_fdr']):10.4e} {p_fdr_mark:>3s}  "
                  f"{r['cohen_d']:+6.3f} {r['rank_biserial']:+6.3f}")

    # ── 汇总 ──
    print(f"\n{'='*100}")
    if ran_any:
        print(f"  结果已写入: {output_path.resolve()}")
    else:
        print("  未执行任何检验，请在 DATA 区域填入数据。")
    print(f"  方法说明:")
    print(f"    p_t / p_t_fdr     = Paired t-test (raw / BH 校正)")
    print(f"    p_wilc / p_wilc_fdr = Wilcoxon signed-rank (raw / BH 校正)")
    print(f"    p_perm / p_perm_fdr = Permutation test (raw / BH 校正)")
    print(f"    d = Cohen's d, r_rb = rank-biserial correlation")
    print(f"    * 表示 p < {args.alpha}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
