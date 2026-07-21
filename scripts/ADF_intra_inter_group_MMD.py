#!/usr/bin/env python
"""
ADF Intra-group / Inter-group MMD Analysis
============================================
Computes per-subject MMD under two perspectives for each task (easy/hard):

  1. Intra-group (组内): Each subject's features vs the SAME-state group baseline.
     - Alert subject vs group alert features (leave-one-out)
     - Fatigue subject vs group fatigue features (leave-one-out)
     Measures cross-subject invariance within the same cognitive state.

  2. Inter-group (组间): Each subject's features vs the OPPOSITE-state group baseline.
     - Alert subject vs group fatigue features
     - Fatigue subject vs group alert features
     Measures cross-state discriminability.

A good feature representation should yield LOW intra-group MMD (invariant across
subjects in the same state) and HIGH inter-group MMD (separable between states).

Output:
  - Per-subject CSV with intra/inter MMD for both ADF and raw gaze features
  - Summary CSV with mean ± std across subjects
  - Bar chart visualisation (optional)

Usage:
  python scripts/ADF_intra_inter_group_MMD.py --data-root <path>
  python scripts/ADF_intra_inter_group_MMD.py  # uses config default
"""

from __future__ import annotations

import sys
import os
import io
import ast
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import ttest_rel, wilcoxon

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from data.io import discover_sequences, filter_sequences_by_task, iter_jsonl, parse_points
from data.features import compute_adf_features


# ===================================================================
# Feature Extraction (same as invariance_quantification_MMD.py)
# ===================================================================

def extract_eye_records(records: list[dict]) -> np.ndarray:
    """Extract flattened facial_landmark_35 as eye features.

    Returns a 2D array of shape (N, D) where D is fixed across all frames.
    """
    D = 0
    for rec in records:
        pts = parse_points(rec.get("facial_landmark_35"))
        if pts:
            D = np.array(pts, dtype=np.float32).flatten().shape[0]
            break

    if D == 0:
        return np.zeros((len(records), 1), dtype=np.float32)

    frames: list[np.ndarray] = []
    for rec in records:
        pts = parse_points(rec.get("facial_landmark_35"))
        if pts:
            flat = np.array(pts, dtype=np.float32).flatten()
            if flat.shape[0] < D:
                flat = np.pad(flat, (0, D - flat.shape[0]))
            elif flat.shape[0] > D:
                flat = flat[:D]
            frames.append(flat)
        else:
            frames.append(np.zeros(D, dtype=np.float32))

    if not frames:
        return np.zeros((0, D), dtype=np.float32)
    return np.stack(frames, axis=0)


def window_level_features(
    seq_records: list[dict],
    task_type: str,
    window_size: int,
    stride: int,
    local_mean_size: int = 16,
):
    """Return (adf_feat, gaze_feat) at window level.

    adf_feat  : [n_windows, 2]  — mean/std of drift (per-sample norm)
    gaze_feat : [n_windows, 2*D] — mean/std of eye landmarks
    """
    adf = compute_adf_features(seq_records, task_type, local_mean_size, per_sample_norm=True)
    raw_feature = extract_eye_records(seq_records)

    T = len(adf)
    if len(raw_feature) < T:
        pad = np.zeros((T - len(raw_feature), raw_feature.shape[1]), dtype=np.float32)
        raw_feature = np.concatenate([raw_feature, pad], axis=0)
    elif len(raw_feature) > T:
        raw_feature = raw_feature[:T]

    adf_windows, raw_feature_windows = [], []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        # ADF: mean & std of drift (ch0)
        adf_win = adf[start:end]  # [W, 3]
        adf_windows.append(np.array([
            adf_win[:, 0].mean(), adf_win[:, 0].std(),
            # adf_win[:, 1].mean(), adf_win[:, 1].std(),
            # adf_win[:, 2].mean(), adf_win[:, 2].std(),
        ], dtype=np.float32))
        # Raw feature: mean & std of eye landmarks (D-dim)
        raw_feature_win = raw_feature[start:end]  # [W, D]
        raw_feature_windows.append(
            np.concatenate([raw_feature_win.mean(axis=0), raw_feature_win.std(axis=0)])
        )

    D = raw_feature.shape[1] if len(raw_feature) > 0 else 0
    feat_dim = D * 2
    if not adf_windows:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, feat_dim), dtype=np.float32))
    return np.stack(adf_windows), np.stack(raw_feature_windows)


def extract_features_for_sequences(sequences, window_size, stride, local_mean_size):
    """Extract ADF and raw gaze window-level features for a list of sequences."""
    all_adf, all_gaze = [], []
    for seq in sequences:
        records = list(iter_jsonl(seq.path))
        if len(records) < window_size:
            continue
        adf_feat, gaze_feat = window_level_features(
            records, seq.task_type, window_size, stride, local_mean_size,
        )
        if len(adf_feat) > 0:
            all_adf.append(adf_feat)
            all_gaze.append(gaze_feat)
    if not all_adf:
        return None, None
    return np.concatenate(all_adf), np.concatenate(all_gaze)


# ===================================================================
# MMD (RBF kernel, median heuristic)
# ===================================================================

def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    return np.exp(-dists / (2.0 * sigma ** 2))


def compute_mmd(P: np.ndarray, Q: np.ndarray) -> float:
    """Biased MMD estimate with RBF kernel (median heuristic bandwidth)."""
    P, Q = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64)
    if len(P) < 2 or len(Q) < 2:
        return 0.0
    combined = np.concatenate([P, Q], axis=0)
    dists = pdist(combined)
    sigma = float(np.median(dists)) if len(dists) > 0 and np.median(dists) > 0 else 1.0

    K_pp = rbf_kernel(P, P, sigma)
    K_qq = rbf_kernel(Q, Q, sigma)
    K_pq = rbf_kernel(P, Q, sigma)

    m, n = len(P), len(Q)
    mmd2 = K_pp.sum() / (m * m) + K_qq.sum() / (n * n) - 2.0 * K_pq.sum() / (m * n)
    return float(np.sqrt(max(mmd2, 0.0)))


# ===================================================================
# Core Analysis: Intra-group & Inter-group MMD
# ===================================================================

def compute_intra_inter_mmd(
    adf_by_subj: dict[str, np.ndarray],
    gaze_by_subj: dict[str, np.ndarray],
    adf_opposite: dict[str, np.ndarray],
    gaze_opposite: dict[str, np.ndarray],
    state_name: str,
    opposite_name: str,
    task_name: str,
) -> pd.DataFrame:
    """Compute per-subject intra-group and inter-group MMD.

    Intra-group: subject vs same-state group (leave-one-out).
    Inter-group: subject vs opposite-state group (all subjects pooled).

    Parameters
    ----------
    adf_by_subj : dict[subject -> ADF features] for current state
    gaze_by_subj : dict[subject -> gaze features] for current state
    adf_opposite : dict[subject -> ADF features] for opposite state
    gaze_opposite : dict[subject -> gaze features] for opposite state
    state_name : e.g. "alert" or "fatigue"
    opposite_name : e.g. "fatigue" or "alert"
    task_name : e.g. "easy" or "hard"

    Returns
    -------
    DataFrame with columns: Subject, State, Task, N_Windows,
        ADF_Intra_MMD, ADF_Inter_MMD, Gaze_Intra_MMD, Gaze_Inter_MMD,
        ADF_Separability, Gaze_Separability
    """
    subjects = sorted(adf_by_subj.keys())
    opposite_subjects = sorted(adf_opposite.keys())

    # Pool opposite-state features (full group, no leave-one-out needed)
    adf_opp_pooled = np.concatenate([adf_opposite[s] for s in opposite_subjects])
    gaze_opp_pooled = np.concatenate([gaze_opposite[s] for s in opposite_subjects])

    rows = []
    for subj in subjects:
        subj_adf = adf_by_subj[subj]
        subj_gaze = gaze_by_subj[subj]

        # --- Intra-group: leave-one-out within same state ---
        others_adf = np.concatenate([adf_by_subj[s] for s in subjects if s != subj])
        others_gaze = np.concatenate([gaze_by_subj[s] for s in subjects if s != subj])

        adf_intra = compute_mmd(subj_adf, others_adf)
        gaze_intra = compute_mmd(subj_gaze, others_gaze)

        # --- Inter-group: subject vs opposite-state group ---
        adf_inter = compute_mmd(subj_adf, adf_opp_pooled)
        gaze_inter = compute_mmd(subj_gaze, gaze_opp_pooled)

        # Separability ratio: inter / intra (higher = better discrimination)
        adf_sep = adf_inter / (adf_intra + 1e-10)
        gaze_sep = gaze_inter / (gaze_intra + 1e-10)

        rows.append({
            "Subject": subj,
            "State": state_name,
            "Task": task_name,
            "N_Windows": len(subj_adf),
            "ADF_Intra_MMD": adf_intra,
            "ADF_Inter_MMD": adf_inter,
            "Gaze_Intra_MMD": gaze_intra,
            "Gaze_Inter_MMD": gaze_inter,
            "ADF_Separability": adf_sep,
            "Gaze_Separability": gaze_sep,
        })

    return pd.DataFrame(rows)


# ===================================================================
# Visualisation
# ===================================================================

def plot_grouped_bar(results_df: pd.DataFrame, task_name: str, output_dir: Path):
    """Grouped bar chart: intra vs inter MMD for ADF and gaze, per subject."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for state in ["alert", "fatigue"]:
        state_df = results_df[results_df["State"] == state].reset_index(drop=True)
        if state_df.empty:
            continue

        n = len(state_df)
        x = np.arange(n)
        w = 0.2

        fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 5))
        ax.bar(x - 1.5 * w, state_df["ADF_Intra_MMD"], w,
               label="ADF Intra-group", color="steelblue", alpha=0.85)
        ax.bar(x - 0.5 * w, state_df["ADF_Inter_MMD"], w,
               label="ADF Inter-group", color="darkblue", alpha=0.85)
        ax.bar(x + 0.5 * w, state_df["Gaze_Intra_MMD"], w,
               label="Gaze Intra-group", color="coral", alpha=0.85)
        ax.bar(x + 1.5 * w, state_df["Gaze_Inter_MMD"], w,
               label="Gaze Inter-group", color="darkred", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(state_df["Subject"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("MMD")
        ax.set_title(f"{task_name.upper()} Task — {state.capitalize()} State\n"
                     f"Intra-group (same-state) vs Inter-group (cross-state) MMD")
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.tight_layout()

        fname = f"mmd_intra_inter_{task_name}_{state}.png"
        plt.savefig(output_dir / fname, dpi=200)
        plt.close()
        print(f"  [saved] {fname}")


def plot_separability_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing ADF vs Gaze separability ratio across conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by (Task, State) and compute mean separability
    groups = results_df.groupby(["Task", "State"]).agg(
        ADF_Sep_Mean=("ADF_Separability", "mean"),
        ADF_Sep_Std=("ADF_Separability", "std"),
        Gaze_Sep_Mean=("Gaze_Separability", "mean"),
        Gaze_Sep_Std=("Gaze_Separability", "std"),
    ).reset_index()

    if groups.empty:
        return

    labels = [f"{row['Task']}_{row['State']}" for _, row in groups.iterrows()]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, groups["ADF_Sep_Mean"], w,
           yerr=groups["ADF_Sep_Std"], label="ADF", color="steelblue",
           capsize=4, alpha=0.85)
    ax.bar(x + w / 2, groups["Gaze_Sep_Mean"], w,
           yerr=groups["Gaze_Sep_Std"], label="Raw Gaze", color="coral",
           capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Separability Ratio (Inter / Intra MMD)")
    ax.set_title("Feature Separability: ADF vs Raw Gaze\n(higher = better state discrimination)")
    ax.legend()
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "separability_comparison.png", dpi=200)
    plt.close()
    print(f"  [saved] separability_comparison.png")


# ===================================================================
# Main
# ===================================================================

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="ADF Intra-group / Inter-group MMD Analysis")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/shenxy/Data/Process0620_calibrate",
                        help="Data directory (default: from configs/default.yaml)")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--local-mean-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="intra_inter_mmd_results")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    # ---- resolve data root ----
    data_root = args.data_root
    if data_root is None:
        import yaml
        cfg_path = _project_root / "configs" / "default.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        data_root = cfg["data"]["root"]
    data_root = Path(data_root)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- discover sequences ----
    print(f"Data root: {data_root}")
    all_seqs = discover_sequences(data_root)
    subjects = sorted(set(s.subject_id for s in all_seqs))
    print(f"Subjects: {len(subjects)}")
    print(f"Total sequences: {len(all_seqs)}")

    # ---- group by 4 conditions: (task_type, label_name) ----
    CONDITIONS = [
        ("easy", "alert"),
        ("easy", "sleepy"),
        ("hard", "alert"),
        ("hard", "sleepy"),
    ]
    COND_LABELS = {
        ("easy", "alert"): "easy_alert",
        ("easy", "sleepy"): "easy_fatigue",
        ("hard", "alert"): "hard_alert",
        ("hard", "sleepy"): "hard_fatigue",
    }

    seqs_by_cond: dict[tuple, dict[str, list]] = {}
    for (task, label_name) in CONDITIONS:
        cond_seqs = [s for s in all_seqs
                     if s.task_type == task and s.label_name == label_name]
        # also accept "sleep" as "sleepy"
        if label_name == "sleepy":
            cond_seqs_extra = [s for s in all_seqs
                               if s.task_type == task and s.label_name == "sleep"]
            cond_seqs = cond_seqs + cond_seqs_extra
        by_subj = defaultdict(list)
        for s in cond_seqs:
            by_subj[s.subject_id].append(s)
        seqs_by_cond[(task, label_name)] = by_subj
        n_seqs = sum(len(v) for v in by_subj.values())
        print(f"  {COND_LABELS[(task, label_name)]:>14s}: "
              f"{len(by_subj)} subjects, {n_seqs} sequences")

    # ---- extract features per condition per subject ----
    print("\n" + "=" * 70)
    print("  Extracting features...")
    print("=" * 70)

    features_by_cond: dict[tuple, dict[str, dict[str, np.ndarray]]] = {}
    # features_by_cond[(task, label)] = {"adf": {subj: arr}, "gaze": {subj: arr}}

    for (task, label_name) in CONDITIONS:
        cond_label = COND_LABELS[(task, label_name)]
        by_subj = seqs_by_cond[(task, label_name)]
        if not by_subj:
            print(f"  [skip] {cond_label}: no data")
            continue

        adf_by_subj = {}
        gaze_by_subj = {}
        for subj in sorted(by_subj.keys()):
            adf_feat, gaze_feat = extract_features_for_sequences(
                by_subj[subj], args.window_size, args.stride, args.local_mean_size,
            )
            if adf_feat is not None:
                adf_by_subj[subj] = adf_feat
                gaze_by_subj[subj] = gaze_feat

        if adf_by_subj:
            features_by_cond[(task, label_name)] = {
                "adf": adf_by_subj,
                "gaze": gaze_by_subj,
            }
            print(f"  {cond_label}: {len(adf_by_subj)} subjects extracted")

    # ---- compute intra/inter MMD ----
    print("\n" + "=" * 70)
    print("  Intra-group / Inter-group MMD Analysis")
    print("=" * 70)

    all_results = []

    for task_name in ["easy", "hard"]:
        alert_key = (task_name, "alert")
        fatigue_key = (task_name, "sleepy")

        if alert_key not in features_by_cond or fatigue_key not in features_by_cond:
            print(f"\n  [skip] {task_name}: missing alert or fatigue data")
            continue

        alert_data = features_by_cond[alert_key]
        fatigue_data = features_by_cond[fatigue_key]

        # Need at least 2 subjects in each state for leave-one-out intra-group
        if len(alert_data["adf"]) < 2:
            print(f"\n  [skip] {task_name}: < 2 alert subjects")
            continue
        if len(fatigue_data["adf"]) < 2:
            print(f"\n  [skip] {task_name}: < 2 fatigue subjects")
            continue

        print(f"\n{'#' * 70}")
        print(f"# Task: {task_name.upper()}")
        print(f"#   Alert subjects:   {len(alert_data['adf'])}")
        print(f"#   Fatigue subjects: {len(fatigue_data['adf'])}")
        print(f"{'#' * 70}")

        # --- Alert: intra (vs other alert) & inter (vs fatigue group) ---
        print(f"\n  --- Alert State ---")
        alert_df = compute_intra_inter_mmd(
            adf_by_subj=alert_data["adf"],
            gaze_by_subj=alert_data["gaze"],
            adf_opposite=fatigue_data["adf"],
            gaze_opposite=fatigue_data["gaze"],
            state_name="alert",
            opposite_name="fatigue",
            task_name=task_name,
        )
        all_results.append(alert_df)

        for _, row in alert_df.iterrows():
            print(f"    {row['Subject']:>6s}  N={row['N_Windows']:>4d}  "
                  f"ADF_intra={row['ADF_Intra_MMD']:.4f}  "
                  f"ADF_inter={row['ADF_Inter_MMD']:.4f}  "
                  f"Sep={row['ADF_Separability']:.2f}x  |  "
                  f"Gaze_intra={row['Gaze_Intra_MMD']:.4f}  "
                  f"Gaze_inter={row['Gaze_Inter_MMD']:.4f}  "
                  f"Sep={row['Gaze_Separability']:.2f}x")

        # --- Fatigue: intra (vs other fatigue) & inter (vs alert group) ---
        print(f"\n  --- Fatigue State ---")
        fatigue_df = compute_intra_inter_mmd(
            adf_by_subj=fatigue_data["adf"],
            gaze_by_subj=fatigue_data["gaze"],
            adf_opposite=alert_data["adf"],
            gaze_opposite=alert_data["gaze"],
            state_name="fatigue",
            opposite_name="alert",
            task_name=task_name,
        )
        all_results.append(fatigue_df)

        for _, row in fatigue_df.iterrows():
            print(f"    {row['Subject']:>6s}  N={row['N_Windows']:>4d}  "
                  f"ADF_intra={row['ADF_Intra_MMD']:.4f}  "
                  f"ADF_inter={row['ADF_Inter_MMD']:.4f}  "
                  f"Sep={row['ADF_Separability']:.2f}x  |  "
                  f"Gaze_intra={row['Gaze_Intra_MMD']:.4f}  "
                  f"Gaze_inter={row['Gaze_Inter_MMD']:.4f}  "
                  f"Sep={row['Gaze_Separability']:.2f}x")

    if not all_results:
        print("\n  No valid results. Check data availability.")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # ---- per-task summary ----
    print(f"\n{'=' * 70}")
    print("  Summary Statistics")
    print(f"{'=' * 70}")

    summary_rows = []
    for (task_name, state_name), grp in results_df.groupby(["Task", "State"]):
        row = {
            "Task": task_name,
            "State": state_name,
            "N_Subjects": len(grp),
            "ADF_Intra_Mean": grp["ADF_Intra_MMD"].mean(),
            "ADF_Intra_Std": grp["ADF_Intra_MMD"].std(),
            "ADF_Inter_Mean": grp["ADF_Inter_MMD"].mean(),
            "ADF_Inter_Std": grp["ADF_Inter_MMD"].std(),
            "ADF_Sep_Mean": grp["ADF_Separability"].mean(),
            "ADF_Sep_Std": grp["ADF_Separability"].std(),
            "Gaze_Intra_Mean": grp["Gaze_Intra_MMD"].mean(),
            "Gaze_Intra_Std": grp["Gaze_Intra_MMD"].std(),
            "Gaze_Inter_Mean": grp["Gaze_Inter_MMD"].mean(),
            "Gaze_Inter_Std": grp["Gaze_Inter_MMD"].std(),
            "Gaze_Sep_Mean": grp["Gaze_Separability"].mean(),
            "Gaze_Sep_Std": grp["Gaze_Separability"].std(),
        }
        summary_rows.append(row)

        print(f"\n  {task_name.upper()} / {state_name.capitalize()} "
              f"(n={len(grp)} subjects):")
        print(f"    ADF  — Intra: {row['ADF_Intra_Mean']:.4f} ± {row['ADF_Intra_Std']:.4f}  "
              f"Inter: {row['ADF_Inter_Mean']:.4f} ± {row['ADF_Inter_Std']:.4f}  "
              f"Sep: {row['ADF_Sep_Mean']:.2f}x")
        print(f"    Gaze — Intra: {row['Gaze_Intra_Mean']:.4f} ± {row['Gaze_Intra_Std']:.4f}  "
              f"Inter: {row['Gaze_Inter_Mean']:.4f} ± {row['Gaze_Inter_Std']:.4f}  "
              f"Sep: {row['Gaze_Sep_Mean']:.2f}x")

    summary_df = pd.DataFrame(summary_rows)

    # ---- statistical tests: intra vs inter (paired, per condition) ----
    print(f"\n{'=' * 70}")
    print("  Paired Statistical Tests (Intra vs Inter MMD)")
    print(f"{'=' * 70}")

    stat_rows = []
    for (task_name, state_name), grp in results_df.groupby(["Task", "State"]):
        if len(grp) < 3:
            continue
        # ADF: intra vs inter
        adf_t, adf_p = ttest_rel(grp["ADF_Intra_MMD"], grp["ADF_Inter_MMD"])
        try:
            adf_w, adf_wp = wilcoxon(grp["ADF_Intra_MMD"], grp["ADF_Inter_MMD"])
        except Exception:
            adf_wp = float("nan")
        # Gaze: intra vs inter
        gaze_t, gaze_p = ttest_rel(grp["Gaze_Intra_MMD"], grp["Gaze_Inter_MMD"])
        try:
            gaze_w, gaze_wp = wilcoxon(grp["Gaze_Intra_MMD"], grp["Gaze_Inter_MMD"])
        except Exception:
            gaze_wp = float("nan")

        stat_rows.append({
            "Task": task_name, "State": state_name,
            "ADF_t": adf_t, "ADF_p": adf_p, "ADF_Wilcoxon_p": adf_wp,
            "Gaze_t": gaze_t, "Gaze_p": gaze_p, "Gaze_Wilcoxon_p": gaze_wp,
        })

        print(f"\n  {task_name.upper()} / {state_name.capitalize()}:")
        print(f"    ADF  paired t-test: t={adf_t:.3f}, p={adf_p:.4e}  "
              f"{'***' if adf_p < 0.001 else '**' if adf_p < 0.01 else '*' if adf_p < 0.05 else 'ns'}")
        print(f"    ADF  Wilcoxon:      p={adf_wp:.4e}")
        print(f"    Gaze paired t-test: t={gaze_t:.3f}, p={gaze_p:.4e}  "
              f"{'***' if gaze_p < 0.001 else '**' if gaze_p < 0.01 else '*' if gaze_p < 0.05 else 'ns'}")
        print(f"    Gaze Wilcoxon:      p={gaze_wp:.4e}")

    # ---- save CSVs ----
    results_df.to_csv(output_dir / "intra_inter_mmd_per_subject.csv",
                      index=False, float_format="%.6f")
    print(f"\n  [saved] intra_inter_mmd_per_subject.csv")

    summary_df.to_csv(output_dir / "intra_inter_mmd_summary.csv",
                      index=False, float_format="%.6f")
    print(f"  [saved] intra_inter_mmd_summary.csv")

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(output_dir / "intra_inter_mmd_statistics.csv",
                       index=False, float_format="%.6f")
        print(f"  [saved] intra_inter_mmd_statistics.csv")

    # ---- plots ----
    if not args.no_plot:
        print(f"\n  Generating plots...")
        try:
            for task_name in ["easy", "hard"]:
                task_df = results_df[results_df["Task"] == task_name]
                if not task_df.empty:
                    plot_grouped_bar(task_df, task_name, output_dir)
        except Exception as e:
            print(f"    [skip] bar plots: {e}")
        try:
            plot_separability_comparison(results_df, output_dir)
        except Exception as e:
            print(f"    [skip] separability plot: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
2026.07.21 Result:

======================================================================
  Summary Statistics
======================================================================

  EASY / Alert (n=20 subjects):
    ADF  — Intra: 0.1962 ± 0.1019  Inter: 0.2156 ± 0.1150  Sep: 1.24x
    Gaze — Intra: 0.6364 ± 0.1411  Inter: 0.6181 ± 0.1406  Sep: 0.98x

  EASY / Fatigue (n=20 subjects):
    ADF  — Intra: 0.2154 ± 0.1548  Inter: 0.2422 ± 0.1529  Sep: 1.31x
    Gaze — Intra: 0.6370 ± 0.1604  Inter: 0.6153 ± 0.1630  Sep: 0.97x

  HARD / Alert (n=20 subjects):
    ADF  — Intra: 0.2299 ± 0.1026  Inter: 0.2173 ± 0.0995  Sep: 0.94x
    Gaze — Intra: 0.6290 ± 0.1431  Inter: 0.7043 ± 0.1595  Sep: 1.12x

  HARD / Fatigue (n=20 subjects):
    ADF  — Intra: 0.1981 ± 0.0994  Inter: 0.1921 ± 0.0981  Sep: 0.96x
    Gaze — Intra: 0.6357 ± 0.1569  Inter: 0.5281 ± 0.1114  Sep: 0.84x

======================================================================
  Paired Statistical Tests (Intra vs Inter MMD)
======================================================================

  EASY / Alert:
    ADF  paired t-test: t=-0.932, p=3.6300e-01  ns
    ADF  Wilcoxon:      p=3.6828e-01
    Gaze paired t-test: t=0.961, p=3.4860e-01  ns
    Gaze Wilcoxon:      p=3.1179e-01

  EASY / Fatigue:
    ADF  paired t-test: t=-1.343, p=1.9502e-01  ns
    ADF  Wilcoxon:      p=3.1179e-01
    Gaze paired t-test: t=1.186, p=2.5040e-01  ns
    Gaze Wilcoxon:      p=2.1617e-01

  HARD / Alert:
    ADF  paired t-test: t=2.802, p=1.1378e-02  *
    ADF  Wilcoxon:      p=4.8441e-02
    Gaze paired t-test: t=-12.969, p=6.9061e-11  ***
    Gaze Wilcoxon:      p=1.9073e-06

  HARD / Fatigue:
    ADF  paired t-test: t=1.577, p=1.3131e-01  ns
    ADF  Wilcoxon:      p=3.4881e-01
    Gaze paired t-test: t=9.559, p=1.0822e-08  ***
    Gaze Wilcoxon:      p=1.9073e-06

"""