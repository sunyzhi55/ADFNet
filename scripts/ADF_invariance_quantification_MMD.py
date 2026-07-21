#!/usr/bin/env python
"""
ADF Invariance Quantification Experiment
=========================================
Validates that ADF features exhibit greater cross-subject distributional
invariance compared to raw gaze features.

Measures:
  - MMD (Maximum Mean Discrepancy, RBF kernel) per subject vs group baseline
  - KL divergence (histogram-based) per subject vs group baseline

Visualisations:
  - t-SNE scatter: raw gaze vs ADF (side-by-side)
  - Violin plot: per-subject MMD for both feature types
  - Distribution density plot: per-subject drift overlay

Usage:
  python scripts/invariance_quantification.py --data-root <path>
  python scripts/invariance_quantification.py  # uses config default
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
from scipy.spatial.distance import pdist, squareform
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
# Feature Extraction
# ===================================================================

def extract_raw_gaze_records(records: list[dict]) -> np.ndarray:
    """Per-frame raw gaze vector: [gaze_x, gaze_y, pitch, yaw].

    Falls back to (gaze_x, gaze_y, 0, 0) when pitch_yaw_rad is absent.
    Coordinates are NOT normalised — we want to preserve inter-subject
    variance for the comparison.
    """
    frames: list[np.ndarray] = []
    for rec in records:
        # --- gaze screen position (reuse robust parse_points) ---
        pts = parse_points(rec.get("gaze_screen_tf_calibrate_xy_px"))
        if not pts:
            pts = parse_points(rec.get("gaze_screen_xy_px"))
        if not pts:
            continue
        gx, gy = pts[0]

        # --- pitch / yaw ---
        pw = rec.get("pitch_yaw_rad")
        if pw is not None:
            try:
                if isinstance(pw, str):
                    pw = ast.literal_eval(pw)
                pitch, yaw = float(pw[0]), float(pw[1])
            except Exception:
                pitch, yaw = 0.0, 0.0
        else:
            pitch, yaw = 0.0, 0.0


        frames.append(np.array([gx, gy, pitch, yaw], dtype=np.float32))

    if not frames:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(frames, axis=0)


def extract_eye_records(records: list[dict]) -> np.ndarray:
    """Extract flattened facialial_landmark_35 as eye features.

    Returns a 2D array of shape (N, D) where D is fixed across all frames.
    Missing landmarks are filled with zeros.
    """
    # First pass: determine fixed dimension D from first valid record
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
            # Pad or truncate to fixed dimension D
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
    """Return (adf_feat, gaze_feat, raw_drift) at window level.

    adf_feat  : [n_windows, 6]  — mean/std of drift, diff, sliding_mean (per-sample norm)
    gaze_feat : [n_windows, 8]  — mean/std of (gx, gy, pitch, yaw)
    raw_drift : [n_windows, 1]  — mean of unnormalized drift (for KL comparison)
    """
    adf = compute_adf_features(seq_records, task_type, local_mean_size, per_sample_norm=True)
    # Raw drift: same gaze-target distance but WITHOUT per-sample normalization
    adf_raw = compute_adf_features(seq_records, task_type, local_mean_size, per_sample_norm=False)
    # raw_gaze = extract_raw_gaze_records(seq_records)
    raw_feature = extract_eye_records(seq_records)

    T = len(adf)
    if len(raw_feature) < T:
        pad = np.zeros((T - len(raw_feature), raw_feature.shape[1]), dtype=np.float32)
        raw_feature = np.concatenate([raw_feature, pad], axis=0)
    elif len(raw_feature) > T:
        raw_feature = raw_feature[:T]

    adf_windows, raw_feature_windows, drift_windows = [], [], []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        # ADF: mean & std of drift (ch0), diff (ch1), sliding_mean (ch2)
        adf_win = adf[start:end]  # [W, 3]
        adf_windows.append(np.array([
            adf_win[:, 0].mean(), adf_win[:, 0].std(),
            # adf_win[:, 1].mean(), adf_win[:, 1].std(),
            # adf_win[:, 2].mean(), adf_win[:, 2].std(),
        ], dtype=np.float32))
        # Raw feature: mean & std of eye landmarks (D-dim)
        raw_feature_win = raw_feature[start:end]  # [W, D]
        raw_feature_windows.append(np.concatenate([raw_feature_win.mean(axis=0), raw_feature_win.std(axis=0)]))
        # Raw drift: mean of unnormalized drift ch0 (1D for KL comparison)
        raw_drift_win = adf_raw[start:end, 0]  # [W]
        drift_windows.append(np.array([raw_drift_win.mean()], dtype=np.float32))

    D = raw_feature.shape[1] if len(raw_feature) > 0 else 0
    feat_dim = D * 2  # mean + std
    if not adf_windows:
        return (np.zeros((0, 6), dtype=np.float32),
                np.zeros((0, feat_dim), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32))
    return np.stack(adf_windows), np.stack(raw_feature_windows), np.stack(drift_windows)


# ===================================================================
# MMD (RBF kernel, median heuristic)
# ===================================================================

def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    return np.exp(-dists / (2.0 * sigma ** 2))


def compute_mmd(P: np.ndarray, Q: np.ndarray) -> float:
    """Biased MMD² estimate with RBF kernel (median heuristic bandwidth)."""
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
# KL Divergence (histogram-based, 1-D drift only)
# ===================================================================

def compute_kl_1d(p_vals: np.ndarray, q_vals: np.ndarray, n_bins: int = 80) -> float:
    """KL(P || Q) using shared histogram bins.  Q is the reference (group baseline)."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    p_hist, _ = np.histogram(p_vals, bins=n_bins, range=(lo, hi), density=True)
    q_hist, _ = np.histogram(q_vals, bins=n_bins, range=(lo, hi), density=True)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


# ===================================================================
# Intra-class / Inter-class Distance & Fisher Separability
# ===================================================================

def compute_intra_class_dist(adf_by_subj, gaze_by_subj, subjects):
    """Average pairwise MMD between all subject pairs within one condition.

    Lower value → features are more compact / cross-subject invariant.
    Returns (adf_intra, gaze_intra, n_pairs).
    """
    from itertools import combinations

    pairs = list(combinations(subjects, 2))
    if not pairs:
        return 0.0, 0.0, 0

    adf_mmds, gaze_mmds = [], []
    for s_i, s_j in pairs:
        adf_mmds.append(compute_mmd(adf_by_subj[s_i], adf_by_subj[s_j]))
        gaze_mmds.append(compute_mmd(gaze_by_subj[s_i], gaze_by_subj[s_j]))

    return float(np.mean(adf_mmds)), float(np.mean(gaze_mmds)), len(pairs)


def compute_inter_class_dist(adf_alert, gaze_alert, adf_fatigue, gaze_fatigue):
    """MMD between pooled alert vs pooled fatigue features.

    Higher value → alert and fatigue are more separable.
    Returns (adf_inter, gaze_inter).
    """
    adf_inter = compute_mmd(adf_alert, adf_fatigue)
    gaze_inter = compute_mmd(gaze_alert, gaze_fatigue)
    return adf_inter, gaze_inter


def build_separability_table(task_name, alert_data, fatigue_data):
    """Build a summary dict for one task (easy or hard).

    alert_data / fatigue_data: dict[subject -> np.ndarray] for ADF and gaze.
    Returns dict with all separability metrics.
    """
    adf_alert_subj, gaze_alert_subj = alert_data["adf"], alert_data["gaze"]
    adf_fatigue_subj, gaze_fatigue_subj = fatigue_data["adf"], fatigue_data["gaze"]

    alert_subjects = sorted(adf_alert_subj.keys())
    fatigue_subjects = sorted(adf_fatigue_subj.keys())

    # --- Intra-class (within alert, within fatigue) ---
    adf_intra_alert, gaze_intra_alert, _ = compute_intra_class_dist(
        adf_alert_subj, gaze_alert_subj, alert_subjects)
    adf_intra_fatigue, gaze_intra_fatigue, _ = compute_intra_class_dist(
        adf_fatigue_subj, gaze_fatigue_subj, fatigue_subjects)

    # Average intra across alert & fatigue
    adf_intra_avg = (adf_intra_alert + adf_intra_fatigue) / 2.0
    gaze_intra_avg = (gaze_intra_alert + gaze_intra_fatigue) / 2.0

    # --- Inter-class (alert vs fatigue pooled) ---
    adf_alert_pooled = np.concatenate([adf_alert_subj[s] for s in alert_subjects])
    adf_fatigue_pooled = np.concatenate([adf_fatigue_subj[s] for s in fatigue_subjects])
    gaze_alert_pooled = np.concatenate([gaze_alert_subj[s] for s in alert_subjects])
    gaze_fatigue_pooled = np.concatenate([gaze_fatigue_subj[s] for s in fatigue_subjects])

    adf_inter, gaze_inter = compute_inter_class_dist(
        adf_alert_pooled, gaze_alert_pooled, adf_fatigue_pooled, gaze_fatigue_pooled)

    # --- Fisher Ratio = Inter / Intra ---
    adf_ratio = adf_inter / (adf_intra_avg + 1e-10)
    gaze_ratio = gaze_inter / (gaze_intra_avg + 1e-10)

    return {
        "Task": task_name,
        "ADF_Intra_Alert": adf_intra_alert,
        "ADF_Intra_Fatigue": adf_intra_fatigue,
        "ADF_Intra_Avg": adf_intra_avg,
        "Gaze_Intra_Alert": gaze_intra_alert,
        "Gaze_Intra_Fatigue": gaze_intra_fatigue,
        "Gaze_Intra_Avg": gaze_intra_avg,
        "ADF_Inter": adf_inter,
        "Gaze_Inter": gaze_inter,
        "ADF_Ratio": adf_ratio,
        "Gaze_Ratio": gaze_ratio,
        "Ratio_Advantage": adf_ratio / (gaze_ratio + 1e-10),
    }


# ===================================================================
# Visualisation
# ===================================================================

def plot_tsne(adf_by_subj, gaze_by_subj, subjects, output_dir):
    """t-SNE scatter: raw gaze (left) vs ADF (right)."""
    from sklearn.manifold import TSNE

    all_adf = np.concatenate([adf_by_subj[s] for s in subjects])
    all_gaze = np.concatenate([gaze_by_subj[s] for s in subjects])
    labels = np.concatenate([np.full(len(adf_by_subj[s]), s) for s in subjects])

    n_total = len(all_adf)
    perplexity = min(30, max(5, n_total // 10))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    adf_2d = tsne.fit_transform(all_adf)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    gaze_2d = tsne.fit_transform(all_gaze)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_subjects = sorted(set(subjects))
    cmap = plt.cm.get_cmap("tab20", len(unique_subjects))
    color_map = {s: cmap(i) for i, s in enumerate(unique_subjects)}
    colors = [color_map[s] for s in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for s in unique_subjects:
        mask = labels == s
        ax1.scatter(gaze_2d[mask, 0], gaze_2d[mask, 1], c=[color_map[s]], label=s, s=6, alpha=0.45)
        ax2.scatter(adf_2d[mask, 0], adf_2d[mask, 1], c=[color_map[s]], label=s, s=6, alpha=0.45)

    ax1.set_title("Raw Gaze Features (t-SNE)", fontsize=14)
    ax2.set_title("ADF Features (t-SNE)", fontsize=14)
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=5, loc="upper right", ncol=2, markerscale=3, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "tsne_comparison.png", dpi=200)
    plt.close()
    print(f"  [saved] tsne_comparison.png")


def plot_violin(results_df, output_dir):
    """Violin plot of per-subject MMD distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(
        [results_df["ADF_MMD"].values, results_df["RawGaze_MMD"].values],
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["ADF", "Raw Gaze"])
    ax.set_ylabel("MMD (subject vs group baseline)")
    ax.set_title("Cross-Subject Distributional Invariance\n(lower MMD = more invariant)")
    plt.tight_layout()
    plt.savefig(output_dir / "mmd_violin.png", dpi=200)
    plt.close()
    print(f"  [saved] mmd_violin.png")


def plot_density_overlay(adf_by_subj, subjects, output_dir):
    """Per-subject drift density overlay — shows ADF distributions overlap tightly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.get_cmap("tab20", len(subjects))
    for i, s in enumerate(sorted(subjects)):
        drift_means = adf_by_subj[s][:, 0]  # mean drift per window
        ax1.hist(drift_means, bins=40, alpha=0.35, density=True, color=cmap(i), label=s)

    # Group baseline
    all_drift = np.concatenate([adf_by_subj[s][:, 0] for s in subjects])
    ax1.hist(all_drift, bins=60, alpha=0.15, density=True, color="black", label="Group", linewidth=2, histtype="step")

    ax1.set_title("ADF Drift Distribution per Subject (Alert State)")
    ax1.set_xlabel("Mean Drift (normalised)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=5, ncol=2, loc="upper right", framealpha=0.7)

    # Bar chart: MMD per subject
    # (will be filled from results_df later)
    ax2.axis("off")
    ax2.text(0.5, 0.5, "(see mmd_violin.png)", ha="center", va="center", fontsize=12, color="gray")

    plt.tight_layout()
    plt.savefig(output_dir / "density_overlay.png", dpi=200)
    plt.close()
    print(f"  [saved] density_overlay.png")


def plot_bar_comparison(results_df, output_dir):
    """Grouped bar chart: ADF MMD vs Raw Gaze MMD per subject."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, len(results_df) * 0.5), 5))
    x = np.arange(len(results_df))
    w = 0.35
    ax.bar(x - w / 2, results_df["ADF_MMD"], w, label="ADF", color="steelblue")
    ax.bar(x + w / 2, results_df["RawGaze_MMD"], w, label="Raw Gaze", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Subject"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MMD")
    ax.set_title("Per-Subject MMD: ADF vs Raw Gaze\n(lower = more cross-subject invariant)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mmd_bar_comparison.png", dpi=200)
    plt.close()
    print(f"  [saved] mmd_bar_comparison.png")


# ===================================================================
# Main
# ===================================================================

def extract_features_for_sequences(sequences, window_size, stride, local_mean_size):
    """Extract ADF, raw gaze, and raw drift window-level features."""
    all_adf, all_gaze, all_drift = [], [], []
    for seq in sequences:
        records = list(iter_jsonl(seq.path))
        if len(records) < window_size:
            continue
        adf_feat, gaze_feat, raw_drift = window_level_features(
            records, seq.task_type, window_size, stride, local_mean_size,
        )
        if len(adf_feat) > 0:
            all_adf.append(adf_feat)
            all_gaze.append(gaze_feat)
            all_drift.append(raw_drift)
    if not all_adf:
        return None, None, None
    return np.concatenate(all_adf), np.concatenate(all_gaze), np.concatenate(all_drift)


def compute_and_report(adf_by_subj, gaze_by_subj, drift_by_subj, valid_subjects, condition_name):
    """Compute MMD/KL for all subjects in one condition, return DataFrame."""
    print(f"\n  --- {condition_name} ---")
    rows = []
    for subj in valid_subjects:
        others_adf = np.concatenate([adf_by_subj[s] for s in valid_subjects if s != subj])
        others_gaze = np.concatenate([gaze_by_subj[s] for s in valid_subjects if s != subj])

        adf_mmd = compute_mmd(adf_by_subj[subj], others_adf)
        gaze_mmd = compute_mmd(gaze_by_subj[subj], others_gaze)

        # KL divergence: ADF drift (normalised) vs Raw drift (unnormalised)
        subj_adf_drift = adf_by_subj[subj][:, 0]
        others_adf_drift = others_adf[:, 0]
        adf_kl = compute_kl_1d(subj_adf_drift, others_adf_drift)

        subj_raw_drift = drift_by_subj[subj][:, 0]
        others_raw_drift = np.concatenate([drift_by_subj[s] for s in valid_subjects if s != subj])[:, 0]
        gaze_kl = compute_kl_1d(subj_raw_drift, others_raw_drift)

        rows.append({
            "Subject": subj,
            "N_Windows": len(adf_by_subj[subj]),
            "ADF_MMD": adf_mmd,
            "RawGaze_MMD": gaze_mmd,
            "ADF_KL": adf_kl,
            "Gaze_KL": gaze_kl,
            "MMD_Ratio": gaze_mmd / (adf_mmd + 1e-10),
            "KL_Ratio": gaze_kl / (adf_kl + 1e-10),
        })
        print(f"  {subj:>6s}  N={len(adf_by_subj[subj]):>4d}  "
              f"ADF_MMD={adf_mmd:.4f}  Gaze_MMD={gaze_mmd:.4f}  "
              f"Ratio={gaze_mmd / (adf_mmd + 1e-10):.2f}x  "
              f"ADF_KL={adf_kl:.4f}  Gaze_KL={gaze_kl:.4f}")

    results_df = pd.DataFrame(rows)

    # ---- summary statistics ----
    print()
    mean_adf = results_df["ADF_MMD"].mean()
    mean_gaze = results_df["RawGaze_MMD"].mean()
    mean_adf_kl = results_df["ADF_KL"].mean()
    mean_gaze_kl = results_df["Gaze_KL"].mean()
    mean_ratio = results_df["MMD_Ratio"].mean()
    mean_kl_ratio = results_df["KL_Ratio"].mean()
    print(f"  Mean ADF MMD:      {mean_adf:.4f} ± {results_df['ADF_MMD'].std():.4f}")
    print(f"  Mean RawGaze MMD:  {mean_gaze:.4f} ± {results_df['RawGaze_MMD'].std():.4f}")
    print(f"  Mean MMD Ratio (Gaze/ADF): {mean_ratio:.2f}x")
    print(f"  Mean ADF KL:       {mean_adf_kl:.4f} ± {results_df['ADF_KL'].std():.4f}")
    print(f"  Mean Gaze KL:      {mean_gaze_kl:.4f} ± {results_df['Gaze_KL'].std():.4f}")
    print(f"  Mean KL Ratio (Gaze/ADF):  {mean_kl_ratio:.2f}x")
    reduction = (1.0 - mean_adf / (mean_gaze + 1e-10)) * 100
    kl_reduction = (1.0 - mean_adf_kl / (mean_gaze_kl + 1e-10)) * 100
    print(f"  >> ADF MMD reduced by {reduction:.1f}% compared to raw gaze")
    print(f"  >> ADF KL  reduced by {kl_reduction:.1f}% compared to raw gaze")

    # ---- paired statistical tests (MMD) ----
    adf_vals = results_df["ADF_MMD"].values
    gaze_vals = results_df["RawGaze_MMD"].values
    t_stat, t_pval = ttest_rel(adf_vals, gaze_vals)
    try:
        w_stat, w_pval = wilcoxon(adf_vals, gaze_vals)
    except Exception:
        w_pval = float("nan")
    print(f"  MMD Paired t-test:  t={t_stat:.3f}, p={t_pval:.4e}")
    print(f"  MMD Wilcoxon test:  p={w_pval:.4e}")
    print(f"  MMD Significance (p<0.05): {'YES' if t_pval < 0.05 else 'NO'}")

    # ---- paired statistical tests (KL) ----
    adf_kl_vals = results_df["ADF_KL"].values
    gaze_kl_vals = results_df["Gaze_KL"].values
    t_kl, p_kl = ttest_rel(adf_kl_vals, gaze_kl_vals)
    try:
        w_kl, pw_kl = wilcoxon(adf_kl_vals, gaze_kl_vals)
    except Exception:
        pw_kl = float("nan")
    print(f"  KL  Paired t-test:  t={t_kl:.3f}, p={p_kl:.4e}")
    print(f"  KL  Wilcoxon test:  p={pw_kl:.4e}")
    print(f"  KL  Significance (p<0.05): {'YES' if p_kl < 0.05 else 'NO'}")

    return results_df


def save_condition_csv(results_df, condition_name, output_dir):
    """Save per-condition CSV with MEAN/STD summary rows."""
    summary_rows = [
        {"Subject": "MEAN", "N_Windows": int(results_df["N_Windows"].sum()),
         "ADF_MMD": results_df["ADF_MMD"].mean(),
         "RawGaze_MMD": results_df["RawGaze_MMD"].mean(),
         "ADF_KL": results_df["ADF_KL"].mean(),
         "Gaze_KL": results_df["Gaze_KL"].mean(),
         "MMD_Ratio": results_df["MMD_Ratio"].mean(),
         "KL_Ratio": results_df["KL_Ratio"].mean()},
        {"Subject": "STD", "N_Windows": 0,
         "ADF_MMD": results_df["ADF_MMD"].std(),
         "RawGaze_MMD": results_df["RawGaze_MMD"].std(),
         "ADF_KL": results_df["ADF_KL"].std(),
         "Gaze_KL": results_df["Gaze_KL"].std(),
         "MMD_Ratio": results_df["MMD_Ratio"].std(),
         "KL_Ratio": results_df["KL_Ratio"].std()},
    ]
    out_df = pd.concat([results_df, pd.DataFrame(summary_rows)], ignore_index=True)
    fname = f"invariance_{condition_name}.csv"
    out_df.to_csv(output_dir / fname, index=False, float_format="%.6f")
    print(f"  [saved] {fname}")


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="ADF Invariance Quantification")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Data directory (default: from configs/default.yaml)")
    parser.add_argument("--task-mode", type=str, default="all",
                        choices=["easy", "hard", "all"])
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--local-mean-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="invariance_results")
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

    # ---- discover & filter ----
    print(f"Data root: {data_root}")
    all_seqs = discover_sequences(data_root)
    all_seqs = filter_sequences_by_task(all_seqs, args.task_mode)

    subjects = sorted(set(s.subject_id for s in all_seqs))
    print(f"Subjects: {len(subjects)}")
    print(f"Total sequences: {len(all_seqs)}")

    # ---- group by 4 conditions: (task_type, label_name) ----
    CONDITIONS = [
        ("easy", "alert"),
        ("easy", "sleepy"),   # "sleepy" covers both sleep & sleepy in LABEL_MAP
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
        # also accept "sleep" as "sleepy" for the sleepy condition
        if label_name == "sleepy":
            cond_seqs_extra = [s for s in all_seqs
                               if s.task_type == task and s.label_name == "sleep"]
            cond_seqs = cond_seqs + cond_seqs_extra
        by_subj = defaultdict(list)
        for s in cond_seqs:
            by_subj[s.subject_id].append(s)
        seqs_by_cond[(task, label_name)] = by_subj
        n_seqs = sum(len(v) for v in by_subj.values())
        print(f"  {COND_LABELS[(task, label_name)]:>14s}: {len(by_subj)} subjects, {n_seqs} sequences")

    # ---- process each condition ----
    print("\n" + "=" * 70)
    print("  Cross-Subject Invariance Quantification (4 conditions)")
    print("=" * 70)

    all_condition_results = {}  # cond_label -> DataFrame
    features_by_cond = {}       # (task, label_name) -> {"adf": {subj: arr}, "gaze": {subj: arr}}

    for (task, label_name) in CONDITIONS:
        cond_label = COND_LABELS[(task, label_name)]
        by_subj = seqs_by_cond[(task, label_name)]

        if not by_subj:
            print(f"\n  [skip] {cond_label}: no data")
            continue

        print(f"\n{'#' * 70}")
        print(f"# Condition: {cond_label}  ({len(by_subj)} subjects)")
        print(f"{'#' * 70}")

        # ---- extract features per subject ----
        adf_by_subj = {}
        gaze_by_subj = {}
        drift_by_subj = {}
        for subj in sorted(by_subj.keys()):
            adf_feat, gaze_feat, raw_drift = extract_features_for_sequences(
                by_subj[subj], args.window_size, args.stride, args.local_mean_size,
            )
            if adf_feat is not None:
                adf_by_subj[subj] = adf_feat
                gaze_by_subj[subj] = gaze_feat
                drift_by_subj[subj] = raw_drift
                print(f"  Subject {subj}: {len(adf_feat)} windows")

        valid_subjects = sorted(adf_by_subj.keys())
        if len(valid_subjects) < 2:
            print(f"  [skip] {cond_label}: only {len(valid_subjects)} subject(s) with data")
            continue

        print(f"\n  Subjects with valid features: {len(valid_subjects)}")

        # ---- store features for later separability analysis ----
        features_by_cond[(task, label_name)] = {
            "adf": {s: adf_by_subj[s] for s in valid_subjects},
            "gaze": {s: gaze_by_subj[s] for s in valid_subjects},
        }

        # ---- intra-class distance for this condition ----
        adf_intra, gaze_intra, n_pairs = compute_intra_class_dist(
            adf_by_subj, gaze_by_subj, valid_subjects)
        print(f"  Intra-class Dist (pairwise MMD, {n_pairs} pairs):")
        print(f"    ADF:      {adf_intra:.4f}")
        print(f"    Raw Gaze: {gaze_intra:.4f}")
        if gaze_intra > adf_intra:
            print(f"    >> ADF is {gaze_intra / (adf_intra + 1e-10):.2f}x more compact than raw gaze")
        else:
            print(f"    >> Raw Gaze is {adf_intra / (gaze_intra + 1e-10):.2f}x more compact than ADF")

        # ---- compute MMD / KL ----
        results_df = compute_and_report(adf_by_subj, gaze_by_subj, drift_by_subj, valid_subjects, cond_label)
        all_condition_results[cond_label] = results_df

        # ---- save CSV ----
        save_condition_csv(results_df, cond_label, output_dir)

        # ---- plots ----
        if not args.no_plot:
            cond_dir = output_dir / cond_label
            cond_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  Generating plots for {cond_label}...")
            try:
                plot_tsne(adf_by_subj, gaze_by_subj, valid_subjects, cond_dir)
            except Exception as e:
                print(f"    [skip] t-SNE: {e}")
            try:
                plot_violin(results_df, cond_dir)
            except Exception as e:
                print(f"    [skip] Violin: {e}")
            try:
                plot_bar_comparison(results_df, cond_dir)
            except Exception as e:
                print(f"    [skip] Bar: {e}")
            try:
                plot_density_overlay(adf_by_subj, valid_subjects, cond_dir)
            except Exception as e:
                print(f"    [skip] Density: {e}")

    # ---- combined summary CSV ----
    if all_condition_results:
        print(f"\n{'=' * 70}")
        print("  Combined Summary")
        print(f"{'=' * 70}")
        summary_rows = []
        for cond_label, df in all_condition_results.items():
            mmd_red = (1.0 - df["ADF_MMD"].mean() / (df["RawGaze_MMD"].mean() + 1e-10)) * 100
            kl_red = (1.0 - df["ADF_KL"].mean() / (df["Gaze_KL"].mean() + 1e-10)) * 100
            summary_rows.append({
                "Condition": cond_label,
                "N_Subjects": len(df),
                "ADF_MMD_Mean": df["ADF_MMD"].mean(),
                "ADF_MMD_Std": df["ADF_MMD"].std(),
                "RawGaze_MMD_Mean": df["RawGaze_MMD"].mean(),
                "RawGaze_MMD_Std": df["RawGaze_MMD"].std(),
                "ADF_KL_Mean": df["ADF_KL"].mean(),
                "ADF_KL_Std": df["ADF_KL"].std(),
                "Gaze_KL_Mean": df["Gaze_KL"].mean(),
                "Gaze_KL_Std": df["Gaze_KL"].std(),
                "MMD_Ratio_Mean": df["MMD_Ratio"].mean(),
                "KL_Ratio_Mean": df["KL_Ratio"].mean(),
                "MMD_Reduction_Pct": mmd_red,
                "KL_Reduction_Pct": kl_red,
            })
        summary_df = pd.DataFrame(summary_rows)
        for _, row in summary_df.iterrows():
            print(f"  {row['Condition']:>14s}  "
                  f"ADF_MMD={row['ADF_MMD_Mean']:.4f}±{row['ADF_MMD_Std']:.4f}  "
                  f"Gaze_MMD={row['RawGaze_MMD_Mean']:.4f}±{row['RawGaze_MMD_Std']:.4f}  "
                  f"MMD_red={row['MMD_Reduction_Pct']:.1f}%  "
                  f"ADF_KL={row['ADF_KL_Mean']:.4f}  "
                  f"Gaze_KL={row['Gaze_KL_Mean']:.4f}  "
                  f"KL_red={row['KL_Reduction_Pct']:.1f}%")

        summary_df.to_csv(output_dir / "invariance_summary.csv", index=False, float_format="%.6f")
        print(f"\n  [saved] invariance_summary.csv")

    # ================================================================
    # Separability Analysis: Intra-class / Inter-class / Fisher Ratio
    # ================================================================
    separability_rows = []
    for task_name in ["easy", "hard"]:
        alert_key = (task_name, "alert")
        fatigue_key = (task_name, "sleepy")

        if alert_key not in features_by_cond or fatigue_key not in features_by_cond:
            print(f"\n  [skip] separability for {task_name}: missing alert or fatigue data")
            continue

        alert_data = features_by_cond[alert_key]
        fatigue_data = features_by_cond[fatigue_key]

        # need at least 2 subjects in each
        if len(alert_data["adf"]) < 2 or len(fatigue_data["adf"]) < 2:
            print(f"\n  [skip] separability for {task_name}: insufficient subjects")
            continue

        print(f"\n{'=' * 70}")
        print(f"  Separability Analysis: {task_name.upper()} task")
        print(f"{'=' * 70}")

        row = build_separability_table(task_name, alert_data, fatigue_data)
        separability_rows.append(row)

        print(f"  Intra-class Distance (avg pairwise MMD):")
        print(f"    Alert   — ADF: {row['ADF_Intra_Alert']:.4f}  |  Gaze: {row['Gaze_Intra_Alert']:.4f}")
        print(f"    Fatigue — ADF: {row['ADF_Intra_Fatigue']:.4f}  |  Gaze: {row['Gaze_Intra_Fatigue']:.4f}")
        print(f"    Average — ADF: {row['ADF_Intra_Avg']:.4f}  |  Gaze: {row['Gaze_Intra_Avg']:.4f}")
        print(f"  Inter-class Distance (alert vs fatigue MMD):")
        print(f"    ADF: {row['ADF_Inter']:.4f}  |  Gaze: {row['Gaze_Inter']:.4f}")
        print(f"  Fisher Ratio (Inter / Intra):")
        print(f"    ADF: {row['ADF_Ratio']:.4f}  |  Gaze: {row['Gaze_Ratio']:.4f}")
        print(f"  >> ADF Fisher Ratio is {row['Ratio_Advantage']:.2f}x that of Raw Gaze")

    if separability_rows:
        sep_df = pd.DataFrame(separability_rows)
        sep_df.to_csv(output_dir / "invariance_separability.csv", index=False, float_format="%.6f")
        print(f"\n  [saved] invariance_separability.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
