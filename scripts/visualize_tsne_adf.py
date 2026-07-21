#!/usr/bin/env python
"""
ADF Feature t-SNE Visualisation
================================
Plots t-SNE of raw 256-dimensional ADF drift windows, coloured by
alert / fatigue label.  Produces three figures:
  1. Easy task only
  2. Hard task only
  3. Easy + Hard mixed (with task marker distinction)

Usage:
  python scripts/visualize_tsne_adf.py --data-root <path>
  python scripts/visualize_tsne_adf.py   # uses config default
"""

from __future__ import annotations

import sys
import io
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from data.io import discover_sequences, filter_sequences_by_task, iter_jsonl
from data.features import compute_adf_features


# ===================================================================
# Data extraction
# ===================================================================

def extract_drift_windows(
    records: list[dict],
    task_type: str,
    window_size: int = 256,
    stride: int = 128,
    local_mean_size: int = 16,
) -> np.ndarray:
    """Return per-window drift vectors of shape [n_windows, window_size].

    Uses per-sample Min-Max normalised drift (ch 0 of ADF features).
    """
    adf = compute_adf_features(records, task_type, local_mean_size, per_sample_norm=True)
    T = len(adf)
    windows = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(adf[start:start + window_size, 0])  # drift channel only
    if not windows:
        return np.zeros((0, window_size), dtype=np.float32)
    return np.stack(windows, axis=0)  # [n_windows, window_size]


# ===================================================================
# Plotting
# ===================================================================

def plot_tsne_2class(embed_2d, labels, title, output_path,
                     task_types=None):
    """Scatter plot of 2-D embedding coloured by alert/fatigue.

    If task_types is provided, easy=circle, hard=triangle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    alert_mask = labels == 0
    fatigue_mask = labels == 1

    if task_types is not None:
        easy_mask = task_types == "easy"
        hard_mask = task_types == "hard"

        # alert + easy: circle
        m = alert_mask & easy_mask
        if m.any():
            ax.scatter(embed_2d[m, 0], embed_2d[m, 1],
                       c="#2196F3", marker="o", s=12, alpha=0.55,
                       label="Alert (Easy)")
        # alert + hard: triangle
        m = alert_mask & hard_mask
        if m.any():
            ax.scatter(embed_2d[m, 0], embed_2d[m, 1],
                       c="#2196F3", marker="^", s=14, alpha=0.55,
                       label="Alert (Hard)")
        # fatigue + easy: circle
        m = fatigue_mask & easy_mask
        if m.any():
            ax.scatter(embed_2d[m, 0], embed_2d[m, 1],
                       c="#F44336", marker="o", s=12, alpha=0.55,
                       label="Fatigue (Easy)")
        # fatigue + hard: triangle
        m = fatigue_mask & hard_mask
        if m.any():
            ax.scatter(embed_2d[m, 0], embed_2d[m, 1],
                       c="#F44336", marker="^", s=14, alpha=0.55,
                       label="Fatigue (Hard)")
    else:
        ax.scatter(embed_2d[alert_mask, 0], embed_2d[alert_mask, 1],
                   c="#2196F3", s=12, alpha=0.6, label=f"Alert ({alert_mask.sum()})")
        ax.scatter(embed_2d[fatigue_mask, 0], embed_2d[fatigue_mask, 1],
                   c="#F44336", s=12, alpha=0.6, label=f"Fatigue ({fatigue_mask.sum()})")

    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  [saved] {output_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="ADF t-SNE Visualisation")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/shenxy/Data/Process0620_calibrate")
    parser.add_argument("--task-mode", type=str, default="all",
                        choices=["easy", "hard", "all"])
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--local-mean-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="tsne_results")
    parser.add_argument("--perplexity", type=int, default=30)
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
    all_seqs = filter_sequences_by_task(all_seqs, args.task_mode)
    print(f"Total sequences: {len(all_seqs)}")

    # ---- extract drift windows per sequence ----
    all_drifts = []   # list of [n_windows_i, 256]
    all_labels = []   # list of [n_windows_i]
    all_tasks = []    # list of [n_windows_i]  ("easy" / "hard")

    for seq in all_seqs:
        records = list(iter_jsonl(seq.path))
        if len(records) < args.window_size:
            continue
        drift_wins = extract_drift_windows(
            records, seq.task_type, args.window_size, args.stride, args.local_mean_size,
        )
        if len(drift_wins) == 0:
            continue
        n = len(drift_wins)
        all_drifts.append(drift_wins)
        all_labels.append(np.full(n, seq.label, dtype=np.int32))
        all_tasks.append(np.array([seq.task_type] * n))

    if not all_drifts:
        print("ERROR: No valid windows extracted.")
        return

    X_all = np.concatenate(all_drifts, axis=0)    # [N, 256]
    y_all = np.concatenate(all_labels, axis=0)     # [N]
    t_all = np.concatenate(all_tasks, axis=0)       # [N]

    print(f"Total windows: {len(X_all)}")
    print(f"  Alert:   {(y_all == 0).sum()}")
    print(f"  Fatigue: {(y_all == 1).sum()}")
    print(f"  Easy:    {(t_all == 'easy').sum()}")
    print(f"  Hard:    {(t_all == 'hard').sum()}")

    # ---- t-SNE ----
    from sklearn.manifold import TSNE

    perp = min(args.perplexity, max(5, len(X_all) // 10))

    # We can compute t-SNE once on all data, then slice for sub-plots.
    print(f"\nRunning t-SNE (perplexity={perp}) on {len(X_all)} samples × {X_all.shape[1]}D ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
                learning_rate="auto", init="pca")
    embed_all = tsne.fit_transform(X_all)
    print("  t-SNE done.")

    # ---- 1) Easy only ----
    easy_mask = t_all == "easy"
    if easy_mask.sum() > 10:
        plot_tsne_2class(
            embed_all[easy_mask],
            y_all[easy_mask],
            f"ADF Drift t-SNE — Easy Task (n={easy_mask.sum()})",
            output_dir / "tsne_easy.png",
        )

    # ---- 2) Hard only ----
    hard_mask = t_all == "hard"
    if hard_mask.sum() > 10:
        plot_tsne_2class(
            embed_all[hard_mask],
            y_all[hard_mask],
            f"ADF Drift t-SNE — Hard Task (n={hard_mask.sum()})",
            output_dir / "tsne_hard.png",
        )

    # ---- 3) Mixed (easy + hard) ----
    plot_tsne_2class(
        embed_all,
        y_all,
        f"ADF Drift t-SNE — All Tasks (n={len(X_all)})",
        output_dir / "tsne_mixed.png",
        task_types=t_all,
    )

    # ---- 4) Bonus: single figure with 3 sub-panels ----
    print("\nGenerating combined 3-panel figure ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    panels = [
        (axes[0], easy_mask,  "Easy Task"),
        (axes[1], hard_mask,  "Hard Task"),
        (axes[2], None,       "All Tasks"),
    ]

    for ax, mask, title in panels:
        if mask is not None:
            e = embed_all[mask]
            y = y_all[mask]
        else:
            e = embed_all
            y = y_all

        alert_m = y == 0
        fatigue_m = y == 1

        ax.scatter(e[alert_m, 0], e[alert_m, 1],
                   c="#2196F3", s=10, alpha=0.55, label=f"Alert ({alert_m.sum()})")
        ax.scatter(e[fatigue_m, 0], e[fatigue_m, 1],
                   c="#F44336", s=10, alpha=0.55, label=f"Fatigue ({fatigue_m.sum()})")

        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=9, loc="best", framealpha=0.8)

    fig.suptitle("ADF Drift Features — t-SNE Embedding (256D → 2D)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_combined.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] tsne_combined.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
