#!/usr/bin/env python
"""
ADF Deviation Qualitative Visualisation (Full Suite)
=====================================================
Generates publication-quality figures demonstrating the physical
interpretability of ADF (Attention Deviation Feature).

Outputs (organised by task type):
  visualisation_results/
  ├── easy/
  │   ├── histogram/           # Per-subject deviation histograms
  │   │   ├── S01_deviation_hist.pdf
  │   │   ├── S02_deviation_hist.pdf
  │   │   └── ...
  │   └── trajectory/          # Per-subject gaze trajectory comparisons
  │       ├── S01_gaze_trajectory.pdf
  │       ├── S02_gaze_trajectory.pdf
  │       └── ...
  ├── hard/
  │   ├── histogram/
  │   └── trajectory/
  ├── average_deviation_distribution.pdf  # Combined: (A) Easy, (B) Hard
  └── paper_figures/
      ├── fig_histogram_2subjects_easy.png
      ├── fig_histogram_2subjects_hard.png
      ├── fig_trajectory_best_easy.png
      ├── fig_trajectory_best_hard.png
      └── panels/                        # Individual subplots (PNG) for PPT layout
          ├── hist_S<id>_<task>.png
          ├── traj_alert_S<id>_<task>.png
          ├── traj_fatigue_S<id>_<task>.png
          ├── avg_dist_easy.png
          └── avg_dist_hard.png

Usage:
  python scripts/visualize_adf_deviation.py --data-root <path>
  python scripts/visualize_adf_deviation.py   # uses default path
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

from data.io import discover_sequences, filter_sequences_by_task, iter_jsonl, parse_points


# ===================================================================
# Data extraction
# ===================================================================

def extract_deviation_and_gaze(
    records: list[dict],
    task_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-frame deviation, gaze positions, and target positions.

    Deviation priority: precomputed field > gaze-target geometry (matches
    compute_adf_features logic in src/data/features.py).

    Returns:
        deviation: [T] float array of gaze-target distances (px)
        gaze_xy:   [T, 2] float array of calibrated gaze screen positions
        target_xy: [T, 2] float array of target positions (nearest for hard)
    """
    deviations: list[float] = []
    gaze_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []

    for record in records:
        # --- deviation: prefer precomputed ---
        precomputed = record.get("deviation_px_before_calibrate")
        deviation = None
        if precomputed is not None:
            try:
                deviation = float(precomputed)
            except (TypeError, ValueError):
                pass

        # --- gaze point ---
        gaze_points = parse_points(record.get("gaze_screen_tf_calibrate_xy_px"))
        if not gaze_points:
            gaze_points = parse_points(record.get("gaze_screen_xy_px"))

        # --- target point ---
        if task_type == "hard":
            target_raw = record.get("target_centers_xy_px")
        else:
            target_raw = record.get("target_xy_px")
        target_points = parse_points(target_raw)

        # Compute geometry if precomputed unavailable
        gaze_xy = None
        target_xy = None
        if gaze_points and target_points:
            gaze_xy = np.array(gaze_points[0], dtype=np.float64)
            targets_arr = np.array(target_points, dtype=np.float64)
            dists = np.linalg.norm(targets_arr - gaze_xy[None, :], axis=1)
            nearest_idx = int(np.argmin(dists))
            target_xy = targets_arr[nearest_idx]
            if deviation is None:
                deviation = float(dists[nearest_idx])

        if deviation is None:
            continue

        deviations.append(deviation)
        if gaze_xy is not None:
            gaze_list.append(gaze_xy)
            target_list.append(target_xy)
        else:
            gaze_list.append(np.array([np.nan, np.nan]))
            target_list.append(np.array([np.nan, np.nan]))

    if not deviations:
        return np.array([]), np.zeros((0, 2)), np.zeros((0, 2))

    return (
        np.array(deviations, dtype=np.float64),
        np.stack(gaze_list, axis=0),
        np.stack(target_list, axis=0),
    )


# ===================================================================
# Plotting: per-subject histogram
# ===================================================================

def plot_subject_histogram(
    alert_dev: np.ndarray,
    fatigue_dev: np.ndarray,
    subject_id: str,
    task_type: str,
    output_path: Path,
    n_bins: int = 50,
):
    """Single-panel Alert vs Fatigue deviation histogram for one subject."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    all_vals = np.concatenate([alert_dev, fatigue_dev])
    upper = max(float(np.percentile(all_vals, 99)), 1.0)
    bins = np.linspace(0, upper, n_bins + 1)

    ax.hist(
        alert_dev, bins=bins, alpha=0.6, color="#2196F3",
        density=True, label=f"Alert (μ={np.mean(alert_dev):.1f}, σ={np.std(alert_dev):.1f})",
        edgecolor="white", linewidth=0.3,
    )
    ax.hist(
        fatigue_dev, bins=bins, alpha=0.6, color="#F44336",
        density=True, label=f"Fatigue (μ={np.mean(fatigue_dev):.1f}, σ={np.std(fatigue_dev):.1f})",
        edgecolor="white", linewidth=0.3,
    )

    ax.axvline(np.mean(alert_dev), color="#1565C0", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.axvline(np.mean(fatigue_dev), color="#C62828", linestyle="--", linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Gaze-Target Deviation (px)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Subject {subject_id} — {task_type.capitalize()} Task", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===================================================================
# Plotting: gaze trajectory (panel helper + combined figure)
# ===================================================================

# Offset-vector colour: green contrasts with both the blue (Alert) and
# red (Fatigue) gaze clouds, so arrows remain readable in either panel.
ARROW_COLOR = "#2E7D32"


def _draw_trajectory_panel(
    ax,
    subject_data: dict,
    state_key: str,
    gaze_color: str,
    max_points: int = 600,
    arrow_subsample: int = 8,
    arrow_color: str = ARROW_COLOR,
) -> bool:
    """Draw one gaze-trajectory panel (target cross + gaze cloud + offset arrows).

    Shows a "Target" / "Gaze Point" legend (identical for both Alert and
    Fatigue panels) but no stats text, keeping the figure clean for paper layout.
    Offset arrows are drawn with ``FancyArrowPatch(clip_on=True)`` and filtered
    so that both head and tail lie inside the axes frame.

    Returns True if the panel was drawn, False if there was no usable data.
    """
    from matplotlib.patches import FancyArrowPatch

    if state_key not in subject_data:
        return False

    gaze_xy = subject_data[state_key]["gaze_xy"]
    target_xy = subject_data[state_key]["target_xy"]

    # Filter NaN entries (precomputed-only records)
    valid_mask = ~np.isnan(gaze_xy[:, 0]) & ~np.isnan(target_xy[:, 0])
    gaze_xy = gaze_xy[valid_mask]
    target_xy = target_xy[valid_mask]
    if len(gaze_xy) == 0:
        return False

    # Subsample for visual clarity
    n_total = len(gaze_xy)
    if n_total > max_points:
        indices = np.linspace(0, n_total - 1, max_points, dtype=int)
    else:
        indices = np.arange(n_total)
    gaze_sub = gaze_xy[indices]

    # Explicit axis limits from the data (with padding) so that every arrow
    # head and tail is guaranteed to fall inside the visible frame.
    all_pts = np.concatenate([gaze_xy, target_xy], axis=0)
    xmin, xmax = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    ymin, ymax = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    xpad = max((xmax - xmin) * 0.05, 1.0)
    ypad = max((ymax - ymin) * 0.05, 1.0)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    lo_x, hi_x = ax.get_xlim()
    lo_y, hi_y = ax.get_ylim()

    # Target cross
    target_center = np.nanmedian(target_xy, axis=0)
    ax.plot(
        target_center[0], target_center[1], marker="+",
        markersize=14, markeredgewidth=2.5, color="black", zorder=5,
        label="Target",
    )

    # Gaze points
    ax.scatter(
        gaze_sub[:, 0], gaze_sub[:, 1],
        c=gaze_color, s=6, alpha=0.35, zorder=3,
        label="Gaze Point",
    )

    # Offset vectors (target -> gaze), clipped to the frame
    for ai in indices[::arrow_subsample]:
        gx, gy = float(gaze_xy[ai, 0]), float(gaze_xy[ai, 1])
        tx, ty = float(target_xy[ai, 0]), float(target_xy[ai, 1])
        if abs(gx - tx) < 0.5 and abs(gy - ty) < 0.5:
            continue
        # Keep only arrows fully inside the frame
        if not (lo_x <= gx <= hi_x and lo_x <= tx <= hi_x
                and lo_y <= gy <= hi_y and lo_y <= ty <= hi_y):
            continue
        ax.add_patch(FancyArrowPatch(
            (tx, ty), (gx, gy),
            arrowstyle="-|>", color=arrow_color,
            lw=0.8, alpha=0.55, mutation_scale=7,
            zorder=2, clip_on=True,
        ))

    # Legend: Target / Gaze Point (auto-collected) + a proxy for the offset arrows
    from matplotlib.lines import Line2D
    arrow_proxy = Line2D(
        [], [], color=arrow_color, lw=1.2, alpha=0.7,
        marker=">", markersize=6, label="Offset Vector",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [arrow_proxy], labels + ["Offset Vector"],
        fontsize=8.5, loc="lower right", framealpha=0.85,
    )
    ax.set_aspect("equal", adjustable="datalim")
    return True


def plot_subject_trajectory(
    subject_data: dict,
    subject_id: str,
    task_type: str,
    output_path: Path,
    max_points: int = 600,
    arrow_subsample: int = 8,
):
    """1x2 panel: (A) Alert vs (B) Fatigue gaze scatter + offset vectors (PDF)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    panel_configs = [
        ("alert", "Alert", "#2196F3", axes[0], "A"),
        ("fatigue", "Fatigue", "#F44336", axes[1], "B"),
    ]

    for state_key, state_label, color, ax, panel_label in panel_configs:
        ok = _draw_trajectory_panel(
            ax, subject_data, state_key, color, max_points, arrow_subsample,
        )
        if not ok:
            ax.set_title(f"{state_label} (no data)", fontsize=12)
            continue
        ax.set_xlabel(f"({panel_label}) {state_label} — X (px)", fontsize=10)
        ax.set_ylabel("Y (px)", fontsize=10)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {output_path}")


def save_trajectory_panels_png(
    subject_data: dict,
    subject_id: str,
    task_type: str,
    png_dir: Path,
    max_points: int = 600,
    arrow_subsample: int = 8,
):
    """Save individual Alert / Fatigue trajectory panels as PNG (for PPT layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    for state_key, state_label, color in [
        ("alert", "Alert", "#2196F3"),
        ("fatigue", "Fatigue", "#F44336"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4.8))
        ok = _draw_trajectory_panel(
            ax, subject_data, state_key, color, max_points, arrow_subsample,
        )
        if not ok:
            plt.close(fig)
            continue
        ax.set_xlabel(f"X (px)", fontsize=10)
        ax.set_ylabel("Y (px)", fontsize=10)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        out = png_dir / f"traj_{state_key}_S{subject_id}_{task_type}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    [png] {out.name}")


# ===================================================================
# Plotting: average deviation distribution (combined Easy + Hard)
# ===================================================================

def _compute_mean_sem_density(
    all_subject_data: dict[str, dict],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute mean±SEM density curves across subjects for one task.

    Returns (bin_centers, alert_mean, alert_sem, fatigue_mean, fatigue_sem,
             global_alert_mean, global_fatigue_mean) or None if insufficient data.
    """
    all_alert_devs = []
    all_fatigue_devs = []
    for sid, data in all_subject_data.items():
        if "alert" in data and len(data["alert"]["deviation"]) > 0:
            all_alert_devs.append(data["alert"]["deviation"])
        if "fatigue" in data and len(data["fatigue"]["deviation"]) > 0:
            all_fatigue_devs.append(data["fatigue"]["deviation"])

    if not all_alert_devs or not all_fatigue_devs:
        return None

    # Shared bin range based on 99th percentile of all data
    all_vals = np.concatenate(all_alert_devs + all_fatigue_devs)
    upper = max(float(np.percentile(all_vals, 99)), 1.0)
    bins = np.linspace(0, upper, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    def per_subject_dists(dev_list):
        return np.array([np.histogram(dev, bins=bins, density=True)[0] for dev in dev_list])

    alert_densities = per_subject_dists(all_alert_devs)
    fatigue_densities = per_subject_dists(all_fatigue_devs)

    return (
        bin_centers,
        alert_densities.mean(axis=0),
        alert_densities.std(axis=0) / np.sqrt(len(alert_densities)),
        fatigue_densities.mean(axis=0),
        fatigue_densities.std(axis=0) / np.sqrt(len(fatigue_densities)),
        float(np.mean(np.concatenate(all_alert_devs))),
        float(np.mean(np.concatenate(all_fatigue_devs))),
    )


def _draw_avg_dist_panel(
    ax,
    stats: tuple,
    panel_label: str,
    task_type: str,
    show_ylabel: bool = True,
):
    """Draw one mean±SEM deviation-distribution panel for a single task."""
    (bin_centers, alert_mean, alert_sem,
     fatigue_mean, fatigue_sem, g_alert_mean, g_fatigue_mean) = stats

    ax.fill_between(bin_centers, alert_mean - alert_sem, alert_mean + alert_sem,
                    color="#2196F3", alpha=0.2)
    ax.plot(bin_centers, alert_mean, color="#1565C0", linewidth=2, label="Alert")

    ax.fill_between(bin_centers, fatigue_mean - fatigue_sem, fatigue_mean + fatigue_sem,
                    color="#F44336", alpha=0.2)
    ax.plot(bin_centers, fatigue_mean, color="#C62828", linewidth=2, label="Fatigue")

    ax.axvline(g_alert_mean, color="#1565C0", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(g_fatigue_mean, color="#C62828", linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_xlabel(f"{panel_label} {task_type.capitalize()} Task", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.tick_params(labelsize=9)


def plot_average_distribution_combined(
    task_subject_data: dict[str, dict[str, dict]],
    output_path: Path,
    n_bins: int = 50,
):
    """Combined 1x2 figure: (A) Easy Task, (B) Hard Task average distributions.

    Panel labels are placed below each subplot for LaTeX caption reference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    panels = []
    for task_type, panel_label in [("easy", "(A)"), ("hard", "(B)")]:
        subject_data = task_subject_data.get(task_type)
        if subject_data is None:
            continue
        result = _compute_mean_sem_density(subject_data, n_bins)
        if result is not None:
            panels.append((panel_label, task_type, result))

    if not panels:
        print("  [skip] Not enough data for average distribution figure")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.8), squeeze=False)
    axes = axes[0]

    for idx, (panel_label, task_type, stats) in enumerate(panels):
        _draw_avg_dist_panel(axes[idx], stats, panel_label, task_type, show_ylabel=(idx == 0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {output_path}")


def save_avg_dist_panels_png(
    task_subject_data: dict[str, dict[str, dict]],
    png_dir: Path,
    n_bins: int = 50,
):
    """Save each task's average distribution as a standalone PNG (for PPT layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    for task_type in ["easy", "hard"]:
        subject_data = task_subject_data.get(task_type)
        if subject_data is None:
            continue
        stats = _compute_mean_sem_density(subject_data, n_bins)
        if stats is None:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        _draw_avg_dist_panel(ax, stats, "", task_type, show_ylabel=True)
        plt.tight_layout()
        out = png_dir / f"avg_dist_{task_type}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    [png] {out.name}")


# ===================================================================
# Plotting: paper figure — 2 representative subjects combined
# ===================================================================

def _draw_histogram_panel(
    ax,
    alert_dev: np.ndarray,
    fatigue_dev: np.ndarray,
    show_ylabel: bool = True,
    n_bins: int = 50,
):
    """Draw one Alert-vs-Fatigue deviation histogram panel (full frame, μ in legend)."""
    from matplotlib.ticker import MaxNLocator

    all_vals = np.concatenate([alert_dev, fatigue_dev])
    upper = max(float(np.percentile(all_vals, 99)), 1.0)
    bins = np.linspace(0, upper, n_bins + 1)

    ax.hist(alert_dev, bins=bins, alpha=0.6, color="#2196F3", density=True,
            label=f"Alert (μ={np.mean(alert_dev):.1f})", edgecolor="white", linewidth=0.3)
    ax.hist(fatigue_dev, bins=bins, alpha=0.6, color="#F44336", density=True,
            label=f"Fatigue (μ={np.mean(fatigue_dev):.1f})", edgecolor="white", linewidth=0.3)

    ax.axvline(np.mean(alert_dev), color="#1565C0", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.axvline(np.mean(fatigue_dev), color="#C62828", linestyle="--", linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Deviation Distance (px)", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(labelsize=9)


def plot_paper_histogram_2subjects(
    all_subject_data: dict[str, dict],
    selected_subjects: list[str],
    task_type: str,
    output_path: Path,
    n_bins: int = 50,
):
    """1x2 panel histogram for 2 representative subjects (paper figure, PDF)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    n_subj = min(len(selected_subjects), 2)
    fig, axes = plt.subplots(1, n_subj, figsize=(4.5 * n_subj, 3.6), squeeze=False)
    axes = axes[0]

    for idx in range(n_subj):
        sid = selected_subjects[idx]
        data = all_subject_data[sid]
        _draw_histogram_panel(
            axes[idx], data["alert"]["deviation"], data["fatigue"]["deviation"],
            show_ylabel=(idx == 0), n_bins=n_bins,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {output_path}")


def save_histogram_panels_png(
    all_subject_data: dict[str, dict],
    selected_subjects: list[str],
    task_type: str,
    png_dir: Path,
    n_bins: int = 50,
):
    """Save each subject's deviation histogram as a standalone PNG (for PPT layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    for sid in selected_subjects[:2]:
        data = all_subject_data[sid]
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        _draw_histogram_panel(
            ax, data["alert"]["deviation"], data["fatigue"]["deviation"],
            show_ylabel=True, n_bins=n_bins,
        )
        plt.tight_layout()
        out = png_dir / f"hist_S{sid}_{task_type}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    [png] {out.name}")


# ===================================================================
# Subject selection
# ===================================================================

def select_representative_subjects(
    subject_data: dict[str, dict],
    n_subjects: int = 3,
) -> list[str]:
    """Select subjects with clearest alert-vs-fatigue deviation difference."""
    scores = []
    for sid, data in subject_data.items():
        if "alert" not in data or "fatigue" not in data:
            continue
        alert_dev = data["alert"]["deviation"]
        fatigue_dev = data["fatigue"]["deviation"]
        if len(alert_dev) == 0 or len(fatigue_dev) == 0:
            continue
        alert_mean = float(np.mean(alert_dev))
        fatigue_mean = float(np.mean(fatigue_dev))
        if alert_mean < 1e-6:
            continue
        ratio = fatigue_mean / alert_mean
        scores.append((sid, ratio, fatigue_mean - alert_mean))

    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [s[0] for s in scores[:n_subjects]]


# ===================================================================
# Main
# ===================================================================

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="ADF Deviation Qualitative Visualisation (Full Suite)"
    )
    parser.add_argument(
        "--data-root", type=str,
        default="/root/autodl-tmp/shenxy/Data/Process0620_calibrate",
        help="Root directory containing JSONL sequence files",
    )
    parser.add_argument(
        "--n-bins", type=int, default=50,
        help="Number of histogram bins (default: 50)",
    )
    parser.add_argument(
        "--max-points", type=int, default=600,
        help="Max gaze points per trajectory panel (default: 600)",
    )
    parser.add_argument(
        "--arrow-subsample", type=int, default=8,
        help="Draw 1 offset arrow every N frames (default: 8)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="visualisation_results",
        help="Output root directory (default: visualisation_results/)",
    )
    args = parser.parse_args()

    if args.arrow_subsample < 1:
        parser.error("--arrow-subsample must be >= 1")

    # ---- resolve data root ----
    data_root = Path(args.data_root)
    if not data_root.exists():
        try:
            import yaml
            cfg_path = _project_root / "configs" / "default.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            data_root = Path(cfg["data"]["root"])
        except Exception:
            pass

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Data root:   {data_root}")
    print(f"Output root: {output_root.resolve()}")
    print("=" * 60)

    # ---- process each task type ----
    task_subject_data: dict[str, dict[str, dict]] = {}  # {task_type: {sid: {...}}}

    for task_type in ["easy", "hard"]:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_type.upper()}")
        print(f"{'='*60}")

        # Discover sequences for this task
        all_seqs = discover_sequences(data_root)
        task_seqs = filter_sequences_by_task(all_seqs, task_type)
        print(f"\nSequences found: {len(task_seqs)}")

        if not task_seqs:
            print(f"  [skip] No {task_type} sequences found.")
            continue

        # ---- load per-subject data ----
        subject_data: dict[str, dict] = defaultdict(dict)

        for seq in task_seqs:
            records = list(iter_jsonl(seq.path))
            if len(records) < 64:
                continue

            deviation, gaze_xy, target_xy = extract_deviation_and_gaze(records, seq.task_type)
            if len(deviation) == 0:
                continue

            state_key = "alert" if seq.label == 0 else "fatigue"

            if state_key in subject_data[seq.subject_id]:
                existing = subject_data[seq.subject_id][state_key]
                existing["deviation"] = np.concatenate([existing["deviation"], deviation])
                existing["gaze_xy"] = np.concatenate([existing["gaze_xy"], gaze_xy], axis=0)
                existing["target_xy"] = np.concatenate([existing["target_xy"], target_xy], axis=0)
            else:
                subject_data[seq.subject_id][state_key] = {
                    "deviation": deviation,
                    "gaze_xy": gaze_xy,
                    "target_xy": target_xy,
                }

        # Filter subjects with both states
        valid_subjects = {
            sid: data for sid, data in subject_data.items()
            if "alert" in data and "fatigue" in data
        }
        print(f"Subjects with both Alert & Fatigue: {len(valid_subjects)}")

        if not valid_subjects:
            print(f"  [skip] No valid subjects for {task_type}.")
            continue

        # ---- create output directories ----
        hist_dir = output_root / task_type / "histogram"
        traj_dir = output_root / task_type / "trajectory"
        hist_dir.mkdir(parents=True, exist_ok=True)
        traj_dir.mkdir(parents=True, exist_ok=True)

        # ---- per-subject figures ----
        sorted_sids = sorted(valid_subjects.keys())
        print(f"\nGenerating per-subject figures for {len(sorted_sids)} subjects ...")

        for sid in sorted_sids:
            data = valid_subjects[sid]
            alert_dev = data["alert"]["deviation"]
            fatigue_dev = data["fatigue"]["deviation"]

            # Histogram
            hist_path = hist_dir / f"S{sid}_deviation_hist.pdf"
            plot_subject_histogram(
                alert_dev, fatigue_dev, sid, task_type, hist_path, n_bins=args.n_bins,
            )

            # Trajectory
            traj_path = traj_dir / f"S{sid}_gaze_trajectory.pdf"
            plot_subject_trajectory(
                data, sid, task_type, traj_path,
                max_points=args.max_points, arrow_subsample=args.arrow_subsample,
            )

            a_mean = np.mean(alert_dev)
            f_mean = np.mean(fatigue_dev)
            print(f"  S{sid}: Alert μ={a_mean:.1f}px, Fatigue μ={f_mean:.1f}px, "
                  f"ratio={f_mean/max(a_mean, 1e-6):.2f}x")

        # ---- store for combined average distribution ----
        task_subject_data[task_type] = valid_subjects

        # ---- paper figures (2 representative subjects) ----
        paper_dir = output_root / "paper_figures"
        panels_dir = paper_dir / "panels"
        paper_dir.mkdir(parents=True, exist_ok=True)
        panels_dir.mkdir(parents=True, exist_ok=True)

        selected = select_representative_subjects(valid_subjects, n_subjects=2)
        if selected:
            paper_hist_path = paper_dir / f"fig_histogram_2subjects_{task_type}.png"
            plot_paper_histogram_2subjects(
                valid_subjects, selected, task_type, paper_hist_path, n_bins=args.n_bins,
            )

            # Best subject trajectory for paper
            best_sid = selected[0]
            paper_traj_path = paper_dir / f"fig_trajectory_best_{task_type}.png"
            plot_subject_trajectory(
                valid_subjects[best_sid], best_sid, task_type, paper_traj_path,
                max_points=args.max_points, arrow_subsample=args.arrow_subsample,
            )
            print(f"  [paper] Best subject for trajectory: S{best_sid}")

            # ---- individual subplots as PNG (for PPT re-layout) ----
            print(f"  Saving individual subplots (PNG) for {task_type} ...")
            save_histogram_panels_png(
                valid_subjects, selected, task_type, panels_dir, n_bins=args.n_bins,
            )
            save_trajectory_panels_png(
                valid_subjects[best_sid], best_sid, task_type, panels_dir,
                max_points=args.max_points, arrow_subsample=args.arrow_subsample,
            )

    # ---- combined average distribution: (A) Easy + (B) Hard ----
    if task_subject_data:
        print(f"\n{'='*60}")
        print("  COMBINED AVERAGE DISTRIBUTION")
        print(f"{'='*60}")
        avg_path = output_root / "average_deviation_distribution.pdf"
        plot_average_distribution_combined(task_subject_data, avg_path, n_bins=args.n_bins)

        # Individual average-distribution panels as PNG (for PPT re-layout)
        panels_dir = output_root / "paper_figures" / "panels"
        panels_dir.mkdir(parents=True, exist_ok=True)
        save_avg_dist_panels_png(task_subject_data, panels_dir, n_bins=args.n_bins)

    # ---- summary ----
    print(f"\n{'='*60}")
    print("DONE. Output structure:")
    print(f"  {output_root.resolve()}/")
    print(f"  ├── easy/")
    print(f"  │   ├── histogram/       (per-subject deviation histograms)")
    print(f"  │   └── trajectory/      (per-subject gaze trajectory comparisons)")
    print(f"  ├── hard/")
    print(f"  │   ├── histogram/")
    print(f"  │   └── trajectory/")
    print(f"  ├── average_deviation_distribution.pdf   (A: Easy, B: Hard)")
    print(f"  └── paper_figures/")
    print(f"      ├── fig_histogram_2subjects_easy.png")
    print(f"      ├── fig_histogram_2subjects_hard.png")
    print(f"      ├── fig_trajectory_best_easy.png")
    print(f"      ├── fig_trajectory_best_hard.png")
    print(f"      └── panels/          (individual subplots as PNG for PPT)")
    print(f"          ├── hist_S<id>_<task>.png")
    print(f"          ├── traj_alert_S<id>_<task>.png / traj_fatigue_S<id>_<task>.png")
    print(f"          └── avg_dist_easy.png / avg_dist_hard.png")


if __name__ == "__main__":
    main()
