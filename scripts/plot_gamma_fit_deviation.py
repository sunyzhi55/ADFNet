#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distribution fitting comparison for gaze deviation data.

Fits Gamma, Gaussian, Lognormal, Weibull, and Rayleigh distributions to
deviation_px values from calibrated JSONL files.  Produces one figure per
field, each showing:
  - histogram
  - five fitted PDF curves (Gamma / Gaussian / Lognormal / Weibull / Rayleigh)
  - goodness-of-fit metrics (AIC, BIC, KS p-value) below the plot

Two figures are produced by default:
    1. deviation_px_before_calibrate_dist_fit.png
    2. deviation_px_after_calibrate_dist_fit.png

Example:
    python scripts/plot_gamma_fit_deviation.py \\
        --input_path /path/to/calibrated_jsonl \\
        --state alert
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

FIELD_BEFORE = "deviation_px_before_calibrate"
FIELD_AFTER = "deviation_px_after_calibrate"
STATE_CHOICES = ("all", "alert", "sleepy")
FILENAME_RE = re.compile(r"^\d+_(easy|hard)_(alert|sleepy)$", re.IGNORECASE)

# DIST_NAMES: Tuple[str, ...] = ("Gamma", "Gaussian", "Lognormal", "Weibull", "Rayleigh")
DIST_NAMES: Tuple[str, ...] = ("Gamma", "Lognormal", "Weibull", "Rayleigh")
DIST_COLORS = {
    "Gamma": "#e74c3c",
    # "Gaussian": "#3498db",
    "Lognormal": "#2ecc71",
    "Weibull": "#9b59b6",
    "Rayleigh": "#3498db",
}


def _require_scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for distribution fitting and KS testing. "
            "Install: pip install -r requirements.txt"
        ) from exc
    return stats


# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GammaFitResult:
    shape: float
    loc: float
    scale: float
    ks_statistic: float
    p_value: float
    sample_count: int


@dataclass(frozen=True)
class DistributionFitResult:
    name: str
    params: dict
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    p_value: float
    sample_count: int

    @property
    def family(self) -> str:
        return self.name.lower()

    @property
    def scipy_args(self) -> tuple:
        if self.family == "gamma":
            return (self.params["shape"], self.params["loc"], self.params["scale"])
        if self.family == "gaussian":
            return (self.params["mu"], self.params["sigma"])
        if self.family == "weibull":
            return (self.params["c"], self.params["loc"], self.params["scale"])
        if self.family == "rayleigh":
            return (self.params["loc"], self.params["scale"])
        return (self.params["s"], self.params["loc"], self.params["scale"])


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def minmax_normalize(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    vmin, vmax = float(data.min()), float(data.max())
    if vmax <= vmin:
        return data.copy()
    return (data - vmin) / (vmax - vmin)


def _iter_jsonl_files(input_path: Path) -> List[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.jsonl") if p.is_file())


def _matches_state_filter(path: Path, state: str) -> bool:
    if state == "all":
        return True
    m = FILENAME_RE.match(path.stem)
    return m is not None and m.group(2).lower() == state


def _valid_positive_number(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v >= 0.0


def collect_deviation_values(
    input_path: Path,
    field: str = FIELD_AFTER,
    state: str = "all",
) -> Tuple[np.ndarray, int]:
    if state not in STATE_CHOICES:
        raise ValueError(f"state must be one of {STATE_CHOICES}, got: {state}")
    jsonl_files = _iter_jsonl_files(Path(input_path))
    jsonl_files = [p for p in jsonl_files if _matches_state_filter(p, state)]
    values: List[float] = []
    for jf in jsonl_files:
        with jf.open("r", encoding="utf-8-sig") as fp:
            for ln, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {jf}:{ln}") from exc
                val = record.get(field)
                if _valid_positive_number(val):
                    values.append(float(val))
    return np.asarray(values, dtype=np.float64), len(jsonl_files)


# ══════════════════════════════════════════════════════════════════════
# Fitting
# ══════════════════════════════════════════════════════════════════════

def fit_gamma_distribution(values: Sequence[float]) -> GammaFitResult:
    """Legacy Gamma-only fit (backward compat)."""
    stats = _require_scipy_stats()
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data) & (data > 0.0)]
    if data.size < 2:
        raise ValueError("Need at least 2 finite positive values.")
    a, loc, s = stats.gamma.fit(data, floc=0.0)
    ks, pv = stats.kstest(data, "gamma", args=(a, loc, s))
    return GammaFitResult(
        shape=float(a), loc=float(loc), scale=float(s),
        ks_statistic=float(ks), p_value=float(pv), sample_count=int(data.size),
    )


def _fit_single(stats, data: np.ndarray, name: str) -> DistributionFitResult:
    """Fit one distribution, compute LogLik / AIC / BIC / KS."""
    n = data.size

    if name == "Gamma":
        k = 2  # shape, scale  (loc fixed to 0)
        a, _, s = stats.gamma.fit(data, floc=0.0)
        ll = float(np.sum(stats.gamma.logpdf(data, a, 0.0, s)))
        ks, pv = stats.kstest(data, "gamma", args=(a, 0.0, s))
        params = {"shape": float(a), "loc": 0.0, "scale": float(s)}
    elif name == "Gaussian":
        k = 2  # mu, sigma
        mu, sig = stats.norm.fit(data)
        ll = float(np.sum(stats.norm.logpdf(data, mu, sig)))
        ks, pv = stats.kstest(data, "norm", args=(mu, sig))
        params = {"mu": float(mu), "sigma": float(sig)}
    elif name == "Lognormal":
        k = 2  # s, scale  (loc fixed to 0)
        s, _, sc = stats.lognorm.fit(data, floc=0.0)
        ll = float(np.sum(stats.lognorm.logpdf(data, s, 0.0, sc)))
        ks, pv = stats.kstest(data, "lognorm", args=(s, 0.0, sc))
        params = {"s": float(s), "loc": 0.0, "scale": float(sc)}
    elif name == "Weibull":
        # scipy.stats.weibull_min: shape c, loc, scale.  Fix loc=0.
        k = 2  # c, scale
        c, _, sc = stats.weibull_min.fit(data, floc=0.0)
        ll = float(np.sum(stats.weibull_min.logpdf(data, c, 0.0, sc)))
        ks, pv = stats.kstest(data, "weibull_min", args=(c, 0.0, sc))
        params = {"c": float(c), "loc": 0.0, "scale": float(sc)}
    elif name == "Rayleigh":
        # scipy.stats.rayleigh: loc, scale.  Fix loc=0.
        k = 1  # scale only
        _, sc = stats.rayleigh.fit(data, floc=0.0)
        ll = float(np.sum(stats.rayleigh.logpdf(data, 0.0, sc)))
        ks, pv = stats.kstest(data, "rayleigh", args=(0.0, sc))
        params = {"loc": 0.0, "scale": float(sc)}
    else:
        raise ValueError(f"Unknown distribution: {name}")

    aic = 2 * k - 2 * ll
    bic = k * math.log(n) - 2 * ll
    return DistributionFitResult(
        name=name, params=params,
        log_likelihood=ll, aic=float(aic), bic=float(bic),
        ks_statistic=float(ks), p_value=float(pv), sample_count=n,
    )


def fit_all_distributions(
    values: Sequence[float],
) -> Dict[str, DistributionFitResult]:
    """Fit Gamma, Gaussian, Lognormal, Weibull, Rayleigh and return metrics."""
    stats = _require_scipy_stats()
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data) & (data > 0.0)]
    if data.size < 3:
        raise ValueError("Need at least 3 finite positive values.")
    return {name: _fit_single(stats, data, name) for name in DIST_NAMES}


# ══════════════════════════════════════════════════════════════════════
# PDF helper
# ══════════════════════════════════════════════════════════════════════

def _pdf_on_grid(x, name, result, stats):
    if name == "Gamma":
        return stats.gamma.pdf(x, *result.scipy_args)
    if name == "Gaussian":
        return stats.norm.pdf(x, *result.scipy_args)
    if name == "Weibull":
        return stats.weibull_min.pdf(x, *result.scipy_args)
    if name == "Rayleigh":
        return stats.rayleigh.pdf(x, *result.scipy_args)
    return stats.lognorm.pdf(x, *result.scipy_args)


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

def _fmt_legend_label(r: DistributionFitResult) -> str:
    """Compact metrics string for legend: Name (AIC=… BIC=… KS=… p=…)."""
    if r.p_value > 0.999:
        pv = "p>0.999"
    elif r.p_value < 0.001:
        pv = f"p={r.p_value:.2e}"
    else:
        pv = f"p={r.p_value:.3f}"
    return (
        f"{r.name}  "
        f"(AIC={r.aic:,.0f}  BIC={r.bic:,.0f}  "
        f"KS={r.ks_statistic:.3f})"
    )


def plot_distribution_fit(
    values: Sequence[float],
    output_path: Path,
    bins=35,
    title: str = "Distribution Comparison",
    xlabel: str = "Deviation Distance (px)",
    figure_dpi: int = 180,
) -> Dict[str, DistributionFitResult]:
    """Single-panel figure: histogram + 3 fitted PDFs with metrics in legend."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install: pip install -r requirements.txt"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = _require_scipy_stats()

    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data) & (data >= 0.0)]
    results = fit_all_distributions(data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(
        data, bins=bins, density=True,
        color="#d5d5d5", edgecolor="#999999",
        linewidth=0.6, alpha=0.7, label="Original Data Histogram",
    )

    # x grid
    x_max = float(np.percentile(data, 99.5)) * 1.15
    x_max = max(x_max, float(data.max()) * 1.05)
    if x_max <= 0:
        x_max = 1.0
    x = np.linspace(0.0, x_max, 800)

    # Fitted PDFs — legend carries metrics
    for name in DIST_NAMES:
        ax.plot(
            x, _pdf_on_grid(x, name, results[name], stats),
            color=DIST_COLORS[name], linewidth=2.2,
            label=_fmt_legend_label(results[name]),
        )

    # Axes
    ax.set_xlabel("Deviation Distance (px)", fontsize=12, labelpad=6)
    ax.set_ylabel("Density", fontsize=12, labelpad=6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        loc="upper right", frameon=True, fontsize=12,
        framealpha=0.92, edgecolor="#cccccc",
    )

    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return results


# ══════════════════════════════════════════════════════════════════════
# Legacy plotting (backward compat)
# ══════════════════════════════════════════════════════════════════════

def _x_grid_for_pdf(values: np.ndarray, fit: GammaFitResult) -> np.ndarray:
    x_max = float(np.percentile(values, 99.5))
    x_max = max(x_max, float(values.max()), fit.scale * fit.shape * 3.0)
    if x_max <= 0:
        x_max = 1.0
    return np.linspace(0.0, x_max, 600)


def plot_gamma_fit(
    values, output_path, bins=35, title="Gamma Fit",
    xlabel="Deviation Distance (px)", figure_dpi=180,
) -> GammaFitResult:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("matplotlib is required.") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = _require_scipy_stats()
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data) & (data >= 0.0)]
    fit = fit_gamma_distribution(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = _x_grid_for_pdf(data, fit)
    y = stats.gamma.pdf(x, fit.shape, loc=fit.loc, scale=fit.scale)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.hist(data, bins=bins, density=True, color="#b8bdb8",
            edgecolor="#6f7771", linewidth=1.0, alpha=0.9)
    ax.plot(x, y, color="#ff2d2d", linewidth=2.5, label="Gamma Curve")
    ax.set_title(f"{title}\nKS={fit.ks_statistic:.3f}, p={fit.p_value:.3g}", fontsize=16)
    ax.set_xlabel("Deviation Distance (px)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color("#444444"); sp.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_dpi)
    plt.close(fig)
    return fit


def plot_gamma_fit_comparison(
    values, output_path, bins=35, title="Gamma Fit",
    xlabel="Deviation Distance (px)", figure_dpi=180,
) -> Tuple[GammaFitResult, GammaFitResult]:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("matplotlib is required.") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = _require_scipy_stats()
    data_raw = np.asarray(values, dtype=np.float64)
    data_raw = data_raw[np.isfinite(data_raw) & (data_raw >= 0.0)]
    data_norm = minmax_normalize(data_raw)
    fit_raw = fit_gamma_distribution(data_raw)
    fit_norm = fit_gamma_distribution(data_norm)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.0, 5.5))
    for ax, data, fit, sub in [
        (ax0, data_raw, fit_raw, "Original"),
        (ax1, data_norm, fit_norm, "Min-Max Normalized"),
    ]:
        x = _x_grid_for_pdf(data, fit)
        y = stats.gamma.pdf(x, fit.shape, loc=fit.loc, scale=fit.scale)
        ax.hist(data, bins=bins, density=True, color="#b8bdb8",
                edgecolor="#6f7771", linewidth=1.0, alpha=0.9)
        ax.plot(x, y, color="#ff2d2d", linewidth=2.5, label="Gamma Curve")
        ax.set_title(f"{sub}\nKS={fit.ks_statistic:.3f}, p={fit.p_value:.3g}", fontsize=16)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color("#444444"); sp.set_linewidth(1.0)
    fig.suptitle(title, fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return fit_raw, fit_norm


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
# /data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_l2cs/Data0620_tf_calibrate
def _parse_bins(value: str):
    """Accept an integer or 'auto' for the --bins CLI argument."""
    if value.lower() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bins must be a positive integer or 'auto', got: {value}"
        )
    if n <= 0:
        raise argparse.ArgumentTypeError(f"bins must be positive, got: {n}")
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit Gamma / Gaussian / Lognormal / Weibull / Rayleigh to "
                    "gaze deviation data and produce comparison figures with "
                    "AIC, BIC, KS metrics."
    )
    p.add_argument(
        "--input_path",
        default=(
            "/root/autodl-tmp/shenxy/Data/Process0620_calibrate"
        ),
    )
    p.add_argument("--output_before", default="deviation_px_before_calibrate_dist_fit.png")
    p.add_argument("--output_after", default="deviation_px_after_calibrate_dist_fit.png")
    p.add_argument("--field", default=None,
                   help="Single field to plot. If unset, both before+after are plotted.")
    p.add_argument("--state", default="alert", choices=STATE_CHOICES)
    p.add_argument("--bins", type=_parse_bins, default="auto",
                   help="Histogram bins: positive integer or 'auto' (default: auto)")
    p.add_argument("--title", default="Distribution Fitting Comparison")
    p.add_argument("--xlabel", default="Deviation Distance (px)")
    p.add_argument("--figure_dpi", type=int, default=300, help="Figure DPI (default: 300)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Single-field mode ───────────────────────────────────────────
    if args.field is not None:
        values, fc = collect_deviation_values(
            Path(args.input_path), field=args.field, state=args.state,
        )
        if values.size == 0:
            raise ValueError(f"No valid values for '{args.field}'")
        results = plot_distribution_fit(
            values, Path(args.output_before),
            bins=args.bins,
            title=f"{args.title} \u2014 {args.field}",
            xlabel=args.xlabel or args.field,
            figure_dpi=args.figure_dpi,
        )
        print(f"Files: {fc}  |  State: {args.state}  |  Field: {args.field}")
        for name in DIST_NAMES:
            r = results[name]
            print(f"  {name:<10} AIC={r.aic:,.1f}  BIC={r.bic:,.1f}  "
                  f"KS={r.ks_statistic:.4f}  p={r.p_value:.4g}")
        print(f"Saved: {args.output_before}")
        return

    # ── Dual-field mode ─────────────────────────────────────────────
    all_results: dict = {}
    for field, out_path, subtitle in [
        (FIELD_BEFORE, Path(args.output_before), "Before Calibration"),
        (FIELD_AFTER, Path(args.output_after), ""),
    ]:
        values, fc = collect_deviation_values(
            Path(args.input_path), field=field, state=args.state,
        )
        if values.size == 0:
            print(f"[SKIP] No valid values for '{field}'")
            continue
        results = plot_distribution_fit(
            values, out_path,
            bins=args.bins,
            title=f"{args.title}{subtitle}",
            xlabel=args.xlabel or field,
            figure_dpi=args.figure_dpi,
        )
        all_results[field] = (results, out_path)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\nInput files : {fc}")
    print(f"State filter: {args.state}\n")
    for field, (results, out_path) in all_results.items():
        n = next(iter(results.values())).sample_count
        print(f"\u2550 {field}  (n={n:,}) \u2550")
        for name in DIST_NAMES:
            r = results[name]
            print(f"  {name:<10}  AIC={r.aic:>12,.1f}  BIC={r.bic:>12,.1f}  "
                  f"KS={r.ks_statistic:.4f}  p={r.p_value:.4g}")
        print(f"  -> {out_path}\n")


if __name__ == "__main__":
    main()
