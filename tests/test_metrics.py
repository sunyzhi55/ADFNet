from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.metrics import binary_metrics


def test_binary_metrics_handles_nan_probabilities() -> None:
    metrics = binary_metrics([0, 1, 1], [0.1, np.nan, np.inf], threshold=0.5)
    assert set(metrics) == {"auc", "acc", "f1", "precision", "recall"}
    assert np.isfinite(metrics["acc"])
