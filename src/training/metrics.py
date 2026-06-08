from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(labels: list[float] | np.ndarray, probabilities: list[float] | np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(labels).reshape(-1).astype(int)
    y_prob = np.asarray(probabilities).reshape(-1)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    return {
        "auc": auc,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "cm_tn": int(tn),
        "cm_fp": int(fp),
        "cm_fn": int(fn),
        "cm_tp": int(tp),
    }
