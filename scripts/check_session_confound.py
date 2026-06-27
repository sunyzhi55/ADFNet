"""快速检查：alert vs sleep 是不是同一录制条件。

目的：区分 [C] 高 AUC 的两种解释
  - 解释1：面部特征携带真实疲劳表达（眼睑/头姿/哈欠）→ 用 gaze + 解耦是对的
  - 解释2：alert/sleep 分两次录制，坐姿/相机/头部位置不同 → 面部信号是 session 伪信号

检测信号：
  1. 同被试 alert vs sleep 的 face_detection_bbox 均值差异（绝对 px 和相对宽高比）
  2. 同被试 alert vs sleep 的 RetinaFace_landmarks 均值差异
  3. 同被试 alert vs sleep 的 pitch_yaw_rad 均值差异（头部姿态 —— 最能区分 session 混淆）
  4. 同被试内：用 bbox/landmark/pitch_yaw 预测 label 的 AUC，看是不是 bbox 单独就能分

如果 alert/sleep 的 bbox 或 pitch_yaw 系统性偏移，且 bbox 单独预测 label 的 AUC 就很高，
那 [C] 的高 AUC 很可能是 session/录制条件混淆，不是纯疲劳表达。

跑法：
  python scripts/check_session_confound.py --config configs/default.yaml --task-mode hard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.io import SequenceInfo, discover_sequences, filter_sequences_by_task, iter_jsonl  # noqa: E402
from utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check alert vs sleep recording-condition confound")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--task-mode", choices=["all", "easy", "hard"], default=None)
    p.add_argument("--output-dir", default="outputs/face_diag")
    return p.parse_args()


def task_mode_from_args(cfg: dict, task_mode: str | None) -> str:
    return task_mode or cfg.get("data", {}).get("task_mode", "all")


def seq_stats(path: Path) -> dict | None:
    """逐帧聚合一条序列的几何统计。"""
    bbox_list, lmk_list, pitch_list, yaw_list = [], [], [], []
    for rec in iter_jsonl(path):
        bbox = rec.get("face_detection_bbox") or rec.get("RetinaFace_bbox")
        if bbox and len(bbox) >= 4:
            bbox_list.append([float(bbox[0]), float(bbox[1]),
                              float(bbox[2]) - float(bbox[0]),   # w
                              float(bbox[3]) - float(bbox[1])])  # h
        lmk = rec.get("RetinaFace_landmarks")
        if lmk:
            try:
                import numpy as _np
                pts = _np.asarray(lmk, dtype=np.float32).reshape(-1, 2)
                if pts.size >= 2:
                    lmk_list.append(pts.mean(axis=0).tolist())
            except Exception:
                pass
        py = rec.get("pitch_yaw_rad")
        if py and len(py) >= 2:
            pitch_list.append(float(py[0]))
            yaw_list.append(float(py[1]))
    if not bbox_list:
        return None
    bbox_arr = np.asarray(bbox_list, dtype=np.float32)
    out = {
        "bbox_x": float(bbox_arr[:, 0].mean()),
        "bbox_y": float(bbox_arr[:, 1].mean()),
        "bbox_w": float(bbox_arr[:, 2].mean()),
        "bbox_h": float(bbox_arr[:, 3].mean()),
        "bbox_aspect": float((bbox_arr[:, 2] / np.clip(bbox_arr[:, 3], 1e-6, None)).mean()),
        "bbox_center_x": float((bbox_arr[:, 0] + bbox_arr[:, 2] / 2).mean()),
        "bbox_center_y": float((bbox_arr[:, 1] + bbox_arr[:, 3] / 2).mean()),
    }
    if lmk_list:
        lmk_arr = np.asarray(lmk_list, dtype=np.float32)
        out["lmk_cx"] = float(lmk_arr[:, 0].mean())
        out["lmk_cy"] = float(lmk_arr[:, 1].mean())
    if pitch_list:
        out["pitch_mean"] = float(np.mean(pitch_list))
        out["yaw_mean"] = float(np.mean(yaw_list))
    out["n_frames_bbox"] = len(bbox_list)
    out["n_frames_lmk"] = len(lmk_list)
    out["n_frames_pose"] = len(pitch_list)
    return out


def auc_single_feature(values: np.ndarray, labels: np.ndarray) -> float:
    """单特征预测 label 的 AUC（取 |AUC-0.5| 的较大方向）。"""
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        a = roc_auc_score(labels, values)
        return float(max(a, 1 - a))
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    task_mode = task_mode_from_args(cfg, args.task_mode)
    sequences = filter_sequences_by_task(discover_sequences(cfg["data"]["root"]), task_mode)
    if not sequences:
        raise RuntimeError(f"No sequences for task_mode={task_mode}")

    # 按 (subject, label) 聚合序列几何均值
    rows = []
    seq_by_subj_label: dict[tuple[str, int], list[dict]] = {}
    for seq in sequences:
        s = seq_stats(seq.path)
        if s is None:
            continue
        s.update(subject_id=seq.subject_id, label=seq.label, task=seq.task_type,
                 label_name=seq.label_name, path=str(seq.path))
        rows.append(s)
        seq_by_subj_label.setdefault((seq.subject_id, seq.label), []).append(s)

    if not rows:
        raise RuntimeError("No face_detection_bbox found in any JSONL. Cannot check confound.")
    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] task_mode={task_mode}  sequences_with_bbox={len(df)}  "
          f"subjects={df['subject_id'].nunique()}")
    print(f"[data] frames: bbox mean={df['n_frames_bbox'].mean():.0f}  "
          f"landmark mean={df['n_frames_lmk'].mean():.0f}  "
          f"pose mean={df['n_frames_pose'].mean():.0f}\n")

    # ---- 1. 同被试 alert vs sleep 几何差异 ----
    geom_cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_aspect",
                 "bbox_center_x", "bbox_center_y", "lmk_cx", "lmk_cy",
                 "pitch_mean", "yaw_mean"]
    geom_cols = [c for c in geom_cols if c in df.columns]

    print("========== 1) 同被试 alert vs sleep 几何均值差异 ==========")
    diff_rows = []
    for sid, g in df.groupby("subject_id"):
        alert = g[g.label == 0]
        sleep = g[g.label == 1]
        if alert.empty or sleep.empty:
            continue
        a, s = alert.iloc[0], sleep.iloc[0]
        row = {"subject_id": sid}
        for c in geom_cols:
            av, sv = float(a[c]), float(s[c])
            row[f"{c}_alert"] = av
            row[f"{c}_sleep"] = sv
            # 相对差异：用 |Δ| / (|alert|+|sleep|+eps) 归一，避免量纲
            denom = abs(av) + abs(sv) + 1e-6
            row[f"{c}_reldiff"] = abs(av - sv) / denom
        diff_rows.append(row)
    diff_df = pd.DataFrame(diff_rows)

    if not diff_df.empty:
        print("各几何量 alert→sleep 的平均相对差异（越大越像 session 混淆）：")
        relcols = [c for c in diff_df.columns if c.endswith("_reldiff")]
        means = diff_df[relcols].mean().sort_values(ascending=False)
        for c, v in means.items():
            base = c.replace("_reldiff", "")
            print(f"    {base:18s} reldiff = {v:.3f}")
        print("\n头部姿态绝对差异（pitch/yaw，rad）：")
        for c in ["pitch_mean", "yaw_mean"]:
            if f"{c}_alert" in diff_df.columns:
                d = (diff_df[f"{c}_alert"] - diff_df[f"{c}_sleep"]).abs()
                print(f"    {c:18s} |Δ| mean={d.mean():.4f}  std={d.std():.4f}")
        print("\n人脸框宽高绝对差异（px）：")
        for c in ["bbox_w", "bbox_h"]:
            if f"{c}_alert" in diff_df.columns:
                d = (diff_df[f"{c}_alert"] - diff_df[f"{c}_sleep"]).abs()
                print(f"    {c:18s} |Δ| mean={d.mean():.2f}  std={d.std():.2f}")
        diff_df.to_csv(out_dir / f"session_confound_per_subject_{task_mode}.csv", index=False)
        print(f"\n[saved] per-subject diff -> {out_dir / f'session_confound_per_subject_{task_mode}.csv'}")

    # ---- 2. 几何量单独预测 label 的 AUC（同被试内） ----
    print("\n========== 2) 几何量单独预测 label 的 AUC（同被试内，控身份） ==========")
    print("若某个几何量 alone 的 within-subject AUC 就接近 1，说明 [C] 高 AUC 来自录制条件而非疲劳表达。")
    feat_cols = [c for c in geom_cols if c in df.columns]
    auc_rows = []
    for sid, g in df.groupby("subject_id"):
        if g.label.nunique() < 2 or len(g) < 4:
            continue
        for c in feat_cols:
            vals = g[c].to_numpy(dtype=np.float64)
            labs = g.label.to_numpy()
            if np.any(~np.isfinite(vals)):
                vals = np.nan_to_num(vals, nan=np.nanmean(vals) if np.isfinite(np.nanmean(vals)) else 0.0)
            auc = auc_single_feature(vals, labs)
            auc_rows.append({"subject_id": sid, "feature": c, "auc": auc, "n": len(g)})
    auc_df = pd.DataFrame(auc_rows)
    if not auc_df.empty:
        # 同被试内只有 2~4 条序列，AUC 会极端（0/0.5/1），用"中位数"和"==1 比例"看趋势
        per_feat = auc_df.groupby("feature")["auc"].agg(["median", "mean", lambda x: np.mean(x >= 0.95)]).reset_index()
        per_feat.columns = ["feature", "auc_median", "auc_mean", "frac_auc_ge0.95"]
        per_feat = per_feat.sort_values("auc_median", ascending=False)
        print(per_feat.to_string(index=False))
        per_feat.to_csv(out_dir / f"session_confound_feature_auc_{task_mode}.csv", index=False)
        print(f"\n[saved] feature AUC -> {out_dir / f'session_confound_feature_auc_{task_mode}.csv'}")

    # ---- 3. 判定 ----
    print("\n========== 判定 ==========")
    if not auc_df.empty:
        # 用 pitch / bbox 单独的 within-subject AUC 中位数判断
        pose_auc = per_feat[per_feat.feature.isin(["pitch_mean", "yaw_mean"])]["auc_median"]
        bbox_auc = per_feat[per_feat.feature.isin(["bbox_w", "bbox_h", "bbox_aspect",
                                                    "bbox_center_x", "bbox_center_y"])]["auc_median"]
        pose_med = pose_auc.median() if len(pose_auc) else float("nan")
        bbox_med = bbox_auc.median() if len(bbox_auc) else float("nan")
        print(f"姿态(pitch/yaw) within-subject AUC 中位数 = {pose_med:.3f}")
        print(f"人脸框(bbox*) within-subject AUC 中位数 = {bbox_med:.3f}")
        # 注意：每被试序列少，AUC 噪声大，这是粗判
        if (not np.isnan(bbox_med) and bbox_med > 0.9) or (not np.isnan(pose_med) and pose_med > 0.9):
            print("→ alert/sleep 的 bbox/姿态系统性偏移 → [C] 高 AUC 很可能是录制条件混淆，"
                  "不是纯疲劳表达。强烈建议确认录制条件；面部特征更不可当对抗目标。")
        else:
            print("→ bbox/姿态偏移不显著 → [C] 高 AUC 更可能来自真实疲劳表达（眼睑/微表情），"
                  "但仍有时间自相关膨胀。面部特征仍不可当对抗目标（泄漏 label）。")


if __name__ == "__main__":
    main()
