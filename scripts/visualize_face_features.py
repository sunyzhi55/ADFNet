"""可视化诊断：面部特征空间编码的是「身份」还是「疲劳」？

回答两个问题，决定 GRL 对抗目标能不能用面部特征：
  A. 身份编码强度 —— 面部特征能否预测 subject_id？（强 → 编码身份）
  B. 疲劳编码强度 —— 面部特征能否预测 label？
     关键是「同被试内」预测：如果同一个人内部面部特征就能区分疲劳，
     说明面部特征携带了疲劳表达本身，用它当对抗目标会把疲劳一起擦掉（泄漏）。

跑法：
  python scripts/visualize_face_features.py --config configs/default.yaml --task-mode hard
  python scripts/visualize_face_features.py --config configs/default.yaml --task-mode easy \
      --max-samples 5000 --output-dir outputs/face_diag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import ADFWindowDataset  # noqa: E402
from data.io import discover_sequences, filter_sequences_by_task  # noqa: E402
from utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose whether face-landmark features encode identity or fatigue")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--task-mode", choices=["all", "easy", "hard"], default=None)
    p.add_argument("--max-samples", type=int, default=8000,
                   help="抽样上限，避免 kNN LOO 过慢；None 表示全量")
    p.add_argument("--output-dir", default="outputs/face_diag")
    p.add_argument("--knn-k", type=int, default=7)
    p.add_argument("--no-plot", action="store_true", help="跳过画图，只做定量诊断")
    return p.parse_args()


def task_mode_from_args(cfg: dict, task_mode: str | None) -> str:
    return task_mode or cfg.get("data", {}).get("task_mode", "all")


def dataset_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg["data"])
    kwargs.pop("root", None)
    kwargs.pop("task_mode", None)
    return kwargs


def collect_dataset(cfg: dict, task_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (landmarks[N,D], subject_ids[N], labels[N])，subject_ids 用整数编码。"""
    sequences = filter_sequences_by_task(discover_sequences(cfg["data"]["root"]), task_mode)
    if not sequences:
        raise RuntimeError(f"No JSONL sequences found for task_mode={task_mode} under {cfg['data']['root']}")
    ds = ADFWindowDataset(sequences=sequences, **dataset_kwargs(cfg))
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty (no windows). Check window_size vs sequence length.")
    lmks, sids, labs = [], [], []
    subj2id: dict[str, int] = {}
    for s in ds.samples:
        if s.subject_id not in subj2id:
            subj2id[s.subject_id] = len(subj2id)
        lmks.append(s.landmarks)
        sids.append(subj2id[s.subject_id])
        labs.append(int(s.label))
    print(f"[data] task_mode={task_mode} subjects={len(subj2id)} windows={len(ds)} "
          f"landmark_dim={ds.samples[0].landmarks.shape}")
    pos = np.mean(labs)
    print(f"[data] positive(label=1, fatigue) ratio = {pos:.3f}")
    print(f"[data] subjects: {dict(zip(subj2id.keys(), subj2id.values()))}")
    return np.asarray(lmks, dtype=np.float32), np.asarray(sids), np.asarray(labs), subj2id


def knn_predict(X: np.ndarray, y: np.ndarray, k: int, idx_train: np.ndarray, idx_test: np.ndarray) -> np.ndarray:
    clf = KNeighborsClassifier(n_neighbors=min(k, len(idx_train)))
    clf.fit(X[idx_train], y[idx_train])
    return clf.predict(X[idx_test])


def loo_knn_accuracy(X: np.ndarray, y: np.ndarray, k: int, max_n: int = 4000) -> float:
    """近似 LOO：样本多时用分层抽样 holdout 多次平均，样本少时真 LOO。"""
    if len(y) <= max_n:
        loo = LeaveOneOut()
        preds = np.empty(len(y), dtype=y.dtype)
        for tr, te in loo.split(X):
            preds[te] = knn_predict(X, y, k, tr, te)
        return accuracy_score(y, preds)
    # 大样本：重复 stratified holdout 平均
    accs = []
    sss = StratifiedShuffleSplit(n_splits=5, test_size=max_n, random_state=42)
    for tr, te in sss.split(X, y):
        preds = knn_predict(X, y, k, tr, te)
        accs.append(accuracy_score(y[te], preds))
    return float(np.mean(accs))


def within_subject_label_predictability(X: np.ndarray, sids: np.ndarray, labs: np.ndarray, k: int) -> dict:
    """同被试内：用面部特征预测疲劳。这是泄漏检测的核心。
    只保留同时含两类标签的被试；每个被试内部做 kNN holdout。"""
    per_subject_acc = []
    per_subject_auc = []
    for sid in np.unique(sids):
        m = sids == sid
        if m.sum() < 10:
            continue
        Xs, ys = X[m], labs[m]
        classes = np.unique(ys)
        if len(classes) < 2:
            continue
        if min(np.bincount(ys)) < k + 1:
            # 少数类样本太少，用最简 holdout
            sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
            accs = []
            for tr, te in sss.split(Xs, ys):
                if len(tr) <= k or len(np.unique(ys[tr])) < 2:
                    continue
                preds = knn_predict(Xs, ys, k, tr, te)
                accs.append(accuracy_score(ys[te], preds))
            if accs:
                per_subject_acc.append(np.mean(accs))
                try:
                    clf = KNeighborsClassifier(n_neighbors=min(k, len(Xs)))
                    clf.fit(Xs, ys)
                    proba = clf.predict_proba(Xs)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else 0
                    per_subject_auc.append(roc_auc_score(ys, proba))
                except Exception:
                    pass
            continue
        # 正常 holdout
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        accs = []
        aucs = []
        for tr, te in sss.split(Xs, ys):
            preds = knn_predict(Xs, ys, k, tr, te)
            accs.append(accuracy_score(ys[te], preds))
            clf = KNeighborsClassifier(n_neighbors=min(k, len(tr)))
            clf.fit(Xs[tr], ys[tr])
            if 1 in clf.classes_:
                proba = clf.predict_proba(Xs[te])[:, list(clf.classes_).index(1)]
                try:
                    aucs.append(roc_auc_score(ys[te], proba))
                except Exception:
                    pass
        per_subject_acc.append(np.mean(accs))
        per_subject_auc.extend(aucs)
    return {
        "n_subjects_both_classes": len(per_subject_acc),
        "within_subject_acc_mean": float(np.mean(per_subject_acc)) if per_subject_acc else float("nan"),
        "within_subject_acc_std": float(np.std(per_subject_acc)) if per_subject_acc else float("nan"),
        "within_subject_auc_mean": float(np.mean(per_subject_auc)) if per_subject_auc else float("nan"),
    }


def silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(labels) < 2 * len(uniq):
        return float("nan")
    # silhouette on full set may be slow/huge; subsample
    if len(labels) > 8000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(labels), 8000, replace=False)
        X, labels = X[idx], labels[idx]
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


def try_plot(X2d: np.ndarray, sids: np.ndarray, labs: np.ndarray, subj2id: dict, out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}), skip PNG; embeddings saved to npz.")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = axes[0].scatter(X2d[:, 0], X2d[:, 1], c=sids, cmap="tab20", s=4, alpha=0.6)
    axes[0].set_title("Face landmarks 2D — color by SUBJECT_ID")
    fig.colorbar(sc1, ax=axes[0], fraction=0.046)
    sc2 = axes[1].scatter(X2d[:, 0], X2d[:, 1], c=labs, cmap="coolwarm", s=4, alpha=0.6)
    axes[1].set_title("Face landmarks 2D — color by LABEL (0=alert, 1=fatigue)")
    fig.colorbar(sc2, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_png}")
    return True


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    task_mode = task_mode_from_args(cfg, args.task_mode)

    lmks, sids, labs, subj2id = collect_dataset(cfg, task_mode)
    n_subjects = len(subj2id)
    pos_ratio = float(np.mean(labs))
    majority_acc = max(pos_ratio, 1 - pos_ratio)

    # 标准化（z-score），让距离公平
    X = StandardScaler().fit_transform(lmks)

    # 抽样上限
    if args.max_samples is not None and len(labs) > args.max_samples:
        rng = np.random.default_rng(42)
        # 分层抽样保证 label 分布
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.max_samples, random_state=42)
        tr, idx = next(sss.split(X, labs))
        X, sids, labs = X[idx], sids[idx], labs[idx]
        print(f"[data] subsampled to {len(labs)} for kNN speed")

    print("\n========== 定量诊断 ==========")
    print(f"task_mode={task_mode}  N={len(labs)}  n_subjects={n_subjects}  "
          f"positive_ratio={pos_ratio:.3f}  majority_acc={majority_acc:.3f}")

    # A. 身份可预测性
    subj_baseline = 1.0 / n_subjects
    subj_acc = loo_knn_accuracy(X, sids, args.knn_k)
    subj_sil = silhouette_safe(X, sids)
    print(f"\n[A] 身份编码：")
    print(f"    kNN predict subject_id acc = {subj_acc:.3f}  (random baseline={subj_baseline:.3f})")
    print(f"    silhouette(by subject)      = {subj_sil:.3f}  (>0 means clusters exist)")

    # B. 疲劳可预测性（全局，可能被身份混淆）
    lab_acc = loo_knn_accuracy(X, labs, args.knn_k)
    lab_sil = silhouette_safe(X, labs)
    print(f"\n[B] 疲劳编码（全局，含身份混淆）：")
    print(f"    kNN predict label acc = {lab_acc:.3f}  (majority baseline={majority_acc:.3f})")
    print(f"    silhouette(by label)  = {lab_sil:.3f}")

    # C. 关键：同被试内疲劳可预测性（剥离身份后的泄漏量）
    within = within_subject_label_predictability(X, sids, labs, args.knn_k)
    print(f"\n[C] 关键泄漏检测 —— 同被试内面部特征预测疲劳（已剥离身份）：")
    print(f"    n_subjects_with_both_classes = {within['n_subjects_both_classes']}")
    print(f"    within-subject acc  = {within['within_subject_acc_mean']:.3f} "
          f"(±{within['within_subject_acc_std']:.3f}, baseline={majority_acc:.3f})")
    print(f"    within-subject AUC  = {within['within_subject_auc_mean']:.3f}  (chance=0.5)")

    # 结论判定
    print("\n========== 判定 ==========")
    identity_signal = subj_acc - subj_baseline
    fatigue_leak = within["within_subject_auc_mean"] - 0.5 if not np.isnan(within["within_subject_auc_mean"]) else float("nan")
    print(f"身份信号强度   (subj_acc - baseline)      = {identity_signal:+.3f}")
    print(f"疲劳泄漏强度   (within-subject AUC - 0.5) = {fatigue_leak:+.3f}")
    if not np.isnan(fatigue_leak) and fatigue_leak > 0.10:
        verdict = ("面部特征携带疲劳表达 → 用它当对抗目标会把疲劳一起擦掉（泄漏）。"
                   "建议改用 subject_id 分类作为对抗目标，弃用面部回归。")
    elif identity_signal > 0.15:
        verdict = ("面部特征主要编码身份、疲劳泄漏弱 → 可作对抗目标，但需先 z-score 归一化、"
                   "解耦 λ、封顶 0.1~0.3。更稳仍推荐 subject_id 分类。")
    else:
        verdict = ("面部特征既不强编码身份也不强编码疲劳 → 对抗目标信号弱，"
                   "GRL 收益有限，建议直接用 subject_id 分类。")
    print("判定：" + verdict)

    # 保存结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "task_mode": task_mode,
        "n_samples": len(labs),
        "n_subjects": n_subjects,
        "positive_ratio": pos_ratio,
        "majority_acc": majority_acc,
        "knn_k": args.knn_k,
        "subj_baseline": subj_baseline,
        "knn_subj_acc": subj_acc,
        "silhouette_subject": subj_sil,
        "knn_label_acc": lab_acc,
        "silhouette_label": lab_sil,
        "within_subject_acc_mean": within["within_subject_acc_mean"],
        "within_subject_acc_std": within["within_subject_acc_std"],
        "within_subject_auc_mean": within["within_subject_auc_mean"],
        "identity_signal": identity_signal,
        "fatigue_leak": fatigue_leak,
        "verdict": verdict,
    }
    stats_path = out_dir / f"face_diag_stats_{task_mode}.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"\n[saved] stats -> {stats_path}")

    # 降维图（可选）
    if not args.no_plot:
        emb_path = out_dir / f"face_diag_emb_{task_mode}.npz"
        pca = PCA(n_components=2).fit_transform(X)
        # t-SNE 抽样上限，避免过慢
        tsne_n = min(len(labs), 4000)
        rng = np.random.default_rng(0)
        tidx = rng.choice(len(labs), tsne_n, replace=False) if len(labs) > tsne_n else np.arange(len(labs))
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, tsne_n - 1),
                    random_state=42).fit_transform(X[tidx])
        np.savez(emb_path, pca=pca, tsne=tsne, tsne_idx=tidx,
                 sids=sids, labs=labs, subj_ids=np.array(list(subj2id.keys())))
        print(f"[saved] embeddings -> {emb_path}")
        try_plot(tsne, sids[tidx], labs[tidx], subj2id, out_dir / f"face_diag_tsne_{task_mode}.png")
        try_plot(pca, sids, labs, subj2id, out_dir / f"face_diag_pca_{task_mode}.png")


if __name__ == "__main__":
    main()
