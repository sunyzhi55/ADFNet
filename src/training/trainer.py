from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.basic import get_optimizer, get_scheduler

from data.dataset import ADFWindowDataset, collect_alert_distances
from data.gaipat_dataset import GaipatWindowDataset, collect_gaipat_alert_distances
from models.adfnet import ADFNet
from models.distribution import GammaReference, ReferenceDistribution
from training.losses import ADFNetLoss, grl_lambda_schedule
from training.metrics import binary_metrics


class EarlyStopping:
    def __init__(
        self,
        monitor: str = "val_auc",
        mode: str = "max",
        patience: int = 8,
        min_delta: float = 1.0e-4,
        enabled: bool = True,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("early_stopping.mode must be 'max' or 'min'")
        if patience < 1:
            raise ValueError("early_stopping.patience must be >= 1")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled
        self.best: float | None = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, metrics: dict[str, float]) -> bool:
        current = metrics.get(self.monitor)
        if current is None:
            raise KeyError(f"Early stopping monitor '{self.monitor}' was not found in epoch metrics")
        if current != current:
            if self.enabled:
                self.bad_epochs += 1
                self.should_stop = self.bad_epochs >= self.patience
            return False

        if self.best is None:
            self.best = current
            self.bad_epochs = 0
            return True

        if self.mode == "max":
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta
        if improved:
            self.best = current
            self.bad_epochs = 0
            return True

        if self.enabled:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience
        return False


class BestEpochTracker:
    def __init__(
        self,
        monitor: str = "val_auc",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("result_selection.mode must be 'max' or 'min'")
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_row: dict | None = None
        self.fallback_row: dict | None = None

    def step(self, row: dict) -> bool:
        if self.fallback_row is None:
            self.fallback_row = dict(row)
        current = row.get(self.monitor)
        if current is None:
            raise KeyError(f"Result selection monitor '{self.monitor}' was not found in epoch metrics")
        if current != current:
            return False
        current = float(current)
        if self.best_value is None:
            self.best_value = current
            self.best_row = dict(row)
            return True
        if self.mode == "max":
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta
        if improved:
            self.best_value = current
            self.best_row = dict(row)
            return True
        return False

    def result(self) -> dict:
        return dict(self.best_row or self.fallback_row or {})


def result_selection_from_config(cfg: dict) -> dict:
    early_cfg = cfg["training"].get("early_stopping", {})
    selection_cfg = cfg["training"].get("result_selection", {})
    return {
        "monitor": selection_cfg.get("monitor", early_cfg.get("monitor", "val_auc")),
        "mode": selection_cfg.get("mode", early_cfg.get("mode", "max")),
        "min_delta": selection_cfg.get("min_delta", 0.0),
    }


def selected_metrics_from_row(row: dict, monitor: str, mode: str) -> dict:
    selected = {
        "best_epoch": row.get("epoch"),
        "selection_monitor": monitor,
        "selection_mode": mode,
        "selection_value": row.get(monitor),
    }
    selected.update({k: v for k, v in row.items() if k.startswith(("train_", "val_"))})
    return selected


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_ablation(cfg: dict) -> dict:
    """从 config 读取消融配置，缺失键用默认值（全部启用）补全。"""
    from models.adfnet import _parse_ablation
    return _parse_ablation(cfg.get("ablation"))


def build_gamma_reference(
    train_dataset,
    cfg: dict,
    seed: int,
    enable_soft_dtw: bool = True,
    reference_distribution: str = "gamma",
) -> ReferenceDistribution:
    if isinstance(train_dataset, GaipatWindowDataset):
        distances = collect_gaipat_alert_distances(train_dataset)
    else:
        distances = collect_alert_distances(train_dataset)
    return ReferenceDistribution.fit(
        distances,
        dist_type=reference_distribution,
        reference_sample_count=cfg["distribution"]["reference_samples"],
        eps=cfg["distribution"]["eps"],
        soft_dtw_gamma=cfg["distribution"]["soft_dtw_gamma"],
        soft_dtw_reference_samples=cfg["distribution"].get("soft_dtw_reference_samples", 64),
        seed=seed,
        enable_soft_dtw=enable_soft_dtw,
    )


def make_model(cfg: dict, n_subjects: int | None = None) -> ADFNet:
    model_cfg = dict(cfg["model"])
    if n_subjects is not None:
        model_cfg["n_subjects"] = n_subjects
    # 把 ablation 配置透传到 ADFNet
    if "ablation" in cfg:
        model_cfg["ablation"] = cfg["ablation"]
    return ADFNet(**model_cfg)


def build_subject_mapping(train_dataset) -> dict[str, int]:
    """从训练 fold 构造 subject_id -> 整数标签映射（按 id 排序，保证可复现）。"""
    subjects = sorted({sample.subject_id for sample in train_dataset.samples})
    return {sid: idx for idx, sid in enumerate(subjects)}


def grl_cfg(cfg: dict) -> dict:
    defaults = {"enabled": True, "max_lambda": 1.0, "warmup_epochs": 0,
                "loss_weight": 1.0, "schedule_slope": 10.0}
    defaults.update(cfg.get("grl", {}))
    return defaults


def run_epoch(
    epoch: int,
    model: ADFNet,
    loader: DataLoader,
    gamma_reference: GammaReference | None,
    criterion: ADFNetLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    grl_lambda: float,
    dist_feature_window: int,
    threshold: float,
    grad_clip_norm: float | None = None,
) -> tuple[dict[str, float], float]:
    is_train = optimizer is not None
    model.train(is_train)
    labels: list[float] = []
    probs: list[float] = []
    subj_preds: list[int] = []
    subj_targets: list[int] = []
    total_loss = 0.0
    total_adv_ce = 0.0
    steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [{'Train' if is_train else 'Val'}]", leave=False)
    for batch in pbar:
        use_non_blocking = device.type == "cuda"
        adf = batch["adf"].to(device, non_blocking=use_non_blocking)
        y = batch["label"].to(device, non_blocking=use_non_blocking)
        subject_ids = batch["subject_label"].to(device, non_blocking=use_non_blocking)
        dist_stats = batch.get("dist_stats")
        if dist_stats is not None and dist_stats.numel() > 0:
            dist_stats = dist_stats.to(device, non_blocking=use_non_blocking)
        else:
            dist_stats = None
        with torch.set_grad_enabled(is_train):
            outputs = model(
                adf,
                dist_stats=dist_stats,
                gamma_reference=gamma_reference,
                dist_feature_window=dist_feature_window,
                grl_lambda=grl_lambda,
            )
            losses = criterion(outputs, y, subject_ids)
            if is_train:
                if not torch.isfinite(losses["loss"]):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{losses['loss'].item():.4f}",
                    "adv": f"{float(losses['adv_ce']):.4f}",
                    "λ": f"{grl_lambda:.3f}",
                    "LR": f"{current_lr:.6f}",
                })
        if torch.isfinite(losses["loss"]):
            total_loss += float(losses["loss"].detach().cpu())
            total_adv_ce += float(losses["adv_ce"].detach().cpu())
            steps += 1
        labels.extend(y.detach().cpu().numpy().reshape(-1).tolist())
        probs.extend(torch.sigmoid(outputs["vigilance_logit"]).detach().cpu().numpy().reshape(-1).tolist())
        sid_arr = subject_ids.detach().cpu().numpy().reshape(-1)
        valid = sid_arr >= 0
        if valid.any():
            pred = outputs["subject_logit"].argmax(dim=-1).detach().cpu().numpy().reshape(-1)
            subj_preds.extend(pred[valid].tolist())
            subj_targets.extend(sid_arr[valid].tolist())
    metrics = binary_metrics(labels, probs, threshold)
    metrics["adv_ce"] = total_adv_ce / max(steps, 1)
    metrics["subject_acc"] = (
        float((np.asarray(subj_preds) == np.asarray(subj_targets)).mean())
        if subj_preds else float("nan")
    )
    return metrics, total_loss / max(steps, 1)


def train_fold(
    cfg: dict,
    train_dataset,
    val_dataset,
    fold_name: str = "default",
) -> dict[str, float]:
    output_dir = Path(cfg["training"]["output_dir"]) / fold_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg["training"]["device"])

    abl = get_ablation(cfg)

    # ── Gamma 分布参考（仅当分布分支启用时构建） ──
    gamma_reference: GammaReference | None = None
    dist_stats_mean: np.ndarray | None = None
    dist_stats_std: np.ndarray | None = None

    if abl["enable_gamma"]:
        gamma_reference = build_gamma_reference(
            train_dataset, cfg, cfg["seed"],
            enable_soft_dtw=abl["enable_soft_dtw"],
            reference_distribution=abl.get("reference_distribution", "gamma"),
        )
        dist_stats_mean, dist_stats_std = train_dataset.attach_distribution_stats(
            gamma_reference,
            cfg["distribution"]["feature_window"],
        )
        val_dataset.attach_distribution_stats(
            gamma_reference,
            cfg["distribution"]["feature_window"],
            dist_stats_mean,
            dist_stats_std,
        )

    # ── GRL 对抗目标映射（仅当 GRL 启用时构建） ──
    subject_mapping: dict[str, int] | None = None
    if abl["enable_grl"]:
        subject_mapping = build_subject_mapping(train_dataset)
        train_dataset.attach_subject_labels(subject_mapping)
        val_dataset.attach_subject_labels(subject_mapping)
        cfg["model"]["n_subjects"] = len(subject_mapping)
    else:
        # GRL 禁用时对抗项不生效，subject_mapping 保持 None
        cfg["model"]["n_subjects"] = 1  # 占位，模型不会创建判别器

    grl = grl_cfg(cfg)
    # 消融：GRL 禁用时强制 λ=0 且 loss_weight=0
    if not abl["enable_grl"]:
        grl["enabled"] = False
        grl["loss_weight"] = 0.0

    n_subjects = len(subject_mapping) if subject_mapping else 1
    model = make_model(cfg, n_subjects=n_subjects).to(device)
    criterion = ADFNetLoss(loss_weight=grl["loss_weight"])
    optimizer = get_optimizer(
        cfg["training"]["optimizer_name"],
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"].get("pin_memory", False) and device.type == "cuda",
        persistent_workers=cfg["training"].get("persistent_workers", False)
        and cfg["training"]["num_workers"] > 0,
    )
    scheduler = get_scheduler(optimizer, cfg, train_loader)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"].get("pin_memory", False) and device.type == "cuda",
        persistent_workers=cfg["training"].get("persistent_workers", False)
        and cfg["training"]["num_workers"] > 0,
    )
    history: list[dict[str, float]] = []
    early_stopping = EarlyStopping(**cfg["training"].get("early_stopping", {}))
    best_epoch = BestEpochTracker(**result_selection_from_config(cfg))
    for epoch in range(cfg["training"]["epochs"]):
        if grl["enabled"]:
            lambd = grl_lambda_schedule(
                epoch,
                cfg["training"]["epochs"],
                max_lambda=grl["max_lambda"],
                warmup_epochs=grl["warmup_epochs"],
                slope=grl["schedule_slope"],
            )
        else:
            lambd = 0.0
        train_metrics, train_loss = run_epoch(
            epoch,
            model,
            train_loader,
            gamma_reference,
            criterion,
            device,
            optimizer,
            scheduler,
            lambd,
            cfg["distribution"]["feature_window"],
            cfg["training"]["threshold"],
            cfg["training"].get("grad_clip_norm"),
        )
        val_metrics, val_loss = run_epoch(
            epoch,
            model,
            val_loader,
            gamma_reference,
            criterion,
            device,
            None,
            None,
            lambd,
            cfg["distribution"]["feature_window"],
            cfg["training"]["threshold"],
            None,
        )
        row = {
            "epoch": epoch + 1,
            "grl_lambda": lambd,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        monitor_improved = early_stopping.step(row)
        row["result_selection_monitor"] = best_epoch.monitor
        result_improved = best_epoch.step(row)
        row["result_selection_best"] = best_epoch.best_value
        row["result_selection_improved"] = result_improved
        if result_improved:
            checkpoint: dict = {
                "model": model.state_dict(),
                "config": cfg,
                "subject_mapping": subject_mapping,
            }
            if gamma_reference is not None:
                checkpoint["gamma_reference"] = gamma_reference.to_checkpoint()
            if dist_stats_mean is not None and dist_stats_std is not None:
                checkpoint["dist_stats_normalizer"] = {
                    "mean": dist_stats_mean,
                    "std": dist_stats_std,
                }
            torch.save(checkpoint, output_dir / "best.pt")
        row["early_stop_monitor"] = early_stopping.monitor
        row["early_stop_best"] = early_stopping.best
        row["early_stop_bad_epochs"] = early_stopping.bad_epochs
        row["early_stop_improved"] = monitor_improved
        row["early_stopped"] = early_stopping.should_stop
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if early_stopping.should_stop:
            break
    best = best_epoch.result()
    selected = selected_metrics_from_row(best, best_epoch.monitor, best_epoch.mode)
    pd.DataFrame([selected]).to_csv(output_dir / "final_metrics.csv", index=False)
    return selected


@torch.no_grad()
def evaluate_checkpoint(
    cfg: dict,
    dataset,
    checkpoint_path: str | Path,
) -> dict[str, float]:
    device = resolve_device(cfg["training"]["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_cfg = checkpoint.get("config", cfg)
    model = make_model(saved_cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    abl = get_ablation(saved_cfg)

    # 重建分布参考（仅当分布分支启用时）
    gamma_reference: ReferenceDistribution | None = None
    if abl["enable_gamma"]:
        gamma_state = checkpoint["gamma_reference"]
        gamma_reference = ReferenceDistribution.from_checkpoint(gamma_state, cfg)
        normalizer = checkpoint.get("dist_stats_normalizer", {})
        stats_mean = normalizer.get("mean")
        stats_std = normalizer.get("std")
        dataset.attach_distribution_stats(
            gamma_reference,
            cfg["distribution"]["feature_window"],
            None if stats_mean is None else stats_mean,
            None if stats_std is None else stats_std,
        )

    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    labels: list[float] = []
    probs: list[float] = []
    model.eval()
    for batch in tqdm(loader, leave=False):
        dist_stats = batch.get("dist_stats")
        if dist_stats is not None and dist_stats.numel() > 0:
            dist_stats = dist_stats.to(device, non_blocking=device.type == "cuda")
        else:
            dist_stats = None
        outputs = model(
            batch["adf"].to(device, non_blocking=device.type == "cuda"),
            dist_stats=dist_stats,
            gamma_reference=gamma_reference,
            dist_feature_window=cfg["distribution"]["feature_window"],
            grl_lambda=0.0,
        )
        labels.extend(batch["label"].numpy().reshape(-1).tolist())
        probs.extend(torch.sigmoid(outputs["vigilance_logit"]).cpu().numpy().reshape(-1).tolist())
    return binary_metrics(labels, probs, cfg["training"]["threshold"])
