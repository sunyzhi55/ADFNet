from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ADFWindowDataset, collect_alert_distances
from models.adfnet import ADFNet
from models.distribution import GammaReference
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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_gamma_reference(train_dataset: ADFWindowDataset, cfg: dict, seed: int) -> GammaReference:
    distances = collect_alert_distances(train_dataset)
    return GammaReference.fit(
        distances,
        reference_sample_count=cfg["distribution"]["reference_samples"],
        eps=cfg["distribution"]["eps"],
        soft_dtw_gamma=cfg["distribution"]["soft_dtw_gamma"],
        soft_dtw_reference_samples=cfg["distribution"].get("soft_dtw_reference_samples", 64),
        seed=seed,
    )


def make_model(cfg: dict) -> ADFNet:
    return ADFNet(**cfg["model"])


def run_epoch(
    model: ADFNet,
    loader: DataLoader,
    gamma_reference: GammaReference,
    criterion: ADFNetLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grl_lambda: float,
    dist_feature_window: int,
    threshold: float,
    grad_clip_norm: float | None = None,
) -> tuple[dict[str, float], float]:
    is_train = optimizer is not None
    model.train(is_train)
    labels: list[float] = []
    probs: list[float] = []
    total_loss = 0.0
    steps = 0
    for batch in tqdm(loader, leave=False):
        use_non_blocking = device.type == "cuda"
        adf = batch["adf"].to(device, non_blocking=use_non_blocking)
        y = batch["label"].to(device, non_blocking=use_non_blocking)
        landmarks = batch["landmarks"].to(device, non_blocking=use_non_blocking)
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
            losses = criterion(outputs, y, landmarks, grl_lambda)
            if is_train:
                if not torch.isfinite(losses["loss"]):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
        if torch.isfinite(losses["loss"]):
            total_loss += float(losses["loss"].detach().cpu())
            steps += 1
        labels.extend(y.detach().cpu().numpy().reshape(-1).tolist())
        probs.extend(torch.sigmoid(outputs["vigilance_logit"]).detach().cpu().numpy().reshape(-1).tolist())
    metrics = binary_metrics(labels, probs, threshold)
    return metrics, total_loss / max(steps, 1)


def train_fold(
    cfg: dict,
    train_dataset: ADFWindowDataset,
    val_dataset: ADFWindowDataset,
    fold_name: str = "default",
) -> dict[str, float]:
    output_dir = Path(cfg["training"]["output_dir"]) / fold_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg["training"]["device"])
    gamma_reference = build_gamma_reference(train_dataset, cfg, cfg["seed"])
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
    model = make_model(cfg).to(device)
    criterion = ADFNetLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
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
    for epoch in range(cfg["training"]["epochs"]):
        lambd = grl_lambda_schedule(epoch, cfg["training"]["epochs"])
        train_metrics, train_loss = run_epoch(
            model,
            train_loader,
            gamma_reference,
            criterion,
            device,
            optimizer,
            lambd,
            cfg["distribution"]["feature_window"],
            cfg["training"]["threshold"],
            cfg["training"].get("grad_clip_norm"),
        )
        val_metrics, val_loss = run_epoch(
            model,
            val_loader,
            gamma_reference,
            criterion,
            device,
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
        history.append(row)
        monitor_improved = early_stopping.step(row)
        if monitor_improved:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "gamma_reference": {
                        "shape": gamma_reference.shape,
                        "loc": gamma_reference.loc,
                        "scale": gamma_reference.scale,
                        "reference_samples": gamma_reference.reference_samples,
                    },
                    "dist_stats_normalizer": {
                        "mean": dist_stats_mean,
                        "std": dist_stats_std,
                    },
                },
                output_dir / "best.pt",
            )
        row["early_stop_monitor"] = early_stopping.monitor
        row["early_stop_best"] = early_stopping.best
        row["early_stop_bad_epochs"] = early_stopping.bad_epochs
        row["early_stop_improved"] = monitor_improved
        row["early_stopped"] = early_stopping.should_stop
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if early_stopping.should_stop:
            break
    final = history[-1] if history else {}
    return {k.replace("val_", ""): v for k, v in final.items() if k.startswith("val_")}


@torch.no_grad()
def evaluate_checkpoint(
    cfg: dict,
    dataset: ADFWindowDataset,
    checkpoint_path: str | Path,
) -> dict[str, float]:
    device = resolve_device(cfg["training"]["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = make_model(checkpoint.get("config", cfg)).to(device)
    model.load_state_dict(checkpoint["model"])
    gamma_state = checkpoint["gamma_reference"]
    gamma_reference = GammaReference(
        gamma_state["shape"],
        gamma_state["loc"],
        gamma_state["scale"],
        gamma_state["reference_samples"],
        eps=cfg["distribution"]["eps"],
        soft_dtw_gamma=cfg["distribution"]["soft_dtw_gamma"],
        soft_dtw_reference_samples=cfg["distribution"].get("soft_dtw_reference_samples", 64),
    )
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
        outputs = model(
            batch["adf"].to(device, non_blocking=device.type == "cuda"),
            dist_stats=batch["dist_stats"].to(device, non_blocking=device.type == "cuda"),
            gamma_reference=gamma_reference,
            dist_feature_window=cfg["distribution"]["feature_window"],
            grl_lambda=0.0,
        )
        labels.extend(batch["label"].numpy().reshape(-1).tolist())
        probs.extend(torch.sigmoid(outputs["vigilance_logit"]).cpu().numpy().reshape(-1).tolist())
    return binary_metrics(labels, probs, cfg["training"]["threshold"])
