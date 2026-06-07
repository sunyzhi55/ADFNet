from __future__ import annotations

import numpy as np
import torch
from scipy.stats import gamma as scipy_gamma
from scipy.stats import wasserstein_distance
from torch import nn


class GammaReference:
    def __init__(
        self,
        shape: float,
        loc: float,
        scale: float,
        reference_samples: np.ndarray,
        eps: float = 1.0e-6,
        soft_dtw_gamma: float = 1.0,
        soft_dtw_reference_samples: int | None = 64,
    ) -> None:
        self.shape = float(shape)
        self.loc = float(loc)
        self.scale = float(scale)
        self.reference_samples = np.asarray(reference_samples, dtype=np.float32)
        self.eps = eps
        self.soft_dtw_gamma = soft_dtw_gamma
        self.soft_dtw_reference_samples = soft_dtw_reference_samples
        self.soft_dtw_samples = self._make_soft_dtw_samples(self.reference_samples)

    @classmethod
    def fit(
        cls,
        distances: np.ndarray,
        reference_sample_count: int = 512,
        eps: float = 1.0e-6,
        soft_dtw_gamma: float = 1.0,
        soft_dtw_reference_samples: int | None = 64,
        seed: int = 42,
    ) -> "GammaReference":
        clean = np.asarray(distances, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        clean = np.clip(clean, eps, None)
        shape, loc, scale = scipy_gamma.fit(clean, floc=0.0)
        rng = np.random.default_rng(seed)
        refs = scipy_gamma.rvs(shape, loc=loc, scale=scale, size=reference_sample_count, random_state=rng)
        return cls(
            shape,
            loc,
            scale,
            refs.astype(np.float32),
            eps,
            soft_dtw_gamma,
            soft_dtw_reference_samples,
        )

    def features(self, adf: torch.Tensor, feature_window: int | None = None) -> torch.Tensor:
        outputs = self.features_numpy(adf.detach().cpu().numpy(), feature_window)
        return torch.tensor(outputs, dtype=adf.dtype, device=adf.device)

    def features_numpy(self, adf: np.ndarray, feature_window: int | None = None) -> np.ndarray:
        distances = np.asarray(adf, dtype=np.float32)[..., 0]
        if feature_window is not None:
            distances = distances[:, -feature_window:]
        outputs: list[list[float]] = []
        for window in distances:
            outputs.append(self.window_features(window))
        return np.asarray(outputs, dtype=np.float32)

    def window_features(self, window: np.ndarray) -> list[float]:
        clean = np.asarray(window, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            return [0.0, 0.0, 0.0]
        clean = np.clip(clean, self.eps, None)
        log_prob = scipy_gamma.logpdf(clean, self.shape, loc=self.loc, scale=self.scale)
        log_prob = np.nan_to_num(log_prob, nan=-1.0e6, posinf=1.0e6, neginf=-1.0e6)
        mean_log_likelihood = float(np.mean(log_prob))
        wasserstein = float(wasserstein_distance(clean, self.reference_samples))
        soft_dtw = soft_dtw_distance(clean, self.soft_dtw_samples, gamma=self.soft_dtw_gamma)
        values = np.nan_to_num(
            np.asarray([mean_log_likelihood, wasserstein, soft_dtw], dtype=np.float64),
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
        )
        return values.astype(np.float32).tolist()

    def _make_soft_dtw_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if self.soft_dtw_reference_samples is None or self.soft_dtw_reference_samples <= 0:
            return samples
        if samples.size <= self.soft_dtw_reference_samples:
            return samples
        indices = np.linspace(0, samples.size - 1, self.soft_dtw_reference_samples).astype(np.int64)
        return samples[indices]


def soft_dtw_distance(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return 0.0
    cost = (x[:, None] - y[None, :]) ** 2
    r = np.full((len(x) + 1, len(y) + 1), np.inf, dtype=np.float64)
    r[0, 0] = 0.0
    gamma = max(float(gamma), 1.0e-6)
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            vals = np.array([r[i - 1, j], r[i, j - 1], r[i - 1, j - 1]])
            softmin = stable_softmin(vals, gamma)
            r[i, j] = cost[i - 1, j - 1] + softmin
    value = r[-1, -1] / max(len(x), len(y))
    return float(np.nan_to_num(value, nan=0.0, posinf=1.0e6, neginf=-1.0e6))


def stable_softmin(vals: np.ndarray, gamma: float) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.inf
    scaled = -finite / gamma
    max_scaled = np.max(scaled)
    return float(-gamma * (max_scaled + np.log(np.exp(scaled - max_scaled).sum())))


class DistributionBranch(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        out_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
