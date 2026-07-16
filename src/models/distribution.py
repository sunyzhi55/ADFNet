from __future__ import annotations

import numpy as np
import torch
from scipy.stats import gamma as scipy_gamma, lognorm as scipy_lognorm
from scipy.stats import norm as scipy_norm
from scipy.stats import wasserstein_distance
from torch import nn


class ReferenceDistribution:
    """警觉基准分布：支持 Gamma / Gaussian / LogNormal / KDE 四种拟合方式。

    统一接口：fit() 拟合 + features()/window_features() 计算分布对齐特征。
    Gamma 分布为默认选项；Gaussian、LogNormal 与 KDE 用于消融替换实验。
    """

    def __init__(
        self,
        dist_type: str,
        params: dict,
        reference_samples: np.ndarray,
        data_points: np.ndarray | None = None,
        eps: float = 1.0e-6,
        soft_dtw_gamma: float = 1.0,
        soft_dtw_reference_samples: int | None = 64,
        enable_soft_dtw: bool = True,
    ) -> None:
        if dist_type not in ("gamma", "gaussian", "lognormal", "kde"):
            raise ValueError(f"dist_type must be gamma/gaussian/lognormal/kde, got {dist_type}")
        self.dist_type = dist_type
        self.params = params
        self.reference_samples = np.asarray(reference_samples, dtype=np.float32)
        self.data_points = (
            np.asarray(data_points, dtype=np.float32) if data_points is not None else None
        )
        self.eps = eps
        self.soft_dtw_gamma = soft_dtw_gamma
        self.soft_dtw_reference_samples = soft_dtw_reference_samples
        self.enable_soft_dtw = enable_soft_dtw
        self.soft_dtw_samples = self._make_soft_dtw_samples(self.reference_samples)

    # ── 工厂方法 ────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        distances: np.ndarray,
        dist_type: str = "gamma",
        reference_sample_count: int = 512,
        eps: float = 1.0e-6,
        soft_dtw_gamma: float = 1.0,
        soft_dtw_reference_samples: int | None = 64,
        seed: int = 42,
        enable_soft_dtw: bool = True,
    ) -> "ReferenceDistribution":
        clean = np.asarray(distances, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        clean = np.clip(clean, eps, None)
        rng = np.random.default_rng(seed)

        if dist_type == "gamma":
            shape, loc, scale = scipy_gamma.fit(clean, floc=0.0)
            refs = scipy_gamma.rvs(
                shape, loc=loc, scale=scale,
                size=reference_sample_count, random_state=rng,
            )
            params = {"shape": float(shape), "loc": float(loc), "scale": float(scale)}

        elif dist_type == "gaussian":
            mu = float(np.mean(clean))
            sigma = float(np.std(clean))
            sigma = max(sigma, eps)
            refs = rng.normal(mu, sigma, size=reference_sample_count)
            params = {"mu": mu, "sigma": sigma}

        elif dist_type == "lognormal":
            # 对数正态分布: log(X) ~ N(mu, sigma^2)
            # scipy 参数化: lognorm(s=sigma, loc=0, scale=exp(mu))
            # floc=0 固定位置参数, 与 Gamma (floc=0) 保持一致
            s, loc, scale = scipy_lognorm.fit(clean, floc=0.0)
            refs = scipy_lognorm.rvs(
                s, loc=loc, scale=scale,
                size=reference_sample_count, random_state=rng,
            )
            params = {"lognorm_s": float(s), "lognorm_loc": float(loc), "lognorm_scale": float(scale)}

        elif dist_type == "kde":
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(clean)
            refs = kde.resample(reference_sample_count, seed=rng).flatten()
            params = {}  # KDE 为非参数方法，不存储解析参数

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

        return cls(
            dist_type=dist_type,
            params=params,
            reference_samples=refs.astype(np.float32),
            data_points=clean.astype(np.float32) if dist_type == "kde" else None,
            eps=eps,
            soft_dtw_gamma=soft_dtw_gamma,
            soft_dtw_reference_samples=soft_dtw_reference_samples,
            enable_soft_dtw=enable_soft_dtw,
        )

    # ── 属性 ────────────────────────────────────────────────

    @property
    def feat_dim(self) -> int:
        return 3 if self.enable_soft_dtw else 2

    # ── 向后兼容属性（旧代码访问 .shape/.loc/.scale） ──────

    @property
    def shape(self) -> float:
        return self.params.get("shape", 0.0)

    @property
    def loc(self) -> float:
        return self.params.get("loc", 0.0)

    @property
    def scale(self) -> float:
        return self.params.get("scale", 0.0)

    # ── 特征计算 ────────────────────────────────────────────

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

    def _log_pdf(self, clean: np.ndarray) -> np.ndarray:
        """根据 dist_type 计算对数概率密度。"""
        if self.dist_type == "gamma":
            return scipy_gamma.logpdf(
                clean, self.params["shape"],
                loc=self.params["loc"], scale=self.params["scale"],
            )
        elif self.dist_type == "gaussian":
            return scipy_norm.logpdf(clean, loc=self.params["mu"], scale=self.params["sigma"])
        elif self.dist_type == "lognormal":
            return scipy_lognorm.logpdf(
                clean,
                self.params["lognorm_s"],
                loc=self.params["lognorm_loc"],
                scale=self.params["lognorm_scale"],
            )
        elif self.dist_type == "kde":
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(self.data_points)
            return kde.logpdf(clean)
        raise ValueError(f"Unknown dist_type: {self.dist_type}")

    def window_features(self, window: np.ndarray) -> list[float]:
        clean = np.asarray(window, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            return [0.0, 0.0, 0.0] if self.enable_soft_dtw else [0.0, 0.0]
        clean = np.clip(clean, self.eps, None)

        # 1) 平均对数似然
        log_prob = self._log_pdf(clean)
        log_prob = np.nan_to_num(log_prob, nan=-1.0e6, posinf=1.0e6, neginf=-1.0e6)
        mean_log_likelihood = float(np.mean(log_prob))

        # 2) Wasserstein 距离
        wasserstein = float(wasserstein_distance(clean, self.reference_samples))

        # 3) Soft-DTW 距离
        values: list[float] = [mean_log_likelihood, wasserstein]
        if self.enable_soft_dtw:
            soft_dtw = soft_dtw_distance(
                clean, self.soft_dtw_samples, gamma=self.soft_dtw_gamma,
            )
            values.append(soft_dtw)

        values_arr = np.nan_to_num(
            np.asarray(values, dtype=np.float64),
            nan=0.0, posinf=1.0e6, neginf=-1.0e6,
        )
        return values_arr.astype(np.float32).tolist()

    def _make_soft_dtw_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if self.soft_dtw_reference_samples is None or self.soft_dtw_reference_samples <= 0:
            return samples
        if samples.size <= self.soft_dtw_reference_samples:
            return samples
        indices = np.linspace(
            0, samples.size - 1, self.soft_dtw_reference_samples,
        ).astype(np.int64)
        return samples[indices]

    # ── 序列化 ──────────────────────────────────────────────

    def to_checkpoint(self) -> dict:
        state: dict = {
            "dist_type": self.dist_type,
            "params": self.params,
            "reference_samples": self.reference_samples,
            "enable_soft_dtw": self.enable_soft_dtw,
        }
        if self.data_points is not None:
            state["data_points"] = self.data_points
        return state

    @classmethod
    def from_checkpoint(cls, state: dict, cfg: dict) -> "ReferenceDistribution":
        dist_type = state.get("dist_type", "gamma")
        params = state.get("params")
        if params is None:
            # 旧格式：shape/loc/scale 在顶层
            params = {
                "shape": state.get("shape", 0.0),
                "loc": state.get("loc", 0.0),
                "scale": state.get("scale", 0.0),
            }
        return cls(
            dist_type=dist_type,
            params=params,
            reference_samples=state["reference_samples"],
            data_points=state.get("data_points"),
            eps=cfg["distribution"]["eps"],
            soft_dtw_gamma=cfg["distribution"]["soft_dtw_gamma"],
            soft_dtw_reference_samples=cfg["distribution"].get("soft_dtw_reference_samples", 64),
            enable_soft_dtw=state.get("enable_soft_dtw", True),
        )


# 向后兼容别名
GammaReference = ReferenceDistribution


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
