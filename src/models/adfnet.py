from __future__ import annotations

import torch
from torch import nn

from models.distribution import DistributionBranch, GammaReference, ReferenceDistribution
from models.grl import GradientReverseLayer
from models.heads import SubjectDiscriminator, VigilanceHead
from models.mamba_encoder import (
    LSTMTemporalEncoder,
    MambaTemporalEncoder,
    TransformerTemporalEncoder,
)


def _parse_ablation(ablation: dict | None) -> dict:
    """将 ablation 配置补全为带默认值的标准字典。"""
    defaults = {
        "enable_gamma": True,
        "enable_grl": True,
        "enable_diff": True,
        "enable_sliding_mean": True,
        "enable_soft_dtw": True,
        "enable_mamba": True,
        "temporal_encoder": "mamba",  # mamba | lstm | transformer
        "reference_distribution": "gamma",  # gamma | gaussian | lognormal | weibull | rayleigh | kde
    }
    if ablation:
        defaults.update(ablation)
    return defaults


class ADFNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        dist_feat_dim: int = 3,
        dist_hidden_dim: int = 32,
        dist_out_dim: int = 16,
        mamba_dim: int = 128,
        mamba_layers: int = 2,
        fusion_dim: int = 144,
        n_subjects: int = 20,
        dropout: float = 0.2,
        ablation: dict | None = None,
    ) -> None:
        super().__init__()
        abl = _parse_ablation(ablation)
        self.ablation = abl

        # ── 分布分支（Gamma 分布对齐流） ──
        if abl["enable_gamma"]:
            actual_dist_feat_dim = dist_feat_dim if abl["enable_soft_dtw"] else max(dist_feat_dim - 1, 1)
            self.distribution_branch = DistributionBranch(
                input_dim=actual_dist_feat_dim,
                hidden_dim=dist_hidden_dim,
                out_dim=dist_out_dim,
                dropout=dropout,
            )
            active_dist_dim = dist_out_dim
        else:
            self.distribution_branch = None
            active_dist_dim = 0

        # ── 时序分支（Mamba-MLA / LSTM / Transformer） ──
        if abl["enable_mamba"]:
            enc_type = abl.get("temporal_encoder", "mamba")
            if enc_type == "lstm":
                self.temporal_encoder = LSTMTemporalEncoder(
                    input_dim=input_dim,
                    mamba_dim=mamba_dim,
                    layers=mamba_layers,
                    dropout=dropout,
                )
            elif enc_type == "transformer":
                self.temporal_encoder = TransformerTemporalEncoder(
                    input_dim=input_dim,
                    mamba_dim=mamba_dim,
                    layers=mamba_layers,
                    dropout=dropout,
                )
            else:
                self.temporal_encoder = MambaTemporalEncoder(
                    input_dim=input_dim,
                    mamba_dim=mamba_dim,
                    layers=mamba_layers,
                    dropout=dropout,
                )
            active_temp_dim = mamba_dim
        else:
            self.temporal_encoder = None
            active_temp_dim = 0

        # ── 融合维度由活跃分支动态决定 ──
        actual_fusion_dim = active_temp_dim + active_dist_dim
        if actual_fusion_dim == 0:
            raise ValueError("At least one branch (temporal or distribution) must be enabled")
        self._actual_fusion_dim = actual_fusion_dim

        # ── GRL + 对抗判别器 ──
        self.grl = GradientReverseLayer()
        self.vigilance_head = VigilanceHead(actual_fusion_dim, dropout)
        if abl["enable_grl"]:
            self.subject_discriminator = SubjectDiscriminator(actual_fusion_dim, n_subjects, dropout)
        else:
            self.subject_discriminator = None

    def forward(
        self,
        adf: torch.Tensor,
        dist_stats: torch.Tensor | None = None,
        gamma_reference: GammaReference | None = None,
        dist_feature_window: int | None = None,
        grl_lambda: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        # ── 输入通道消融：被禁用的特征通道置零 ──
        if not self.ablation["enable_diff"] and adf.shape[-1] >= 2:
            adf = adf.clone()
            adf[..., 1] = 0.0
        if not self.ablation["enable_sliding_mean"] and adf.shape[-1] >= 3:
            adf = adf.clone()
            adf[..., 2] = 0.0

        parts: list[torch.Tensor] = []

        # ── 时序分支 ──
        if self.temporal_encoder is not None:
            temp_feature = self.temporal_encoder(adf)
            parts.append(temp_feature)
        else:
            temp_feature = None

        # ── 分布分支 ──
        if self.distribution_branch is not None:
            if dist_stats is None:
                if gamma_reference is None:
                    raise ValueError("dist_stats and gamma_reference cannot both be None")
                dist_stats = gamma_reference.features(adf, feature_window=dist_feature_window)
            dist_feature = self.distribution_branch(dist_stats)
            parts.append(dist_feature)
        else:
            dist_feature = None

        fusion_feature = torch.cat(parts, dim=-1)
        vigilance_logit = self.vigilance_head(fusion_feature)

        # ── GRL 对抗头 ──
        if self.subject_discriminator is not None:
            adv_feature = self.grl(fusion_feature, grl_lambda)
            subject_logit = self.subject_discriminator(adv_feature)
        else:
            B = fusion_feature.shape[0]
            subject_logit = fusion_feature.new_zeros(B, 1)
            adv_feature = fusion_feature

        return {
            "vigilance_logit": vigilance_logit,
            "subject_logit": subject_logit,
            "fusion_feature": fusion_feature,
            "dist_feature": dist_feature,
            "temp_feature": temp_feature,
        }
