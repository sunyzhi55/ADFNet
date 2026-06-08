from __future__ import annotations

import torch
from torch import nn

from models.distribution import DistributionBranch, GammaReference
from models.grl import GradientReverseLayer
from models.heads import LandmarkHead, VigilanceHead
from models.mamba_encoder import MambaTemporalEncoder


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
        landmark_dim: int = 70,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if fusion_dim != mamba_dim + dist_out_dim:
            raise ValueError("fusion_dim must equal mamba_dim + dist_out_dim")
        self.distribution_branch = DistributionBranch(
            input_dim=dist_feat_dim,
            hidden_dim=dist_hidden_dim,
            out_dim=dist_out_dim,
            dropout=dropout,
        )
        self.temporal_encoder = MambaTemporalEncoder(
            input_dim=input_dim,
            mamba_dim=mamba_dim,
            layers=mamba_layers,
            dropout=dropout,
        )
        self.grl = GradientReverseLayer()
        self.vigilance_head = VigilanceHead(fusion_dim, dropout)
        self.landmark_head = LandmarkHead(fusion_dim, landmark_dim, dropout)

    def forward(
        self,
        adf: torch.Tensor,
        dist_stats: torch.Tensor | None = None,
        gamma_reference: GammaReference | None = None,
        dist_feature_window: int | None = None,
        grl_lambda: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if dist_stats is None:
            if gamma_reference is None:
                raise ValueError("dist_stats and gamma_reference cannot both be None")
            dist_stats = gamma_reference.features(adf, feature_window=dist_feature_window)
        dist_feature = self.distribution_branch(dist_stats)
        temp_feature = self.temporal_encoder(adf)
        fusion_feature = torch.cat([temp_feature, dist_feature], dim=-1)
        vigilance_logit = self.vigilance_head(fusion_feature)
        adv_feature = self.grl(fusion_feature, grl_lambda)
        landmark_pred = self.landmark_head(adv_feature)
        return {
            "vigilance_logit": vigilance_logit,
            "landmark_pred": landmark_pred,
            "fusion_feature": fusion_feature,
            "dist_feature": dist_feature,
            "temp_feature": temp_feature,
        }
