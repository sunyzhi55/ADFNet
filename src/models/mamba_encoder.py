from __future__ import annotations

import torch
from torch import nn


class MambaTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        mamba_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "ADFNet 的时序分支要求安装 mamba-ssm。请先运行: pip install mamba-ssm"
            ) from exc

        self.input_proj = nn.Linear(input_dim, mamba_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(mamba_dim),
                    Mamba(d_model=mamba_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(mamba_dim)

    def forward(self, adf: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(adf)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        hidden = self.output_norm(hidden)
        return hidden.mean(dim=1)
