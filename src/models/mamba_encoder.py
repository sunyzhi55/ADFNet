from __future__ import annotations

import math

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


class LSTMTemporalEncoder(nn.Module):
    """LSTM 时序编码器——与 MambaTemporalEncoder 接口一致，用于消融替换实验。"""

    def __init__(
        self,
        input_dim: int = 3,
        mamba_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, mamba_dim)
        self.lstm = nn.LSTM(
            input_size=mamba_dim,
            hidden_size=mamba_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(mamba_dim)

    def forward(self, adf: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(adf)
        hidden, _ = self.lstm(hidden)
        hidden = self.output_norm(hidden)
        return hidden.mean(dim=1)


class _SinusoidalPositionalEncoding(nn.Module):
    """标准正弦位置编码，用于 Transformer 时序编码器。"""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerTemporalEncoder(nn.Module):
    """Transformer 时序编码器——与 MambaTemporalEncoder 接口一致，用于消融替换实验。"""

    def __init__(
        self,
        input_dim: int = 3,
        mamba_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, mamba_dim)
        self.pos_encoder = _SinusoidalPositionalEncoding(mamba_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mamba_dim,
            nhead=4,
            dim_feedforward=mamba_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(mamba_dim)

    def forward(self, adf: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(adf)
        hidden = self.pos_encoder(hidden)
        hidden = self.transformer_encoder(hidden)
        hidden = self.output_norm(hidden)
        return hidden.mean(dim=1)
