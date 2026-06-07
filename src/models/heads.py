from __future__ import annotations

from torch import nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class VigilanceHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, features):
        return self.net(features)


class LandmarkHead(nn.Module):
    def __init__(self, in_dim: int, landmark_dim: int = 70, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, landmark_dim),
        )

    def forward(self, features):
        return self.net(features)
