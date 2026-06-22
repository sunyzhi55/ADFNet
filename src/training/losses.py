from __future__ import annotations

import math

import torch
from torch import nn


class ADFNetLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        landmarks: torch.Tensor,
        grl_lambda: float,
    ) -> dict[str, torch.Tensor]:
        bce_loss = self.bce(outputs["vigilance_logit"], labels)
        landmark_loss = self.mse(outputs["landmark_pred"], landmarks)
        total = bce_loss + grl_lambda * landmark_loss
        return {
            "loss": total,
            "bce": bce_loss.detach(),
            "landmark_mse": landmark_loss.detach(),
        }


def grl_lambda_schedule(epoch: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return 1.0
    progress = epoch / float(total_epochs - 1)
    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
    # return 0.0


