from __future__ import annotations

import torch
from torch import nn


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


class GradientReverseLayer(nn.Module):
    def forward(self, inputs: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return _GradientReverseFn.apply(inputs, float(lambd))
