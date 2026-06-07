from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.mark.skipif(importlib.util.find_spec("mamba_ssm") is None, reason="mamba-ssm 未安装")
def test_adfnet_forward_shapes_and_backward() -> None:
    from models.adfnet import ADFNet
    from training.losses import ADFNetLoss

    model = ADFNet(mamba_dim=16, mamba_layers=1, fusion_dim=32, dist_out_dim=16)
    adf = torch.randn(2, 8, 3)
    dist_stats = torch.randn(2, 3)
    labels = torch.tensor([[0.0], [1.0]])
    landmarks = torch.randn(2, 70)

    outputs = model(adf, dist_stats=dist_stats, grl_lambda=0.5)
    assert outputs["vigilance_logit"].shape == (2, 1)
    assert outputs["landmark_pred"].shape == (2, 70)
    assert outputs["dist_feature"].shape == (2, 16)
    assert outputs["temp_feature"].shape == (2, 16)
    assert outputs["fusion_feature"].shape == (2, 32)

    losses = ADFNetLoss()(outputs, labels, landmarks, grl_lambda=0.5)
    losses["loss"].backward()
