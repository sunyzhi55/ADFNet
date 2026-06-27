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

    model = ADFNet(mamba_dim=16, mamba_layers=1, fusion_dim=32, dist_out_dim=16, n_subjects=5)
    adf = torch.randn(2, 8, 3)
    dist_stats = torch.randn(2, 3)
    labels = torch.tensor([[0.0], [1.0]])
    subject_ids = torch.tensor([0, 3], dtype=torch.long)

    outputs = model(adf, dist_stats=dist_stats, grl_lambda=0.5)
    assert outputs["vigilance_logit"].shape == (2, 1)
    assert outputs["subject_logit"].shape == (2, 5)
    assert outputs["dist_feature"].shape == (2, 16)
    assert outputs["temp_feature"].shape == (2, 16)
    assert outputs["fusion_feature"].shape == (2, 32)

    losses = ADFNetLoss()(outputs, labels, subject_ids)
    assert "adv_ce" in losses
    losses["loss"].backward()


@pytest.mark.skipif(importlib.util.find_spec("mamba_ssm") is None, reason="mamba-ssm 未安装")
def test_loss_ignores_unknown_subjects() -> None:
    """验证集全部为未知被试（subject_id=-1）时，对抗项不产生 nan 且可反传。"""
    from models.adfnet import ADFNet
    from training.losses import ADFNetLoss

    model = ADFNet(mamba_dim=16, mamba_layers=1, fusion_dim=32, dist_out_dim=16, n_subjects=5)
    adf = torch.randn(2, 8, 3)
    labels = torch.tensor([[0.0], [1.0]])
    subject_ids = torch.tensor([-1, -1], dtype=torch.long)

    outputs = model(adf, grl_lambda=0.0)
    losses = ADFNetLoss()(outputs, labels, subject_ids)
    assert torch.isfinite(losses["loss"])
    assert float(losses["adv_ce"]) == 0.0
    losses["loss"].backward()


def test_grl_lambda_schedule_warmup_and_cap() -> None:
    from training.losses import grl_lambda_schedule

    assert grl_lambda_schedule(0, 10, max_lambda=0.3, warmup_epochs=3) == 0.0
    assert grl_lambda_schedule(2, 10, max_lambda=0.3, warmup_epochs=3) == 0.0
    peak = grl_lambda_schedule(9, 10, max_lambda=0.3, warmup_epochs=3)
    assert 0.0 < peak <= 0.3
    assert grl_lambda_schedule(5, 10, max_lambda=0.0) == 0.0
