"""消融配置前向/反向传播测试。

覆盖每个单独消融变体、LSTM/Transformer 替换、以及若干极端组合，
确保模型在各种 ablation 配置下均能正确构建、前向推理和反向传播。
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HAS_MAMBA = importlib.util.find_spec("mamba_ssm") is not None

# 小型模型参数（节省测试时间）
SMALL_KW = dict(mamba_dim=16, mamba_layers=1, dist_out_dim=16, dist_hidden_dim=8, n_subjects=5)


def _make_inputs(batch=2, seq_len=8, feat_dim=3):
    adf = torch.randn(batch, seq_len, feat_dim)
    dist_stats = torch.randn(batch, 3)
    labels = torch.tensor([[0.0], [1.0]])
    subject_ids = torch.tensor([0, 3], dtype=torch.long)
    return adf, dist_stats, labels, subject_ids


# ══════════════════════════════════════════════════════════════
# 向后兼容：无 ablation 配置 = 完整模型
# ══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm 未安装")
def test_backward_compatible_no_ablation():
    """不带 ablation 参数时应等价于完整模型。"""
    from models.adfnet import ADFNet
    from training.losses import ADFNetLoss

    model = ADFNet(**SMALL_KW, fusion_dim=32)
    adf, dist_stats, labels, subject_ids = _make_inputs()
    outputs = model(adf, dist_stats=dist_stats, grl_lambda=0.5)
    losses = ADFNetLoss()(outputs, labels, subject_ids)
    losses["loss"].backward()
    assert outputs["vigilance_logit"].shape == (2, 1)
    assert outputs["subject_logit"].shape == (2, 5)


# ══════════════════════════════════════════════════════════════
# 每个单独消融变体
# ══════════════════════════════════════════════════════════════

SINGLE_ABLATIONS = [
    ({"enable_gamma": False},       "no_gamma"),
    ({"enable_grl": False},         "no_grl"),
    ({"enable_diff": False},        "no_diff"),
    ({"enable_sliding_mean": False},"no_sliding_mean"),
    ({"enable_soft_dtw": False},    "no_soft_dtw"),
]


@pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm 未安装")
@pytest.mark.parametrize("abl,name", SINGLE_ABLATIONS, ids=[n for _, n in SINGLE_ABLATIONS])
def test_single_ablation_forward_backward(abl, name):
    from models.adfnet import ADFNet
    from training.losses import ADFNetLoss

    model = ADFNet(**SMALL_KW, fusion_dim=32, ablation=abl)
    adf, dist_stats, labels, subject_ids = _make_inputs()

    # 如果 gamma 禁用，dist_stats 不应被使用
    ds = None if not abl.get("enable_gamma", True) else dist_stats
    outputs = model(adf, dist_stats=ds, grl_lambda=0.5)

    assert outputs["vigilance_logit"].shape == (2, 1)
    losses = ADFNetLoss()(outputs, labels, subject_ids)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


# ══════════════════════════════════════════════════════════════
# no_mamba（时序分支完全禁用，仅靠分布分支）
# ══════════════════════════════════════════════════════════════

def test_no_mamba_distribution_only():
    """禁用 Mamba 时仅用分布分支，无需 mamba-ssm。"""
    from models.adfnet import ADFNet
    from training.losses import ADFNetLoss

    model = ADFNet(**SMALL_KW, fusion_dim=32, ablation={"enable_mamba": False})
    adf, dist_stats, labels, subject_ids = _make_inputs()
    outputs = model(adf, dist_stats=dist_stats, grl_lambda=0.5)

    assert outputs["temp_feature"] is None
    assert outputs["dist_feature"] is not None
    assert outputs["vigilance_logit"].shape == (2, 1)
    losses = ADFNetLoss()(outputs, labels, subject_ids)
    losses["loss"].backward()


# ══════════════════════════════════════════════════════════════
# LSTM / Transformer 替换变体
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("enc_type", ["lstm", "transformer"])
def test_temporal_encoder_replacement(enc_type):
    from models.adfnet import ADFNet
    from training.losses import ADFNetLoss

    model = ADFNet(**SMALL_KW, fusion_dim=32,
                   ablation={"temporal_encoder": enc_type})
    adf, dist_stats, labels, subject_ids = _make_inputs()
    outputs = model(adf, dist_stats=dist_stats, grl_lambda=0.5)

    assert outputs["temp_feature"] is not None
    assert outputs["temp_feature"].shape == (2, SMALL_KW["mamba_dim"])
    assert outputs["vigilance_logit"].shape == (2, 1)
    losses = ADFNetLoss()(outputs, labels, subject_ids)
    losses["loss"].backward()


# ══════════════════════════════════════════════════════════════
# GRL 禁用时 subject_logit 行为
# ══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm 未安装")
def test_no_grl_subject_discriminator_absent():
    from models.adfnet import ADFNet

    model = ADFNet(**SMALL_KW, fusion_dim=32, ablation={"enable_grl": False})
    assert model.subject_discriminator is None

    adf, dist_stats, labels, subject_ids = _make_inputs()
    outputs = model(adf, dist_stats=dist_stats, grl_lambda=1.0)
    # subject_logit 应为占位零张量
    assert outputs["subject_logit"].shape[0] == 2


# ══════════════════════════════════════════════════════════════
# Soft-DTW 禁用时分布特征维度变化
# ══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm 未安装")
def test_no_soft_dtw_reduces_dist_feat_dim():
    from models.adfnet import ADFNet

    model = ADFNet(**SMALL_KW, fusion_dim=32, ablation={"enable_soft_dtw": False})
    # 分布分支输入维度应为 2（原 3 - 1 soft_dtw）
    dist_branch = model.distribution_branch
    first_linear = dist_branch.net[0]
    assert first_linear.in_features == 2


# ══════════════════════════════════════════════════════════════
# 通道消融：diff 和 sliding_mean 置零
# ══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm 未安装")
def test_channel_ablation_zeros_disabled_channels():
    from models.adfnet import ADFNet

    model = ADFNet(**SMALL_KW, fusion_dim=32,
                   ablation={"enable_diff": False, "enable_sliding_mean": False})
    adf = torch.randn(2, 8, 3)
    # 手动验证：forward 后 channel 1,2 被置零
    # 通过 hook 抓取传入 temporal_encoder 的实际张量
    captured = {}
    def hook(module, inp, out):
        captured["input"] = inp[0].detach().clone()
    handle = model.temporal_encoder.register_forward_hook(hook)
    model(adf, dist_stats=torch.randn(2, 3), grl_lambda=0.0)
    handle.remove()
    assert (captured["input"][..., 1] == 0).all()
    assert (captured["input"][..., 2] == 0).all()


# ══════════════════════════════════════════════════════════════
# 极端组合：仅 temporal、仅 distribution
# ══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm 未安装")
def test_only_temporal_branch():
    from models.adfnet import ADFNet
    from training.losses import ADFNetLoss

    abl = {"enable_gamma": False, "enable_grl": False}
    model = ADFNet(**SMALL_KW, fusion_dim=32, ablation=abl)
    adf, _, labels, subject_ids = _make_inputs()
    outputs = model(adf, grl_lambda=0.0)
    assert outputs["dist_feature"] is None
    assert outputs["temp_feature"] is not None
    losses = ADFNetLoss(loss_weight=0.0)(outputs, labels, subject_ids)
    losses["loss"].backward()


def test_both_branchs_disabled_raises():
    from models.adfnet import ADFNet

    with pytest.raises(ValueError, match="At least one branch"):
        ADFNet(**SMALL_KW, fusion_dim=32,
               ablation={"enable_gamma": False, "enable_mamba": False})


# ══════════════════════════════════════════════════════════════
# 多组件组合消融（采样若干代表性组合）
# ══════════════════════════════════════════════════════════════

COMBO_SAMPLES = [
    {"enable_gamma": False, "enable_grl": False},
    {"enable_diff": False, "enable_sliding_mean": False},
    {"enable_soft_dtw": False, "enable_gamma": False},
    {"enable_grl": False, "enable_diff": False, "enable_sliding_mean": False},
]


@pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm 未安装")
@pytest.mark.parametrize("abl", COMBO_SAMPLES, ids=[str(a) for a in COMBO_SAMPLES])
def test_multi_component_ablation(abl):
    from models.adfnet import ADFNet
    from training.losses import ADFNetLoss

    model = ADFNet(**SMALL_KW, fusion_dim=32, ablation=abl)
    adf, dist_stats, labels, subject_ids = _make_inputs()
    ds = None if not abl.get("enable_gamma", True) else dist_stats
    outputs = model(adf, dist_stats=ds, grl_lambda=0.5)

    assert outputs["vigilance_logit"].shape == (2, 1)
    lw = 0.0 if not abl.get("enable_grl", True) else 1.0
    losses = ADFNetLoss(loss_weight=lw)(outputs, labels, subject_ids)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


# ══════════════════════════════════════════════════════════════
# GammaReference enable_soft_dtw 特性
# ══════════════════════════════════════════════════════════════

def test_gamma_reference_soft_dtw_toggle():
    import numpy as np
    from models.distribution import GammaReference

    rng = np.random.default_rng(42)
    data = rng.gamma(2.0, 1.0, size=200).astype(np.float32)

    gr_full = GammaReference.fit(data, reference_sample_count=64, enable_soft_dtw=True)
    gr_no_dtw = GammaReference.fit(data, reference_sample_count=64, enable_soft_dtw=False)

    assert gr_full.feat_dim == 3
    assert gr_no_dtw.feat_dim == 2

    feats_full = gr_full.window_features(data[:50])
    feats_no_dtw = gr_no_dtw.window_features(data[:50])
    assert len(feats_full) == 3
    assert len(feats_no_dtw) == 2
    # 前两个特征（mean_log_likelihood, wasserstein）应一致
    assert abs(feats_full[0] - feats_no_dtw[0]) < 1e-5
    assert abs(feats_full[1] - feats_no_dtw[1]) < 1e-5


# ══════════════════════════════════════════════════════════════
# _parse_ablation 默认值
# ══════════════════════════════════════════════════════════════

def test_parse_ablation_defaults():
    from models.adfnet import _parse_ablation

    defaults = _parse_ablation(None)
    assert defaults["enable_gamma"] is True
    assert defaults["enable_grl"] is True
    assert defaults["enable_diff"] is True
    assert defaults["enable_sliding_mean"] is True
    assert defaults["enable_soft_dtw"] is True
    assert defaults["enable_mamba"] is True
    assert defaults["temporal_encoder"] == "mamba"

    partial = _parse_ablation({"enable_grl": False})
    assert partial["enable_grl"] is False
    assert partial["enable_gamma"] is True  # 未指定的保持默认


# ══════════════════════════════════════════════════════════════
# LSTM / Transformer 独立编码器测试
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("EncoderCls_name", ["LSTMTemporalEncoder", "TransformerTemporalEncoder"])
def test_standalone_encoder_shapes(EncoderCls_name):
    import models.mamba_encoder as me
    EncoderCls = getattr(me, EncoderCls_name)
    enc = EncoderCls(input_dim=3, mamba_dim=16, layers=1, dropout=0.1)
    x = torch.randn(4, 32, 3)
    out = enc(x)
    assert out.shape == (4, 16)


# ══════════════════════════════════════════════════════════════
# Gaussian / LogNormal / KDE 分布替换消融
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dist_type", ["gamma", "gaussian", "lognormal", "kde"])
def test_reference_distribution_fit_and_features(dist_type):
    """四种分布类型均能拟合、计算特征、输出正确维度。"""
    import numpy as np
    from models.distribution import ReferenceDistribution

    rng = np.random.default_rng(42)
    data = rng.gamma(2.0, 1.0, size=300).astype(np.float32)

    ref = ReferenceDistribution.fit(
        data, dist_type=dist_type,
        reference_sample_count=64,
        enable_soft_dtw=True,
    )
    assert ref.feat_dim == 3
    assert ref.dist_type == dist_type
    assert ref.reference_samples.shape == (64,)

    feats = ref.window_features(data[:50])
    assert len(feats) == 3
    assert all(np.isfinite(f) for f in feats)


def test_reference_distribution_checkpoint_roundtrip():
    """四种分布类型的 checkpoint 序列化/反序列化均正确。"""
    import numpy as np
    from models.distribution import ReferenceDistribution

    rng = np.random.default_rng(42)
    data = rng.gamma(2.0, 1.0, size=200).astype(np.float32)
    cfg = {"distribution": {"eps": 1e-6, "soft_dtw_gamma": 1.0,
                            "soft_dtw_reference_samples": 32}}

    for dist_type in ["gamma", "gaussian", "lognormal", "kde"]:
        ref = ReferenceDistribution.fit(data, dist_type=dist_type,
                                        reference_sample_count=32)
        state = ref.to_checkpoint()
        assert state["dist_type"] == dist_type

        restored = ReferenceDistribution.from_checkpoint(state, cfg)
        assert restored.dist_type == dist_type
        assert restored.feat_dim == ref.feat_dim
        np.testing.assert_array_equal(restored.reference_samples, ref.reference_samples)

        # 特征计算结果一致
        feats_orig = ref.window_features(data[:30])
        feats_rest = restored.window_features(data[:30])
        for a, b in zip(feats_orig, feats_rest):
            assert abs(a - b) < 1e-4


def test_reference_distribution_backward_compat_old_checkpoint():
    """旧格式 checkpoint（无 dist_type 字段）应被正确加载为 gamma。"""
    import numpy as np
    from models.distribution import ReferenceDistribution

    old_state = {
        "shape": 2.0, "loc": 0.0, "scale": 1.0,
        "reference_samples": np.ones(32, dtype=np.float32),
        "enable_soft_dtw": True,
    }
    cfg = {"distribution": {"eps": 1e-6, "soft_dtw_gamma": 1.0,
                            "soft_dtw_reference_samples": 16}}
    ref = ReferenceDistribution.from_checkpoint(old_state, cfg)
    assert ref.dist_type == "gamma"
    assert ref.params["shape"] == 2.0


def test_parse_ablation_reference_distribution_default():
    from models.adfnet import _parse_ablation

    defaults = _parse_ablation(None)
    assert defaults["reference_distribution"] == "gamma"

    abl = _parse_ablation({"reference_distribution": "kde"})
    assert abl["reference_distribution"] == "kde"

    abl_ln = _parse_ablation({"reference_distribution": "lognormal"})
    assert abl_ln["reference_distribution"] == "lognormal"


# ══════════════════════════════════════════════════════════════
# LogNormal 分布拟合专项测试
# ══════════════════════════════════════════════════════════════

def test_lognormal_distribution_fit_params():
    """LogNormal 拟合应存储 lognorm_s/lognorm_loc/lognorm_scale 参数。"""
    import numpy as np
    from models.distribution import ReferenceDistribution

    rng = np.random.default_rng(42)
    # 生成对数正态分布数据: log(X) ~ N(1.0, 0.5^2)
    data = rng.lognormal(mean=1.0, sigma=0.5, size=500).astype(np.float32)

    ref = ReferenceDistribution.fit(
        data, dist_type="lognormal",
        reference_sample_count=128,
        enable_soft_dtw=True,
    )
    assert ref.dist_type == "lognormal"
    assert "lognorm_s" in ref.params
    assert "lognorm_loc" in ref.params
    assert "lognorm_scale" in ref.params
    assert ref.params["lognorm_s"] > 0
    assert ref.reference_samples.shape == (128,)
    assert all(s > 0 for s in ref.reference_samples)  # 对数正态样本 > 0

    # 特征计算
    feats = ref.window_features(data[:50])
    assert len(feats) == 3
    assert all(np.isfinite(f) for f in feats)


def test_lognormal_distribution_synthetic_goodness_of_fit():
    """合成对数正态数据上, LogNormal 的 AIC 应优于 Gaussian。"""
    import numpy as np
    from scipy.stats import lognorm as scipy_lognorm, norm as scipy_norm
    from models.distribution import ReferenceDistribution

    rng = np.random.default_rng(123)
    data = rng.lognormal(mean=0.5, sigma=0.8, size=500).astype(np.float32)
    clean = np.clip(data, 1e-6, None).astype(np.float64)

    ref_ln = ReferenceDistribution.fit(data, dist_type="lognormal",
                                       reference_sample_count=64)
    ref_gauss = ReferenceDistribution.fit(data, dist_type="gaussian",
                                          reference_sample_count=64)

    # Log-likelihood under each model
    ll_ln = scipy_lognorm.logpdf(
        clean, ref_ln.params["lognorm_s"],
        loc=ref_ln.params["lognorm_loc"],
        scale=ref_ln.params["lognorm_scale"],
    ).sum()
    ll_gauss = scipy_norm.logpdf(
        clean, loc=ref_gauss.params["mu"], scale=ref_gauss.params["sigma"],
    ).sum()

    # 对数正态数据的 LogNormal 似然应显著优于 Gaussian
    assert ll_ln > ll_gauss
