"""
计算 ADFNet 的参数量、FLOPs、MACs。

使用 calflops 库，对默认配置（Mamba-MLA 时序编码器 + Gamma 分布对齐）进行测量。
由于 mamba_ssm 在非 CUDA 环境不可用，用参数等价的 mock 替代。
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

# ── 1. 检查 mamba_ssm + CUDA 可用性 ──────────────────────────
USE_REAL_MAMBA = False
try:
    from mamba_ssm import Mamba  # noqa: F401
    if torch.cuda.is_available():
        USE_REAL_MAMBA = True
        print("[OK] mamba_ssm + CUDA available, using real Mamba on GPU\n")
    else:
        print("[WARN] mamba_ssm found but CUDA unavailable, falling back to mock\n")
except ImportError:
    print("[WARN] mamba_ssm not installed, using parameter-equivalent mock (FLOPs excludes SSM selective-scan)\n")

if not USE_REAL_MAMBA:

    class _MockMamba(nn.Module):
        """参数数量与 mamba_ssm.Mamba(d_model=D) 完全一致的 mock。

        真实 Mamba 内部:
          - in_proj:  Linear(D, 4D, bias=False)   → 4D²  params
          - conv1d:   Conv1d(2D, 2D, 4, groups=2D, padding=3, bias=True) → 2D·4+2D params
          - out_proj: Linear(2D, D, bias=False)    → 2D²  params
        """
        def __init__(self, d_model: int, **kwargs):
            super().__init__()
            self.in_proj = nn.Linear(d_model, 4 * d_model, bias=False)
            self.conv1d = nn.Conv1d(
                2 * d_model, 2 * d_model, kernel_size=4,
                groups=2 * d_model, padding=3, bias=True,
            )
            self.out_proj = nn.Linear(2 * d_model, d_model, bias=False)

        def forward(self, x):
            # 仅保证 tensor shape 正确，不模拟真实 SSM 计算
            zxb = self.in_proj(x)                       # (B, L, 4D)
            zx = zxb[..., : 2 * self.in_proj.in_features]  # (B, L, 2D)
            zx_t = zx.transpose(1, 2)                   # (B, 2D, L)
            zx_t = self.conv1d(zx_t)[..., :zx_t.shape[-1]]  # truncate to L
            zx = zx_t.transpose(1, 2)                   # (B, L, 2D)
            return self.out_proj(torch.tanh(zx))         # (B, L, D)

    # 注入 mock 模块
    mamba_mock = types.ModuleType("mamba_ssm")
    mamba_mock.Mamba = _MockMamba
    sys.modules["mamba_ssm"] = mamba_mock

# ── 2. 导入项目模型 ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.adfnet import ADFNet

# ── 3. 从 default.yaml 读取默认配置 ────────────────────────────
import yaml
cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
window_size = cfg["data"]["window_size"]

# ── 4. 构建模型 ───────────────────────────────────────────────
# 不传 ablation → 使用代码内的默认值（全部组件启用, temporal_encoder=mamba）
model = ADFNet(
    input_dim=model_cfg["input_dim"],          # 3
    dist_feat_dim=model_cfg["dist_feat_dim"],  # 3
    dist_hidden_dim=model_cfg["dist_hidden_dim"],  # 32
    dist_out_dim=model_cfg["dist_out_dim"],    # 64
    mamba_dim=model_cfg["mamba_dim"],          # 128
    mamba_layers=model_cfg["mamba_layers"],    # 2
    fusion_dim=model_cfg["fusion_dim"],        # 192（动态计算覆盖）
    n_subjects=model_cfg["n_subjects"],        # 20
    dropout=model_cfg["dropout"],              # 0.2
)
model.eval()

# ── 5. 设备选择（真实 Mamba 必须用 CUDA） ─────────────────────
device = torch.device("cuda" if USE_REAL_MAMBA else "cpu")
model = model.to(device)

# ── 6. 准备 dummy 输入 ────────────────────────────────────────
B = 2
T = window_size   # 256 frames
adf = torch.randn(B, T, model_cfg["input_dim"], device=device)
dist_stats = torch.randn(B, model_cfg["dist_feat_dim"], device=device)

# ── 7. 用 calflops 测量 ──────────────────────────────────────
from calflops import calculate_flops

flops, macs, params = calculate_flops(
    model=model,
    args=[adf, dist_stats],
    output_as_string=True,
    print_results=False,
)

# ── 8. 输出结果 ───────────────────────────────────────────────
print("=" * 60)
print("  ADFNet Model Complexity")
print("=" * 60)
print(f"  Device: {device}")
print(f"  Config: input_dim={model_cfg['input_dim']}, "
      f"mamba_dim={model_cfg['mamba_dim']}, "
      f"mamba_layers={model_cfg['mamba_layers']}")
print(f"          dist_out_dim={model_cfg['dist_out_dim']}, "
      f"dist_hidden_dim={model_cfg['dist_hidden_dim']}, "
      f"n_subjects={model_cfg['n_subjects']}")
print(f"  Input:  adf({B}, {T}, {model_cfg['input_dim']}), "
      f"dist_stats({B}, {model_cfg['dist_feat_dim']})")
print("-" * 60)
print(f"  FLOPs:  {flops}")
print(f"  MACs:   {macs}")
print(f"  Params: {params}")
print("=" * 60)

if not USE_REAL_MAMBA:
    print("\nNote: Using mock Mamba, FLOPs/MACs exclude SSM selective-scan O(L*D) cost.")
    print("      Parameter count is exact; FLOPs covers projection + conv layers (dominant cost).")

# ── 8. 各子模块参数明细 ──────────────────────────────────────
print("\n-- Module parameter breakdown --")
for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters()) if module is not None else 0
    print(f"  {name:30s}  {n:>10,} params")
total = sum(p.numel() for p in model.parameters())
print(f"  {'TOTAL':30s}  {total:>10,} params")
