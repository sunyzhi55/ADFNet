# ADFNet：任务完成驱动的跨受试者疲劳检测框架

ADFNet 基于 TF-GAM 与网络架构流程实现。模型将疲劳/专注状态建模为视线动态与任务目标轨迹之间的耦合质量，并通过 **Gamma 分布偏移**、**Mamba-MLA 时序编码**和 **GRL 对抗解耦**提升跨受试者泛化能力。

## 项目结构

```
ADFNet/
├── configs/
│   └── default.yaml              # 唯一配置文件（含 ablation 段）
├── scripts/
│   ├── train.py                  # 单 fold 调试
│   ├── run_loso.py               # LOSO 交叉验证
│   ├── run_group_kfold.py        # GroupKFold 交叉验证
│   ├── run_ablation.py           # 消融实验启动器（64 组合 + LSTM/Transformer 替换）
│   ├── evaluate.py               # checkpoint 评估（单模型/批量）
│   ├── check_session_confound.py # 诊断：session 混淆检查
│   ├── diagnose_grl_trajectory.py# 诊断：GRL 训练轨迹分析
│   └── visualize_face_features.py# 诊断：面部特征编码可视化
├── src/
│   ├── data/
│   │   ├── dataset.py            # ADFWindowDataset, WindowSample
│   │   ├── gaipat_dataset.py     # GaipatWindowDataset (GAIPAT 公开数据集)
│   │   ├── features.py           # compute_adf_features, sliding_mean
│   │   ├── io.py                 # SequenceInfo, discover_sequences
│   │   └── split.py              # SubjectFold, loso_folds, group_kfold_folds
│   ├── models/
│   │   ├── adfnet.py             # ADFNet 主模型（支持 ablation 配置）
│   │   ├── distribution.py       # GammaReference, DistributionBranch, soft-DTW
│   │   ├── grl.py                # GradientReverseLayer
│   │   ├── heads.py              # VigilanceHead, SubjectDiscriminator
│   │   └── mamba_encoder.py      # Mamba / LSTM / Transformer 时序编码器
│   ├── training/
│   │   ├── losses.py             # ADFNetLoss, grl_lambda_schedule
│   │   ├── metrics.py            # binary_metrics
│   │   ├── seed.py               # set_seed
│   │   └── trainer.py            # train_fold, evaluate_checkpoint
│   └── utils/
│       ├── basic.py              # get_optimizer, get_scheduler
│       ├── config.py             # load_config, save_hparams
│       └── logging.py            # setup_logger
├── tests/
│   ├── test_ablation.py          # 消融配置测试（26 项）
│   ├── test_dataset.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_model_forward.py
└── requirements.txt
```

## 模型架构

```
                          ┌─────────────────────┐
  ADF 序列 [B,T,3] ──────▶│  Mamba-MLA 时序编码器 │──▶ temp_feature [B, mamba_dim]
  (drift, diff, mean)     │  (或 LSTM/Transformer)│
                          └─────────────────────┘
                                                        ┌──────────┐
                          ┌─────────────────────┐       │          │   ┌───────────────┐
  dist_stats [B,3] ──────▶│   分布偏移分支       │──▶ dist_feature  │ concat │──▶ fusion_feature
  (log-lik, wass, sdtw)   │ (Gamma/LogNormal/   │   [B, dist_dim]  │        │   [B, fusion_dim]
                          │  Gaussian/Weibull/  │       └──────────┘   └───────┬───────┘
                          │  Rayleigh/KDE + MLP)│
                                                                               │
                                                              ┌────────────────┼────────────────┐
                                                              ▼                │                ▼
                                                     VigilanceHead             │         GRL → SubjectDiscriminator
                                                     (疲劳二分类)              │         (身份对抗, 梯度反转)
                                                              │                │                │
                                                              ▼                │                ▼
                                                       vigilance_logit         │         subject_logit
```

双分支融合后经主任务头输出疲劳预测；GRL 对抗头从融合特征中剥离身份信息（subject_id），迫使编码器学习跨被试通用的疲劳表征。

## 核心组件

- **ADF 三通道特征**：空间漂移（gaze-target 距离）、一阶差分（变化率）、局部滑动均值（趋势平滑）。
- **Gamma 清醒基准分布**：仅使用当前训练 fold 的 alert 样本拟合 Gamma 分布（也支持 Gaussian / LogNormal / Weibull / Rayleigh / KDE 替换），计算 Mean Log-Likelihood、Wasserstein 距离、Soft-DTW 距离作为分布偏移特征，避免测试主体泄露。
- **Mamba-MLA 时序编码器**：基于 `mamba-ssm` 的选择性状态空间模型，带残差连接和 LayerNorm，mean pooling 输出。
- **GRL 对抗解耦**：以 subject_id 分类为对抗目标（交叉熵 + GRL），剥离个人身份/注视习惯噪声。身份标签与疲劳标签正交（每被试 alert/sleep 各半），对抗擦除不会系统性擦除疲劳信号。
- **任务模式过滤**：支持 `easy`、`hard`、`all` 三种训练/评估模式。
- **LOSO 与 GroupKFold**：按 subject 物理隔离的交叉验证策略。

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
pip install mamba-ssm  # 需要 CUDA 环境
```

### 训练

LOSO 交叉验证：

```bash
python scripts/run_loso.py --config configs/default.yaml --task-mode easy
python scripts/run_loso.py --config configs/default.yaml --task-mode hard
```

GroupKFold 交叉验证：

```bash
python scripts/run_group_kfold.py --config configs/default.yaml --n-splits 5 --task-mode easy
```

调试单个 fold：

```bash
python scripts/run_loso.py --config configs/default.yaml --task-mode all --max-folds 1
```

所有入口都支持 `--task-mode`：`train.py`、`run_loso.py`、`run_group_kfold.py`、`evaluate.py`。

### 评估

单模型评估：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/loso_01_easy/best.pt \
  --data-root /path/to/test_jsonl \
  --task-mode easy
```

批量测试集评估（推荐）：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --run-dir outputs/<timestamp>_ADFNet_Exp_loso/ \
  --task-mode hard
```

## FatigueGuard 数据集

### 预处理结果

(1) Easy 任务数据格式：
```
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "gaze_screen_tf_calibrate_xy_px": [1268.4, 713.2],
  "target_xy_px": [1280, 720],
  "deviation_px_before_calibrate": 65.35,
  "deviation_px_after_calibrate": 13.19,
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], [541.0, 320.0]],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998
}
```

(2) Hard 任务数据格式：
```
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "gaze_screen_tf_calibrate_xy_px": [1268.4, 713.2],
  "target_centers_xy_px": [[1280, 720], [960, 540]],
  "deviation_px_before_calibrate": 67.12,
  "deviation_px_after_calibrate": 13.70,
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], [541.0, 320.0]],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998
}
```


## GAIPAT 公开数据集与跨数据集实验

### GAIPAT 数据集

GAIPAT 是一个公开的视线交互数据集，包含 `release` 和 `grasp` 两类交互任务。每个试次（trial）以 JSONL 文件存储，包含 256 帧的视线偏差数据。

**目录结构：**

```
<gaipat_root>/
├── release/
│   ├── 5530740_house_4_release_21_0.jsonl
│   └── ...
└── grasp/
    ├── 5530740_house_4_grasp_21_1.jsonl
    └── ...
```

**文件命名规则：** `{subject_id}_{task}_{step}_{event}_{block_id}_{label}.jsonl`

**标签映射：**

| 标签 | 含义 | 对应 FatigueGuard |
|------|------|-------------------|
| 0 | 分心 (Distracted/Wandering) | 1 = sleepy |
| 1 | 专注 (Focused) | 0 = alert |
| 2, 3 | 丢弃 | — |

**核心特征：** `deviation_cm`（视线偏差距离，厘米）。

### 归一化策略

由于 FatigueGuard（像素）和 GAIPAT（厘米）的单位不一致，所有实验启用 **per-sample Min-Max 归一化**：

```
drift_normalized = (drift - min) / (max - min + eps)  →  [0, 1]
```

在 `configs/default.yaml` 中控制：

```yaml
data:
  per_sample_norm: true   # FatigueGuard

gaipat:
  per_sample_norm: true   # GAIPAT
```

### 四种实验模式

通过 `--eval-mode` 参数控制训练与测试的数据集组合：

| eval_mode | 训练集 | 测试集 | 说明 |
|-----------|--------|--------|------|
| `fatigue` | FatigueGuard | FatigueGuard | 同源实验（默认） |
| `fatigue_to_gaipat` | FatigueGuard | GAIPAT | 跨数据集泛化 |
| `gaipat` | GAIPAT | GAIPAT | GAIPAT 同源实验 |
| `gaipat_to_fatigue` | GAIPAT | FatigueGuard | 反向跨数据集泛化 |

### 运行跨数据集实验

**实验 1：FatigueGuard 同源（已实现）**

```bash
python scripts/run_loso.py --eval-mode fatigue --task-mode easy
python scripts/run_group_kfold.py --eval-mode fatigue --task-mode hard
```

**实验 2：FatigueGuard 训练 → GAIPAT 测试**

```bash
# 训练完成后自动评估 GAIPAT
nohup python scripts/run_loso.py --eval-mode fatigue_to_gaipat --task-mode easy --gaipat-dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 > result_loso_easy_fatigue_to_gaipat_0720.out &

# 跳过训练，直接加载已有 checkpoint 评估 GAIPAT
python scripts/run_loso.py --eval-mode fatigue_to_gaipat --task-mode easy --gaipat-dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 --checkpoint-dir outputs/<prev_run_dir>
```

**实验 3：GAIPAT 同源**

```bash
python scripts/run_loso.py --eval-mode gaipat \
  --gaipat-dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719

python scripts/run_group_kfold.py --eval-mode gaipat --n-splits 5 \
  --gaipat-dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719
```

**实验 4：GAIPAT 训练 → FatigueGuard 测试**

```bash
python scripts/run_loso.py --eval-mode gaipat_to_fatigue \
  --gaipat-dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719

# 仅评估（跳过训练）
python scripts/run_loso.py --eval-mode gaipat_to_fatigue \
  --gaipat-dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 \
  --checkpoint-dir outputs/<gaipat_run_dir>
```

### 消融实验 + 跨数据集

`run_ablation.py` 同样支持 `--eval-mode`：

```bash
# 完整模型：FG 训练 → GAIPAT 测试
python scripts/run_ablation.py --preset full --eval-mode fatigue_to_gaipat --gaipat-dir /path/to/gaipat


nohup python scripts/run_ablation.py --preset full --eval-mode fatigue_to_gaipat --gaipat-dir /data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719 > result_ablation_full_fatigue_to_gaipat_0720.out &


# 去掉 GRL 的跨数据集评估
python scripts/run_ablation.py --preset no_grl --eval-mode fatigue_to_gaipat --gaipat-dir /path/to/gaipat

# GAIPAT 同源消融
python scripts/run_ablation.py --preset all_combinations --eval-mode gaipat --gaipat-dir /path/to/gaipat --cv loso

# 仅加载 checkpoint 做跨数据集评估
python scripts/run_ablation.py --preset full --eval-mode gaipat_to_fatigue --gaipat-dir /path/to/gaipat   --checkpoint-dir outputs/ablation/<gaipat_run>
```

### 输出结构

跨数据集实验每个 fold 同时输出同源和跨数据集两份 CSV:

```
outputs/<run_dir>/
├── loso_metrics_easy.csv                    # 同源（训练集验证）
├── loso_cross_to_gaipat_easy.csv            # 跨数据集（GAIPAT 测试）
├── loso_<fold>_easy/best.pt                 # 模型权重
└── hparams.json
```

## 消融实验

`scripts/run_ablation.py` 提供统一的消融实验入口，以 `configs/default.yaml` 为基线，系统性地禁用/替换各组件来衡量其贡献。

### 消融维度

6 个二值开关（2^6 = 64 种组合）：

| 开关 | 组件 | 禁用方式 |
|------|------|----------|
| `enable_gamma` | Gamma 分布对齐流 | 移除 DistributionBranch + GammaReference |
| `enable_grl` | GRL 对抗解耦 | 移除梯度反转层 + 身份判别器 |
| `enable_diff` | 一阶差分通道 | ADF channel 1 置零 |
| `enable_sliding_mean` | 滑动均值通道 | ADF channel 2 置零 |
| `enable_soft_dtw` | Soft-DTW 距离 | 分布特征维度 3 → 2 |
| `enable_mamba` | Mamba-MLA 时序编码器 | 移除整个时序分支 |

2 种时序编码器替换实验（独立运行，不参与组合遍历）：

| 替换 | 说明 |
|------|------|
| `temporal_encoder: lstm` | 用双向 LSTM 替换 Mamba-MLA |
| `temporal_encoder: transformer` | 用 Transformer（含正弦位置编码）替换 Mamba-MLA |

5 种分布拟合替换实验（独立运行，不参与组合遍历）：

| 替换 | 说明 |
|------|------|
| `reference_distribution: gaussian` | 用高斯分布替换 Gamma 分布（对称分布假设对比） |
| `reference_distribution: lognormal` | 用对数正态分布替换 Gamma 分布（重尾正偏分布对比） |
| `reference_distribution: weibull` | 用 Weibull 分布替换 Gamma 分布（灵活形状参数正偏分布对比） |
| `reference_distribution: rayleigh` | 用 Rayleigh 分布替换 Gamma 分布（单参数正偏分布对比） |
| `reference_distribution: kde` | 用核密度估计替换 Gamma 分布（非参数方法对比） |

### 配置

在 `configs/default.yaml` 中的 `ablation` 段控制，所有开关默认 `true`（完整模型）：

```yaml
ablation:
  enable_gamma: true
  enable_grl: true
  enable_diff: true
  enable_sliding_mean: true
  enable_soft_dtw: true
  enable_mamba: true
  temporal_encoder: "mamba"           # mamba | lstm | transformer
  reference_distribution: "gamma"     # gamma | gaussian | lognormal | weibull | rayleigh | kde
```

### 用法

```bash
# 完整模型基线（kfold + loso, easy + hard）
python scripts/run_ablation.py --preset full

# 单个消融
python scripts/run_ablation.py --preset no_gamma
python scripts/run_ablation.py --preset no_grl
python scripts/run_ablation.py --preset no_diff
python scripts/run_ablation.py --preset no_sliding_mean
python scripts/run_ablation.py --preset no_soft_dtw
python scripts/run_ablation.py --preset no_mamba

# 全部 64 种组合
python scripts/run_ablation.py --preset all_combinations

# LSTM / Transformer 替换
python scripts/run_ablation.py --preset lstm
python scripts/run_ablation.py --preset transformer

# Gaussian / KDE / LogNormal / Weibull / Rayleigh 分布替换
python scripts/run_ablation.py --preset gaussian
python scripts/run_ablation.py --preset kde
python scripts/run_ablation.py --preset lognormal
python scripts/run_ablation.py --preset weibull
python scripts/run_ablation.py --preset rayleigh

# 仅 kfold + easy（加速调试）
python scripts/run_ablation.py --preset no_grl --cv kfold --task-mode easy

# 指定 GPU
python scripts/run_ablation.py --preset all_combinations --device cuda:1

# 调试：每种配置只跑 1 个 fold
python scripts/run_ablation.py --preset all_combinations --max-folds 1
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--preset` | `full` | 消融预设：`full`, `no_gamma`, `no_grl`, `no_diff`, `no_sliding_mean`, `no_soft_dtw`, `no_mamba`, `all_combinations`, `lstm`, `transformer`, `gaussian`, `kde`, `lognormal`, `weibull`, `rayleigh` |
| `--cv` | `both` | 交叉验证：`kfold`, `loso`, `both` |
| `--task-mode` | `easy hard` | 任务难度，可多选 |
| `--config` | `configs/default.yaml` | 基线配置文件 |
| `--output-base` | `./outputs/ablation` | 消融输出根目录 |
| `--n-splits` | `5` | GroupKFold 折数 |
| `--max-folds` | `None` | 每种 CV 最多跑几个 fold（调试用） |
| `--device` | `None` | 覆盖 `training.device` |

### 输出结构

```
outputs/ablation/
├── {timestamp}_full/
│   ├── kfold_easy/
│   │   ├── fold_01_easy/history.csv
│   │   ├── fold_01_easy/final_metrics.csv
│   │   ├── fold_01_easy/best.pt
│   │   └── group_kfold_metrics_easy.csv
│   ├── kfold_hard/ ...
│   ├── loso_easy/ ...
│   └── loso_hard/ ...
├── {timestamp}_no_gamma/ ...
├── {timestamp}_no_grl/ ...
├── {timestamp}_lstm/ ...
└── {timestamp}_ablation_summary.csv   # 全部结果汇总
```

每个消融实验的 `hparams.json` 中额外记录 `ablation_label`、`ablation_overrides` 和 `preset` 字段，便于溯源。

### 批量并行运行（推荐）

`run_ablation.py` 是串行的（前一个跑完后一个才开始）。在服务器上推荐使用 `scripts/run_all_ablations.sh` 将所有实验以 `nohup` 并行提交到多张 GPU：

```bash
# 全部实验（64 组合 + LSTM/Transformer/Gaussian/KDE/LogNormal/Weibull/Rayleigh 替换），自动分配所有 GPU
bash scripts/run_all_ablations.sh all

# 指定 GPU 和并行度
bash scripts/run_all_ablations.sh all --gpus "0 1 2 3" --max-parallel 2

# 仅 64 种组合
bash scripts/run_all_ablations.sh combinations

# 仅 LSTM 替换的 32 种组合
bash scripts/run_all_ablations.sh lstm

# 仅 Transformer 替换的 32 种组合
bash scripts/run_all_ablations.sh transformer

# 仅 Gaussian 替换的 32 种组合
bash scripts/run_all_ablations.sh gaussian

# 仅 KDE 替换的 32 种组合
bash scripts/run_all_ablations.sh kde

# 仅 LogNormal 替换的 32 种组合
bash scripts/run_all_ablations.sh lognormal

# 仅 Weibull 替换的 32 种组合
bash scripts/run_all_ablations.sh weibull

# 仅 Rayleigh 替换的 32 种组合
bash scripts/run_all_ablations.sh rayleigh

# 单个预设
bash scripts/run_all_ablations.sh single no_grl
bash scripts/run_all_ablations.sh single full

# 冒烟测试：每种配置只跑 1 个 fold，仅 kfold + easy
bash scripts/run_all_ablations.sh all --max-folds 1 --cv kfold --task easy

# 查看运行状态
bash scripts/run_all_ablations.sh status
```

脚本特性：

- GPU 自动检测与 round-robin 分配，也可通过 `--gpus "0 1"` 手动指定。
- 每张 GPU 默认最多 2 个并行进程（`--max-parallel` 可调），超出自动排队等待。
- 每个实验独立日志文件在 `outputs/ablation/logs/` 下。
- `status` 命令实时显示每个 job 的完成/运行/失败状态。

输出目录结构：

```
outputs/ablation/
├── full/kfold_easy/        # 完整模型, kfold, easy
├── full/loso_hard/         # 完整模型, loso, hard
├── no_gamma/kfold_easy/    # 去掉 Gamma 分布, kfold, easy
├── no_gamma/loso_hard/
├── no_grl/...              # 去掉 GRL
├── lstm/...                # LSTM 替换
├── logs/                   # 所有实验的日志
│   ├── full_kfold_easy.log
│   ├── no_gamma_loso_hard.log
│   └── ...
└── ...
```

### CLI 消融参数

`run_loso.py` 和 `run_group_kfold.py` 均支持 `--ablation` 参数，可以直接在命令行覆盖消融配置，无需修改 YAML：

```bash
# 去掉 GRL 的 LOSO
python scripts/run_loso.py \
  --task-mode easy \
  --ablation enable_grl=false \
  --exp-name ablation_no_grl

# 去掉 Gamma + 去掉 Soft-DTW 的 KFold
python scripts/run_group_kfold.py \
  --task-mode hard \
  --ablation enable_gamma=false enable_soft_dtw=false \
  --exp-name ablation_noGamma_noSDTW \
  --output-dir ./outputs/ablation/noGamma_noSDTW

# LSTM 替换
python scripts/run_loso.py \
  --task-mode easy \
  --ablation temporal_encoder=lstm \
  --exp-name ablation_lstm

# Gaussian 分布替换
python scripts/run_loso.py --task-mode easy --ablation reference_distribution=gaussian --exp-name ablation_gaussian

# KDE 分布替换
python scripts/run_loso.py --task-mode easy --ablation reference_distribution=kde --exp-name ablation_kde

# LogNormal 分布替换
python scripts/run_loso.py --task-mode easy --ablation reference_distribution=lognormal --exp-name ablation_lognormal

# Weibull 分布替换
nohup python scripts/run_loso.py --task-mode easy --ablation reference_distribution=weibull --exp-name ablation_weibull > result_loso_easy_weibull_0722.out &

# Rayleigh 分布替换
nohup python scripts/run_loso.py --task-mode easy --ablation reference_distribution=rayleigh --exp-name ablation_rayleigh > result_loso_easy_rayleigh_0722.out &


CUDA_VISIBLE_DEVICES=6 nohup python scripts/run_loso.py --task-mode hard > result_loso_hard_1024_512_0710.out &
```

也支持 `--exp-name` 和 `--output-dir` 覆盖实验名称和输出路径。

手动 `nohup` 单个实验:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_loso.py \
  --config configs/default.yaml \
  --task-mode easy \
  --ablation enable_grl=false \
  --exp-name ablation_no_grl \
  --output-dir ./outputs/ablation/no_grl/loso_easy \
  > logs/no_grl_loso_easy.log 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_loso.py   --config configs/default.yaml   --task-mode easy   --ablation enable_grl=false   --exp-name ablation_no_grl   --output-dir ./outputs/ablation/no_grl/loso_easy   > no_grl_loso_easy.log & 

nohup python scripts/run_loso.py   --config configs/default.yaml   --task-mode easy   --ablation enable_gamma=false   --exp-name ablation_no_gamma   --output-dir ./outputs/ablation/no_gamma/loso_easy   > no_gamma_loso_easy.log & 

CUDA_VISIBLE_DEVICES=3 nohup python scripts/run_loso.py   --config configs/default.yaml   --task-mode easy   --ablation enable_gamma=false   --exp-name ablation_no_gamma   --output-dir ./outputs/ablation/no_gamma/loso_easy  > no_gamma_loso_easy.log &
```

## 数据划分：验证集与测试集

在 `configs/default.yaml` 的 `split` 段配置：

```yaml
split:
  test_subjects: ["19", "20"]   # hold-out 测试被试；[] 表示不设独立测试集
  explicit_folds: [["01","05","14","19"], ["02","06","10","15"], ...]  # 显式 fold 划分
```

- **`test_subjects`**：训练时整体 hold-out，不进入任何 fold 的 train/val。训练结束后用 `evaluate.py --run-dir` 做最终评估。留空 `[]` 则不设独立测试集。
- **`explicit_folds`**：形如 `[["01","02"], ...]`，每个子列表是该 fold 的 val 被试。设为 `null` 时按 LOSO/GroupKFold 自动生成。

## 最佳轮选择与早停

训练按 `training.result_selection` 从所有 epoch 中选出最佳一轮写入 `final_metrics.csv`，而非使用最后一个 epoch：

```yaml
training:
  result_selection:
    monitor: "val_f1"
    mode: "max"
    min_delta: 0.0
  early_stopping:
    enabled: true
    monitor: "val_f1"
    mode: "max"
    patience: 1000      # 等效于不提前停止，跑完全部 epoch 后选最佳轮
    min_delta: 1.0e-4
```

`best.pt` 也按 `result_selection` 保存，保证模型权重和最终 CSV 对应同一轮。

## CSV 输出

每个 fold：

```
outputs/<fold>_<task_mode>/history.csv         # 逐 epoch 训练记录
outputs/<fold>_<task_mode>/final_metrics.csv   # 最佳轮指标
outputs/<fold>_<task_mode>/best.pt             # 最佳轮 checkpoint
```

汇总（末尾含 `mean`/`std` 行）：

```
loso_metrics_<task_mode>.csv
group_kfold_metrics_<task_mode>.csv
```

### 指标字段

基础指标：`auc`, `acc`, `f1`, `precision`, `recall`

混淆矩阵：`cm_tn`, `cm_fp`, `cm_fn`, `cm_tp`

训练记录中每个指标带 `train_`/`val_` 前缀，另含 `train_loss`, `val_loss`, `grl_lambda`, `adv_ce`, `subject_acc`。

`final_metrics.csv` 额外包含：`best_epoch`, `selection_monitor`, `selection_mode`, `selection_value`。

## 超参数记录

每次训练在输出目录写 `hparams.json`，集中记录全部超参数，便于对比与消融溯源：

```json
{
  "script": "run_loso.py",
  "task_mode": "easy",
  "seed": 42,
  "exp_name": "ADFNet_Exp_loso",
  "git_commit": "...",
  "n_folds": 20,
  "val_subjects_per_fold": [["01"], ["02"], ...],
  "config": { ... }
}
```

## 性能调优

```yaml
training:
  batch_size: 128
  num_workers: 16
  pin_memory: true
  persistent_workers: true

distribution:
  soft_dtw_reference_samples: 64  # 降到 32 可加速，但精度略降
```

Soft-DTW 是 O(n*m) 复杂度，`soft_dtw_reference_samples` 控制参考序列下采样长度。分布特征在每个 fold 开始前预计算并缓存，避免每个 batch 重复计算。

## 测试

```bash
pytest                      # 运行全部测试
pytest tests/test_ablation.py -v   # 仅运行消融测试
```

需要 `mamba-ssm` 的测试会自动跳过（`@pytest.mark.skipif`），在 Linux CUDA 服务器上运行即可覆盖。

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- mamba-ssm（需 CUDA，Linux 环境）
- scipy, scikit-learn, pandas, numpy, pyyaml, tqdm, matplotlib

```bash
pip install -r requirements.txt
pip install mamba-ssm
```
