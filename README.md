# ADFNet：任务完成驱动的跨受试者疲劳检测框架

ADFNet 基于 `TF-GAM Task-Completion-Driven Adaptive Modeling for Robust Cross-Subject Fatigue Detection.pdf` 和 `网络架构流程.pdf` 实现。模型将疲劳/专注状态建模为“视线动态与任务锚定轨迹之间的耦合质量”，并通过分布对齐、Mamba 时序编码和 GRL 对抗解耦提升跨受试者泛化能力。

## 核心模块

- ADF 三通道特征：空间漂移、一阶差分、局部滑动均值。
- 清醒状态 Gamma 基准分布：仅使用训练 fold 的 alert 数据拟合，避免测试主体泄露。
- 分布偏移分支：Mean Log-Likelihood、Wasserstein 距离、soft-DTW 距离。
- 真实 `mamba-ssm` 时序分支：提取长程视线动态。
- GRL 对抗解耦：利用 landmarks 回归头迫使融合表征剥离个体物理指纹。
- LOSO 与 GroupKFold：按 subject 物理隔离评估跨人泛化能力。
- 早停策略：默认监控 `val_auc`，连续 8 个 epoch 无显著提升后停止训练。

## 目录结构

```text
configs/default.yaml          # 默认配置
requirements.txt              # Python 依赖
scripts/train.py              # 单次训练入口
scripts/evaluate.py           # checkpoint 评估入口
scripts/run_loso.py           # 留一受试者验证
scripts/run_group_kfold.py    # 按受试者分组 K 折验证
src/data                      # JSONL 读取、ADF 特征、subject split
src/models                    # ADFNet、Mamba、Gamma 分布、GRL、预测头
src/training                  # 训练循环、损失、指标、随机种子
tests                         # 单元测试
```

## 环境安装

建议在 Linux/CUDA 环境安装，因为 `mamba-ssm` 在 Windows 上可能需要额外编译配置。

```bash
pip install -r requirements.txt
```

本项目按设计使用真实 `mamba-ssm`，不提供近似替代编码器。

## 数据格式

`configs/default.yaml` 中的 `data.root` 指向 JSONL 数据目录。文件名格式：

```text
[subject_id]_[easy|hard]_[alert|sleep].jsonl
```

代码也兼容历史命名 `sleepy`，但推荐统一使用 `sleep`。

easy 任务每行示例：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "gaze_screen_tf_calibrate_xy_px": [320, 240],
  "target_xy_px": [300, 220],
  "bbox": [0, 0, 10, 10],
  "landmarks": [[0, 1], [2, 3]],
  "confidence": 0.99
}
```

hard 任务将 `target_xy_px` 替换为多目标中心：

```json
{
  "target_centers_xy_px": [[100, 100], [320, 240], [500, 300]]
}
```

标签映射：`alert=0`，`sleep=1`。

## 训练与评估

检查配置和数据窗口：

```bash
python scripts/train.py --config configs/default.yaml --dry-run
```

普通训练：

```bash
python scripts/train.py --config configs/default.yaml
```

LOSO：

```bash
python scripts/run_loso.py --config configs/default.yaml
```

调试单个 fold：

```bash
python scripts/run_loso.py --config configs/default.yaml --max-folds 1
```

GroupKFold：

```bash
python scripts/run_group_kfold.py --config configs/default.yaml --n-splits 5
```

评估 checkpoint：

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/groupkfold_1/best.pt
```

## 早停策略

默认配置：

```yaml
training:
  early_stopping:
    enabled: true
    monitor: "val_auc"
    mode: "max"
    patience: 8
    min_delta: 1.0e-4
```

含义：

- `monitor`：监控 `history.csv` 中的字段，例如 `val_auc`、`val_acc`、`val_loss`。
- `mode`：`max` 表示越大越好，`min` 表示越小越好。
- `patience`：连续多少个 epoch 没有超过 `min_delta` 的改善后停止。
- `min_delta`：判定“有效提升”的最小变化量。

`best.pt` 会按照早停监控指标保存，与停止条件保持一致。每个 epoch 的早停状态会写入 `history.csv`，包括 `early_stop_best`、`early_stop_bad_epochs`、`early_stop_improved`、`early_stopped`。

## 输出内容

- `outputs/<fold>/best.pt`
- `outputs/<fold>/history.csv`
- `outputs/loso_metrics.csv`
- `outputs/group_kfold_metrics.csv`
- `outputs/run.log`

## 防止数据泄露

LOSO 和 GroupKFold 都按 `subject_id` 切分。Gamma 清醒基准分布只使用当前训练 fold 的 alert 样本拟合，再应用到训练集和测试集窗口。不要先切窗口再随机拆分训练/测试集，否则同一受试者的窗口会同时进入两侧，导致跨人泛化指标虚高。

## 测试

```bash
pytest
```

当前测试覆盖 ADF 特征计算、JSONL 数据集窗口切分、landmarks 点数不一致处理，以及模型前向 shape。
