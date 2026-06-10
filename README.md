# ADFNet：任务完成驱动的跨受试者疲劳检测框架

ADFNet 基于 TF-GAM 与当前目录中的网络架构流程实现。模型将疲劳/专注状态建模为视线动态与任务目标轨迹之间的耦合质量，并通过 Gamma 分布偏移、Mamba 时序编码和 GRL 对抗解耦提升跨受试者泛化能力。

## 核心功能

- ADF 三通道特征：空间漂移、一阶差分、局部滑动均值。
- 任务模式过滤：支持 `easy`、`hard`、`all` 三种训练/评估模式。
- Gamma 清醒基准分布：只使用当前训练 fold 的 alert 样本拟合，避免测试主体泄露。
- 分布偏移分支：Mean Log-Likelihood、Wasserstein 距离、soft-DTW 距离。
- 真实 `mamba-ssm` 时序分支。
- GRL 对抗解耦 landmark 物理指纹。
- LOSO 与 GroupKFold 按 subject 物理隔离。
- 早停、最佳轮选择、checkpoint 评估和 CSV 导出。

## 任务模式

在 [configs/default.yaml](configs/default.yaml) 中设置：

```yaml
data:
  task_mode: "easy"  # all / easy / hard
```

也可以用命令行覆盖：

```bash
python scripts/run_loso.py --config configs/default.yaml --task-mode easy
python scripts/run_loso.py --config configs/default.yaml --task-mode hard
python scripts/run_loso.py --config configs/default.yaml --task-mode all
```

所有入口都支持 `--task-mode`：`train.py`、`run_loso.py`、`run_group_kfold.py`、`evaluate.py`。

## 训练

检查数据：

```bash
python scripts/train.py --config configs/default.yaml --dry-run --task-mode easy
```

LOSO：

```bash
python scripts/run_loso.py --config configs/default.yaml --task-mode easy
python scripts/run_loso.py --config configs/default.yaml --task-mode hard
python scripts/run_loso.py --config configs/default.yaml --task-mode all
```

调试单个 fold：

```bash
python scripts/run_loso.py --config configs/default.yaml --task-mode all --max-folds 1
```

GroupKFold：

```bash
python scripts/run_group_kfold.py --config configs/default.yaml --n-splits 5 --task-mode easy
```

## 最佳轮选择

训练过程会完整写入每个 fold 的 `history.csv`。训练结束后，代码不会再使用 `history[-1]` 作为最终结果，而是按照配置中的 `training.result_selection` 从所有 epoch 中选出最佳一轮，并写入：

```text
outputs/<fold>_<task_mode>/final_metrics.csv
```

默认配置：

```yaml
training:
  result_selection:
    monitor: "val_f1"
    mode: "max"
    min_delta: 0.0
```

含义：

- `monitor`：用于选择最终最佳轮的字段，例如 `val_f1`、`val_auc`、`val_acc`、`val_loss`。
- `mode`：`max` 表示越大越好，`min` 表示越小越好。
- `min_delta`：只有超过该变化量才认为是更好的结果。

`best.pt` 也按 `result_selection` 保存，保证模型权重和最终 CSV 是同一轮结果。

## 早停策略

早停配置仍然独立存在：

```yaml
training:
  early_stopping:
    enabled: true
    monitor: "val_f1"
    mode: "max"
    patience: 1000
    min_delta: 1.0e-4
```

如果没有配置 `training.result_selection`，代码会默认沿用 `early_stopping.monitor` 和 `early_stopping.mode` 作为最佳轮选择标准。

## CSV 输出

每个 fold：

```text
outputs/<fold>_<task_mode>/history.csv
outputs/<fold>_<task_mode>/final_metrics.csv
outputs/<fold>_<task_mode>/best.pt
```

LOSO 汇总：

```text
outputs/loso_metrics_<task_mode>.csv
```

GroupKFold 汇总：

```text
outputs/group_kfold_metrics_<task_mode>.csv
```

这些汇总 CSV 中每个 fold 的行来自 `final_metrics.csv` 对应的最佳 epoch，而不是最后一个 epoch。

## 指标字段

基础指标：

```text
auc, acc, f1, precision, recall
```

混淆矩阵被拆成稳定数值列：

```text
cm_tn, cm_fp, cm_fn, cm_tp
```

训练过程中的字段包含：

```text
train_auc, train_acc, train_f1, train_precision, train_recall
train_cm_tn, train_cm_fp, train_cm_fn, train_cm_tp
val_auc, val_acc, val_f1, val_precision, val_recall
val_cm_tn, val_cm_fp, val_cm_fn, val_cm_tp
```

`final_metrics.csv` 额外包含：

```text
best_epoch, selection_monitor, selection_mode, selection_value
```

## 独立评估

读取已训练好的模型，在指定测试数据上评估并导出 CSV：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/loso_01_easy/best.pt \
  --data-root /path/to/test_jsonl \
  --task-mode easy \
  --output-csv outputs/eval_external_easy.csv
```

如果不传 `--data-root`，默认使用配置文件中的 `data.root`。如果不传 `--output-csv`，默认写入：

```text
outputs/eval_metrics_<task_mode>.csv
```

## 加速相关

当前实现会在每个 fold 开始前预计算分布统计特征，避免每个 batch 重复计算 soft-DTW。可调参数：

```yaml
training:
  batch_size: 128
  num_workers: 16
  pin_memory: true
  persistent_workers: true

distribution:
  soft_dtw_reference_samples: 64
```

如果 CPU 仍然慢，可以把 `soft_dtw_reference_samples` 降到 `32`。

## 测试

```bash
pytest
```
