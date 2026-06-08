# ADFNet：任务完成驱动的跨受试者疲劳检测框架

ADFNet 基于 TF-GAM 与当前目录中的网络架构流程实现。模型将疲劳/专注状态建模为视线动态与任务目标轨迹之间的耦合质量，并通过 Gamma 分布偏移、Mamba 时序编码和 GRL 对抗解耦提升跨受试者泛化能力。

## 核心功能

- ADF 三通道特征：空间漂移、一阶差分、局部滑动均值。
- 任务模式过滤：支持只训练/评估 `easy`、只训练/评估 `hard`、或 `all` 混合任务。
- 清醒状态 Gamma 基准分布：只使用当前训练 fold 的 alert 样本拟合，避免测试主体泄露。
- 分布偏移分支：Mean Log-Likelihood、Wasserstein 距离、soft-DTW 距离。
- 真实 `mamba-ssm` 时序分支。
- GRL 对抗解耦 landmark 物理指纹。
- LOSO 与 GroupKFold 按 subject 物理隔离。
- 早停策略与 checkpoint 评估 CSV 导出。
- 混淆矩阵以数值列写入 CSV，避免矩阵对象/字符串导致解析问题。

## 数据格式

配置文件中的 `data.root` 指向 JSONL 数据目录。文件名格式：

```text
[subject_id]_[easy|hard]_[alert|sleep].jsonl
```

代码兼容历史标签名 `sleepy`，但推荐统一使用 `sleep`。

## 任务模式

在 [configs/default.yaml](configs/default.yaml) 中设置：

```yaml
data:
  task_mode: "all"  # all / easy / hard
```

也可以用命令行临时覆盖：

```bash
python scripts/run_loso.py --config configs/default.yaml --task-mode easy
python scripts/run_loso.py --config configs/default.yaml --task-mode hard
python scripts/run_loso.py --config configs/default.yaml --task-mode all
```

所有入口都支持 `--task-mode`：`train.py`、`run_loso.py`、`run_group_kfold.py`、`evaluate.py`。

## 训练

检查数据和窗口数量：

```bash
python scripts/train.py --config configs/default.yaml --dry-run --task-mode all
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
python scripts/run_group_kfold.py --config configs/default.yaml --n-splits 5 --task-mode all
```

## 独立评估

读取已训练好的模型，在指定测试数据上评估并导出 CSV：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/loso_01_all/best.pt \
  --data-root /path/to/test_jsonl \
  --task-mode all \
  --output-csv outputs/eval_external_all.csv
```

如果不传 `--data-root`，默认使用配置文件中的 `data.root`。如果不传 `--output-csv`，默认写入：

```text
outputs/eval_metrics_<task_mode>.csv
```

## CSV 指标字段

基础指标：

```text
auc, acc, f1, precision, recall
```

混淆矩阵不会以 `[[tn, fp], [fn, tp]]` 这种对象形式写入 CSV，而是拆成四个稳定数值列：

```text
cm_tn, cm_fp, cm_fn, cm_tp
```

训练过程的 [history.csv](outputs/history.csv) 会分别包含：

```text
train_cm_tn, train_cm_fp, train_cm_fn, train_cm_tp
val_cm_tn, val_cm_fp, val_cm_fn, val_cm_tp
```

独立评估 CSV 会包含：

```text
checkpoint, data_root, task_mode, windows, auc, acc, f1, precision, recall, cm_tn, cm_fp, cm_fn, cm_tp
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

`best.pt` 按早停监控指标保存。每个 epoch 的早停状态写入 `history.csv`。

## 加速相关

当前实现会在每个 fold 开始前预计算分布统计特征，避免每个 batch 重复计算 soft-DTW。可调参数：

```yaml
training:
  batch_size: 128
  num_workers: 8
  pin_memory: true
  persistent_workers: true

distribution:
  soft_dtw_reference_samples: 64
```

如果 CPU 仍然慢，可以把 `soft_dtw_reference_samples` 降到 `32`。

## 输出

- `outputs/<fold>_<task_mode>/best.pt`
- `outputs/<fold>_<task_mode>/history.csv`
- `outputs/loso_metrics_<task_mode>.csv`
- `outputs/group_kfold_metrics_<task_mode>.csv`
- `outputs/eval_metrics_<task_mode>.csv`

## 测试

```bash
pytest
```
