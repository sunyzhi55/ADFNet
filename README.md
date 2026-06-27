# ADFNet：任务完成驱动的跨受试者疲劳检测框架

ADFNet 基于 TF-GAM 与当前目录中的网络架构流程实现。模型将疲劳/专注状态建模为视线动态与任务目标轨迹之间的耦合质量，并通过 Gamma 分布偏移、Mamba 时序编码和 GRL 对抗解耦提升跨受试者泛化能力。

## 核心功能

- ADF 三通道特征：空间漂移、一阶差分、局部滑动均值。
- 任务模式过滤：支持 `easy`、`hard`、`all` 三种训练/评估模式。
- Gamma 清醒基准分布：只使用当前训练 fold 的 alert 样本拟合，避免测试主体泄露。
- 分布偏移分支：Mean Log-Likelihood、Wasserstein 距离、soft-DTW 距离。
- 真实 `mamba-ssm` 时序分支。
- GRL 对抗解耦：以 subject_id 分类为对抗目标（交叉熵 + GRL），剥离个人身份/注视习惯噪声，而非面部关键点回归（面部特征会泄漏疲劳标签）。
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

## 数据划分：验证集与测试集

为保证跨实验（对比/消融）的 fold 完全一致，并把「选轮用的验证集」与「最终报告用的测试集」严格分开，在 [configs/default.yaml](configs/default.yaml) 的 `split` 段配置：

```yaml
split:
  test_subjects: ["19", "20"]   # 被整体 hold-out 的测试被试 id；[] 表示不设独立测试集
  explicit_folds: [["01","05", "14", "19"], ["02","06", "10", "15"], ["07","11", "16", "20"], ["03","08", "12","17"], ["04","09", "13", "18"]]          # 可选：显式指定每 fold 的 val 被试，保证跨实验一致
```

- **`test_subjects`**：其中的被试在训练时被整体 hold-out，不进入任何 fold 的 train/val。训练结束后用 `evaluate.py --run-dir` 在这些被试上做最终评估。留空 `[]` 则不设独立测试集，LOSO/GroupKFold 在全部被试上进行（此时每个 fold 的 val 即作为评估集）。
- **`explicit_folds`**：形如 `[["01","02"], ["03"], ...]`，每个子列表是该 fold 的 val 被试；其余非测试被试作为该 fold 的 train。设为 `null` 时按 LOSO/GroupKFold 自动生成。显式指定可保证不同实验跑的是同一套 fold。

被试 id 必须与数据文件名前缀一致（字符串，如 `"01"`～`"20"`）。

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

### 单模型评估

读取已训练好的单个 checkpoint，在指定数据上评估并导出 CSV：

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

### 批量测试集评估（推荐）

训练时若在 `split.test_subjects` 中指定了 hold-out 测试被试，训练结束后用 `--run-dir` 指向训练输出目录，`evaluate.py` 会读取该目录下每个 fold 子目录的 `best.pt`，在测试集上逐 fold 评估并汇总：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --run-dir outputs/<timestamp>_ADFNet_Exp_loso/ \
  --task-mode hard
```

- 测试集只构建一次；每个 fold 用各自 checkpoint 里保存的 Gamma 参考分布与归一化参数对测试集重新做分布特征对齐，保证与训练时一致。
- 输出默认写到 `<run-dir>/test_metrics_<task_mode>.csv`，含每个 fold 一行以及末尾 `mean`/`std` 汇总行。
- 批量模式要求配置 `split.test_subjects` 非空；否则报错（此时应改用单模型 `--checkpoint` 模式）。

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
