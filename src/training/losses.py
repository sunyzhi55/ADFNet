from __future__ import annotations

import math

import torch
from torch import nn


class ADFNetLoss(nn.Module):
    """主任务 BCE + 身份对抗交叉熵。

    与旧版（bce + grl_lambda * landmark_mse）的区别：
      - 对抗目标由面部回归换成 subject_id 分类，避免面部特征携带的疲劳标签泄漏。
      - 解耦 λ：GRL 内部单独施加调度 λ，loss 里只用常数 ``loss_weight``，
        不再把同一个 λ 乘两遍（旧实现实际为 λ²）。
      - 未知 subject（验证集/留出被试）映射为 -1，由 CrossEntropyLoss(ignore_index=-1)
        忽略；全部为 -1 时（如纯验证）adv_ce 置 0，保证 val 不会出 nan。
    """

    def __init__(self, loss_weight: float = 1.0, ignore_index: int = -1) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        # self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.75]))
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(outputs["vigilance_logit"].device)
        bce_loss = self.bce(outputs["vigilance_logit"], labels)
        subject_ids = subject_ids.reshape(-1).long()
        if bool((subject_ids >= 0).any()):
            adv_ce = self.ce(outputs["subject_logit"], subject_ids)
        else:
            # 验证集全部为未知被试：对抗项无意义，置零以避免 mean-of-empty 的 nan。
            adv_ce = outputs["subject_logit"].new_zeros(())
        total = bce_loss + self.loss_weight * adv_ce
        return {
            "loss": total,
            "bce": bce_loss.detach(),
            "adv_ce": adv_ce.detach(),
        }


def grl_lambda_schedule(
    epoch: int,
    total_epochs: int,
    max_lambda: float = 1.0,
    warmup_epochs: int = 0,
    slope: float = 10.0,
) -> float:
    """GRL 系数调度：warmup 期内为 0，之后按 sigmoid 从 0 平滑升到 ``max_lambda``。

    相比旧版直接 0→1，这里上限可封顶（``max_lambda``），并支持先让主任务特征
    稳定的 warmup，避免一开始对抗梯度就压垮 BCE。
    """
    if max_lambda <= 0.0:
        return 0.0
    if epoch < warmup_epochs:
        return 0.0
    eff_total = max(total_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / float(eff_total - 1) if eff_total > 1 else 1.0
    progress = min(max(progress, 0.0), 1.0)
    base = 2.0 / (1.0 + math.exp(-slope * progress)) - 1.0  # 0 -> 1
    return float(max_lambda) * base
