from __future__ import annotations

from torch import nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class VigilanceHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, features):
        return self.net(features)


class SubjectDiscriminator(nn.Module):
    """对抗判别器：从融合特征预测 subject_id。

    与原先的 landmark 回归头不同，这里把对抗目标换成离散的身份分类
    （交叉熵 + GRL）。身份标签与疲劳标签正交（每被试 alert/sleep 各半），
    因此擦除身份不会系统性擦除疲劳信号；且视线特征确实携带个人注视习惯，
    对抗博弈成立。GRL 接在主任务之后，本头只在反向时通过反转梯度约束编码器。
    """

    def __init__(self, in_dim: int, n_subjects: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_subjects = int(n_subjects)
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, self.n_subjects),
        )

    def forward(self, features):
        return self.net(features)
