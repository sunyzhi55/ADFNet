import numpy as np
from torch.nn import init
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim import lr_scheduler

def get_optimizer(optimizer_name, parameters, **kwargs):
    """
    Get optimizer by name

    Args:
        optimizer_name: 优化器名称（如 'Adam', 'SGD', 'RMSprop', 'Adadelta'）
        parameters: 模型参数
        lr: 学习率
        weight_decay: 权重衰减
    Returns:
        optimizer: 对应的优化器实例
    """
    lr = kwargs.get('lr', 1e-3)
    weight_decay = kwargs.get('weight_decay', 0)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = kwargs.get('momentum', 0.9)
        optimizer = optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Please implement it in get_optimizer function.")

    return optimizer

def get_scheduler(optimizer, cfg, train_loader=None):
    """
    定义学习率调度器（scheduler）

    Args:
        optimizer: torch.optim 优化器
        cfg: 参数配置字典，需包含 lr_policy 等属性
        train_loader: 仅在 OneCycleLR 时需要，用于计算 steps_per_epoch

    Returns:
        scheduler: 对应的学习率调度器
    """

    policy = cfg["training"]["lr_policy"]

    if policy == 'none' or policy is None:
        return None
    
    total_epochs = cfg["training"]["epochs"]
    if policy == 'lambda':
        warm_epochs = getattr(cfg["training"], 'niter', max(total_epochs // 2, 1))
        decay_epochs = getattr(cfg["training"], 'niter_decay', max(total_epochs - warm_epochs, 1))

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - warm_epochs) / float(decay_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif policy == 'step':
        step_size = getattr(cfg["training"], 'lr_decay_iters', getattr(cfg["training"], 'niter', 30))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,        # 温和衰减：每次减半
            patience=10,      # 连续10轮不提升再降lr
            threshold=1e-4,   # 极小阈值，贴合loss正常波动
            min_lr=1e-6,      # 限制最低学习率，防止卡死
            cooldown=3        # 降完lr后，冷却3轮再重新计数
        )

    elif policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = cfg["training"].get("cos_T0", 10),
            T_mult = cfg["training"].get("cos_Tmult", 3),
            eta_min = cfg["training"].get("cos_eta_min", 1e-5)
        )

    elif policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg["training"]["lr_decay"])

    elif policy == 'onecycle':
        if train_loader is None:
            raise ValueError("❌ OneCycleLR 策略需要传入 train_loader 参数以计算 steps_per_epoch")
        # scheduler = lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=cfg["training"]["lr"] * 5,  # 峰值学习率（通常为初始lr的3~10倍）
        #     steps_per_epoch=len(train_loader),
        #     epochs=cfg["training"]["epochs"],
        #     anneal_strategy='cos',  # 余弦退火
        #     pct_start=0.1,          # 10% 的时间用于 warm-up
        #     # div_factor=25.0,        # 初始学习率 = max_lr / div_factor
        #     # final_div_factor=1e4,   # 最低学习率 = max_lr / final_div_factor
        #     # three_phase=False       # 可选：是否三阶段策略
        # )
        div_factor = 5.0
        max_lr = cfg["training"]["lr"] * div_factor  # max_lr = base_lr * 10
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(train_loader),
            epochs=cfg["training"]["epochs"],
            anneal_strategy='cos',
            pct_start=0.25,        # 改为 25% 步数升温，更稳
            div_factor=div_factor, # 初始 lr = max_lr / 10 = base_lr
            final_div_factor=100   # 末端最低 lr = max_lr / 100
        )

    else:
        raise NotImplementedError(f'learning rate policy [{policy}] is not implemented')

    return scheduler
