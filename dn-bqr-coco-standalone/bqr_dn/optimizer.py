from __future__ import annotations

import torch

from .config import ExperimentConfig


def build_optimizer(model: torch.nn.Module, config: ExperimentConfig):
    backbone = []
    other = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".backbone." in name or name.startswith("backbone."):
            backbone.append(parameter)
        else:
            other.append(parameter)
    groups = [{"params": other, "lr": config.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": config.backbone_lr})
    optimizer = torch.optim.AdamW(
        groups, lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_drop_epoch, gamma=config.lr_drop_gamma
    )
    return optimizer, scheduler
