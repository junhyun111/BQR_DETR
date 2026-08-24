from __future__ import annotations

import torch
import torch.nn as nn

from .config import ExperimentConfig


class ResearchCriterion(nn.Module):
    def __init__(self, official_criterion: nn.Module, config: ExperimentConfig) -> None:
        super().__init__()
        self.official = official_criterion
        self.weight_dict = dict(official_criterion.weight_dict)
        if config.method == "gt_guided_aux":
            self.weight_dict.update(
                {
                    "loss_aux_bbox": config.aux_weight * config.aux_l1_coef,
                    "loss_aux_giou": config.aux_weight * config.aux_giou_coef,
                }
            )

    def forward(self, model, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        losses, indices_list = self.official(outputs, targets, return_indices=True)
        if model.auxiliary is not None:
            num_boxes = max(sum(len(target["labels"]) for target in targets), 1)
            losses.update(model.auxiliary_losses(targets, indices_list[-1], float(num_boxes)))
        return losses

    def weighted_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        weighted = [losses[name] * weight for name, weight in self.weight_dict.items() if name in losses]
        if not weighted:
            raise RuntimeError("No weighted training losses were produced")
        return torch.stack(weighted).sum()
