from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleRegionSampler(nn.Module):
    def __init__(self, hidden_dim: int, num_levels: int, points_per_level: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.points_per_level = points_per_level

    def forward(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        batch_indices: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        samples = []
        for level in range(self.num_levels):
            height = int(spatial_shapes[level, 0].item())
            width = int(spatial_shapes[level, 1].item())
            start = int(level_start_index[level].item())
            level_memory = memory[:, start : start + height * width]
            level_map = level_memory.reshape(
                memory.shape[0], height, width, self.hidden_dim
            ).permute(0, 3, 1, 2)
            object_maps = level_map[batch_indices]
            grid = points[:, level] * valid_ratios[batch_indices, level, None, :]
            grid = grid.mul(2.0).sub(1.0).reshape(
                len(batch_indices), 1, self.points_per_level, 2
            )
            sampled = F.grid_sample(
                object_maps,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            samples.append(sampled.squeeze(2).transpose(1, 2))
        return torch.stack(samples, dim=1)
