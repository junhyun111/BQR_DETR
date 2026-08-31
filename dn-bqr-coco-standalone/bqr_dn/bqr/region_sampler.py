from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedBoxRegionSampler(nn.Module):
    """Sample a fixed grid over each clean box from every encoder level."""

    def __init__(self, hidden_dim: int, num_levels: int, grid_size: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.grid_size = grid_size
        coordinates = (
            torch.arange(grid_size, dtype=torch.float32).add(0.5).div(grid_size).sub(0.5)
        )
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        offsets = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        self.register_buffer("box_offsets", offsets, persistent=False)

    @staticmethod
    def _positions_within_batch(
        batch_indices: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, int]:
        positions = torch.empty_like(batch_indices)
        maximum = 0
        for batch_index in range(batch_size):
            selected = batch_indices.eq(batch_index).nonzero(as_tuple=False).flatten()
            positions[selected] = torch.arange(len(selected), device=batch_indices.device)
            maximum = max(maximum, len(selected))
        return positions, maximum

    def points_for_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        points = boxes[:, None, :2] + boxes[:, None, 2:] * self.box_offsets[None]
        return points.clamp(0.0, 1.0)

    def forward(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        batch_indices: torch.Tensor,
        clean_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[active_dn, levels, grid_points, hidden]`` samples."""
        batch_size = memory.shape[0]
        positions, max_active = self._positions_within_batch(batch_indices, batch_size)
        points = self.points_for_boxes(clean_boxes)
        point_count = self.grid_size**2
        samples = []
        for level in range(self.num_levels):
            height = int(spatial_shapes[level, 0].item())
            width = int(spatial_shapes[level, 1].item())
            start = int(level_start_index[level].item())
            level_map = memory[:, start : start + height * width].reshape(
                batch_size, height, width, self.hidden_dim
            ).permute(0, 3, 1, 2)

            active_grid = points * valid_ratios[batch_indices, level, None, :]
            grid = torch.zeros(
                batch_size,
                max_active,
                point_count,
                2,
                dtype=active_grid.dtype,
                device=active_grid.device,
            )
            grid[batch_indices, positions] = active_grid.mul(2.0).sub(1.0)
            sampled_all = F.grid_sample(
                level_map,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled_all[batch_indices, :, positions, :].transpose(1, 2)
            samples.append(sampled)
        return torch.stack(samples, dim=1)
