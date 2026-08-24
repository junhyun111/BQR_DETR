from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .region_sampler import MultiScaleRegionSampler


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            (nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, output_dim))
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers[1](F.gelu(self.layers[0](tensor)))


class RegionGuidedDNFusion(nn.Module):
    """Fuse noisy-box region evidence into valid DINO DN content queries."""

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        points_per_level: int,
        gate_bias: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.points_per_level = points_per_level
        count = num_levels * points_per_level
        self.offset_head = MLP(hidden_dim, hidden_dim, count * 2)
        self.attention_head = MLP(hidden_dim, hidden_dim, count)
        self.sampler = MultiScaleRegionSampler(hidden_dim, num_levels, points_per_level)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters(gate_bias)

    def _reset_parameters(self, gate_bias: float) -> None:
        nn.init.zeros_(self.offset_head.layers[-1].weight)
        if self.points_per_level == 4:
            prior = torch.tensor(
                [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]],
                dtype=torch.float32,
            )
        else:
            angles = torch.arange(self.points_per_level, dtype=torch.float32)
            angles = angles * (2 * math.pi / self.points_per_level)
            prior = torch.stack((angles.cos(), angles.sin()), dim=-1) * 0.25
        raw_prior = torch.atanh((prior / 0.5).clamp(-0.999, 0.999))
        self.offset_head.layers[-1].bias.data.copy_(
            raw_prior.repeat(self.num_levels, 1).flatten()
        )
        nn.init.zeros_(self.attention_head.layers[-1].weight)
        nn.init.zeros_(self.attention_head.layers[-1].bias)
        # Start close to the official DN path without blocking first-step
        # gradients to the sampler and aggregation modules.
        nn.init.normal_(self.feature_projection[-1].weight, std=1e-3)
        nn.init.zeros_(self.feature_projection[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(
        self,
        queries: torch.Tensor,
        noisy_boxes: torch.Tensor,
        valid_mask: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Inputs use batch-first query/memory and normalized cxcywh boxes."""
        active = valid_mask.nonzero(as_tuple=False)
        if len(active) == 0:
            zero = memory.sum().detach() * 0.0
            return queries, {
                "bqr_valid_queries": zero,
                "bqr_gate_mean": zero,
                "bqr_offset_abs_mean": zero,
                "bqr_attention_entropy": zero,
                "bqr_region_norm": zero,
                "bqr_fusion_delta_norm": zero,
            }

        original_dtype = queries.dtype
        with torch.autocast(device_type=queries.device.type, enabled=False):
            query_float = queries.float()
            memory_float = memory.float()
            active_queries = query_float[active[:, 0], active[:, 1]]
            active_boxes = noisy_boxes.float()[active[:, 0], active[:, 1]]
            offsets = self.offset_head(active_queries).reshape(
                -1, self.num_levels, self.points_per_level, 2
            )
            offsets = offsets.tanh() * 0.5
            points = active_boxes[:, None, None, :2] + active_boxes[:, None, None, 2:] * offsets
            points = points.clamp(0.0, 1.0)
            sampled = self.sampler(
                memory_float,
                spatial_shapes,
                level_start_index,
                valid_ratios.float(),
                active[:, 0],
                points,
            )
            weights = self.attention_head(active_queries).reshape(
                -1, self.num_levels, self.points_per_level
            )
            weights = weights.flatten(1).softmax(dim=-1).reshape_as(weights)
            region_feature = (sampled * weights[..., None]).sum(dim=(1, 2))
            projected = self.feature_projection(region_feature)
            alpha = self.gate(torch.cat((active_queries, projected), dim=-1)).sigmoid()
            fused_active = active_queries + alpha * projected
            fused = query_float.clone()
            fused[active[:, 0], active[:, 1]] = fused_active

        entropy = -(
            weights.flatten(1) * weights.flatten(1).clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        diagnostics = {
            "bqr_valid_queries": torch.as_tensor(float(len(active)), device=queries.device),
            "bqr_gate_mean": alpha.detach().mean(),
            "bqr_offset_abs_mean": offsets.detach().abs().mean(),
            "bqr_attention_entropy": entropy.detach(),
            "bqr_region_norm": region_feature.detach().norm(dim=-1).mean(),
            "bqr_fusion_delta_norm": (fused_active.detach() - active_queries.detach()).norm(dim=-1).mean(),
        }
        return fused.to(dtype=original_dtype), diagnostics
