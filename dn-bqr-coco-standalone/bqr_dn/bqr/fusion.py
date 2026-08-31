from __future__ import annotations

import torch
import torch.nn as nn

from .dn_slots import active_clean_metadata
from .region_sampler import FixedBoxRegionSampler


class CleanGTRegionFusion(nn.Module):
    """Fuse simple multi-scale clean-GT evidence into valid DN content queries."""

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        grid_size: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.alpha = float(alpha)
        self.sampler = FixedBoxRegionSampler(hidden_dim, num_levels, grid_size)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        projection = self.feature_projection[-1]
        nn.init.normal_(projection.weight, std=1e-3)
        nn.init.zeros_(projection.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _parameter_zero(self, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.new_zeros(())
        for parameter in self.parameters():
            zero = zero + parameter.sum().to(reference.dtype) * 0.0
        return zero

    def forward(
        self,
        queries: torch.Tensor,
        valid_mask: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        targets: list[dict],
        *,
        diagnostics_enabled: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        active = valid_mask.nonzero(as_tuple=False)
        if len(active) == 0:
            return queries + self._parameter_zero(queries), {}

        original_dtype = queries.dtype
        with torch.autocast(device_type=queries.device.type, enabled=False):
            query_float = queries.float()
            memory_float = memory.float()
            active_queries = query_float[active[:, 0], active[:, 1]]
            clean_boxes, areas = active_clean_metadata(targets, active, queries.shape[1])
            sampled = self.sampler(
                memory_float,
                spatial_shapes,
                level_start_index,
                valid_ratios.float(),
                active[:, 0],
                clean_boxes,
            )
            per_level = sampled.mean(dim=2)
            region_feature = per_level.mean(dim=1)
            projected = self.feature_projection(region_feature)
            gate = self.gate(torch.cat((active_queries, projected), dim=-1)).sigmoid()
            fused_active = active_queries + self.alpha * gate * projected
            fused = query_float.clone()
            fused[active[:, 0], active[:, 1]] = fused_active

            diagnostics: dict[str, torch.Tensor] = {}
            if diagnostics_enabled:
                delta = fused_active - active_queries
                diagnostics = {
                    "bqr_valid_queries": torch.as_tensor(
                        float(len(active)), device=queries.device
                    ),
                    "bqr_gate_sum": gate.flatten().sum().detach(),
                    "bqr_region_norm_sum": region_feature.norm(dim=-1).sum().detach(),
                    "bqr_fusion_delta_norm_sum": delta.norm(dim=-1).sum().detach(),
                }
                for level in range(self.num_levels):
                    diagnostics[f"bqr_level{level}_norm_sum"] = (
                        per_level[:, level].norm(dim=-1).sum().detach()
                    )
                size_masks = {
                    "small": areas < 32.0**2,
                    "medium": (areas >= 32.0**2) & (areas < 96.0**2),
                    "large": areas >= 96.0**2,
                }
                for name, mask in size_masks.items():
                    diagnostics[f"bqr_{name}_queries"] = mask.sum().float().detach()

        return fused.to(original_dtype), diagnostics
