from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dn_slots import active_slot_metadata
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
            angles *= 2 * math.pi / self.points_per_level
            prior = torch.stack((angles.cos(), angles.sin()), dim=-1) * 0.25
        raw_prior = torch.atanh((prior / 0.5).clamp(-0.999, 0.999))
        self.offset_head.layers[-1].bias.data.copy_(
            raw_prior.repeat(self.num_levels, 1).flatten()
        )
        nn.init.zeros_(self.attention_head.layers[-1].weight)
        nn.init.zeros_(self.attention_head.layers[-1].bias)
        nn.init.normal_(self.feature_projection[-1].weight, std=1e-3)
        nn.init.zeros_(self.feature_projection[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def _parameter_zero(self, queries: torch.Tensor) -> torch.Tensor:
        # Keep every V2 parameter in the graph when an augmented micro-batch
        # happens to contain no valid boxes. This avoids DDP unused-parameter
        # mismatches without enabling expensive find_unused_parameters.
        zero = queries.sum() * 0.0
        for parameter in self.parameters():
            if parameter.requires_grad:
                zero = zero + parameter.reshape(-1)[0] * 0.0
        return zero

    def _diagnostics(
        self,
        *,
        targets: list[dict],
        active: torch.Tensor,
        pad_size: int,
        clean_boxes: torch.Tensor,
        areas: torch.Tensor,
        positive: torch.Tensor,
        points: torch.Tensor,
        weights: torch.Tensor,
        alpha: torch.Tensor,
        offsets: torch.Tensor,
        region_feature: torch.Tensor,
        fused_active: torch.Tensor,
        active_queries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del targets, pad_size
        per_query_entropy = -(
            weights.flatten(1) * weights.flatten(1).clamp_min(1e-8).log()
        ).sum(dim=1)
        per_query_offset = offsets.abs().mean(dim=(1, 2, 3))
        per_query_region = region_feature.norm(dim=-1)
        per_query_delta = (fused_active - active_queries).norm(dim=-1)
        inside = (
            (points - clean_boxes[:, None, None, :2]).abs()
            <= clean_boxes[:, None, None, 2:] * 0.5
        ).all(dim=-1)
        per_query_inside = inside.float().mean(dim=(1, 2))
        level_mass = weights.sum(dim=2)

        result = {
            "bqr_valid_queries": torch.as_tensor(float(len(active)), device=active.device),
            "bqr_gate_sum": alpha.flatten().sum(),
            "bqr_offset_abs_sum": per_query_offset.sum(),
            "bqr_attention_entropy_sum": per_query_entropy.sum(),
            "bqr_region_norm_sum": per_query_region.sum(),
            "bqr_fusion_delta_norm_sum": per_query_delta.sum(),
        }
        size_masks = {
            "small": areas < 32.0**2,
            "medium": (areas >= 32.0**2) & (areas < 96.0**2),
            "large": areas >= 96.0**2,
        }
        for size_name, size_mask in size_masks.items():
            count = size_mask.sum()
            result[f"bqr_{size_name}_queries"] = count.float()
            result[f"bqr_{size_name}_gt_inside_sum"] = per_query_inside[size_mask].sum()
            for level in range(self.num_levels):
                result[f"bqr_{size_name}_level{level}_attention_sum"] = level_mass[
                    size_mask, level
                ].sum()
            for polarity_name, polarity_mask in (
                ("positive", positive),
                ("negative", ~positive),
            ):
                selected = size_mask & polarity_mask
                result[f"bqr_{size_name}_{polarity_name}_queries"] = selected.sum().float()
                result[f"bqr_{size_name}_{polarity_name}_gt_inside_sum"] = (
                    per_query_inside[selected].sum()
                )
        return {name: value.detach() for name, value in result.items()}

    def forward(
        self,
        queries: torch.Tensor,
        noisy_boxes: torch.Tensor,
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
            active_boxes = noisy_boxes.float()[active[:, 0], active[:, 1]]
            offsets = self.offset_head(active_queries).reshape(
                -1, self.num_levels, self.points_per_level, 2
            ).tanh() * 0.5
            points = (
                active_boxes[:, None, None, :2]
                + active_boxes[:, None, None, 2:] * offsets
            ).clamp(0.0, 1.0)
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

            diagnostics = {}
            if diagnostics_enabled:
                clean_boxes, areas, positive = active_slot_metadata(
                    targets, active, queries.shape[1]
                )
                diagnostics = self._diagnostics(
                    targets=targets,
                    active=active,
                    pad_size=queries.shape[1],
                    clean_boxes=clean_boxes,
                    areas=areas,
                    positive=positive,
                    points=points,
                    weights=weights,
                    alpha=alpha,
                    offsets=offsets,
                    region_feature=region_feature,
                    fused_active=fused_active,
                    active_queries=active_queries,
                )
        return fused.to(dtype=original_dtype), diagnostics
