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


class ScaleAwareRegionGuidedDNFusion(nn.Module):
    """BQR-DN V2 with five-point sampling and a geometric level prior."""

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        points_per_level: int,
        gate_bias: float,
        scale_aware: bool,
        target_cells: float,
        scale_sigma: float,
        scale_weight: float,
        scale_logit_floor: float,
    ) -> None:
        super().__init__()
        if num_levels != 4:
            raise ValueError(f"BQR-DN V3 requires four feature levels, got {num_levels}")
        if points_per_level != 5:
            raise ValueError(
                f"BQR-DN V3 requires five points per level, got {points_per_level}"
            )
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.points_per_level = points_per_level
        self.scale_aware = scale_aware
        self.target_cells = target_cells
        self.scale_sigma = scale_sigma
        self.scale_weight = scale_weight
        self.scale_logit_floor = scale_logit_floor
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
        prior = torch.tensor(
            [
                [0.00, 0.00],
                [-0.25, -0.25],
                [0.25, -0.25],
                [-0.25, 0.25],
                [0.25, 0.25],
            ],
            dtype=torch.float32,
        )
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

    @staticmethod
    def _entropy(weights: torch.Tensor) -> torch.Tensor:
        flat = weights.flatten(1)
        return -(flat * flat.clamp_min(1e-8).log()).sum(dim=1)

    def compute_scale_prior(
        self,
        boxes: torch.Tensor,
        spatial_shapes: torch.Tensor,
        valid_ratios: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return stabilized level logits from noisy-box footprint in valid cells."""
        shapes_wh = spatial_shapes.to(device=boxes.device, dtype=boxes.dtype).flip(-1)
        valid_shapes_wh = shapes_wh.unsqueeze(0) * valid_ratios[batch_indices].to(
            dtype=boxes.dtype
        )
        object_cells_wh = boxes[:, None, 2:] * valid_shapes_wh
        footprint = object_cells_wh.prod(dim=-1).clamp_min(1e-12).sqrt()
        log_distance = footprint.log2() - math.log2(self.target_cells)
        scale_logits = -0.5 * (log_distance / self.scale_sigma).square()
        scale_logits = scale_logits - scale_logits.max(dim=1, keepdim=True).values
        return scale_logits.clamp_min(self.scale_logit_floor)

    def _size_diagnostics(
        self,
        size_ids: torch.Tensor | None,
        active: torch.Tensor,
        query_entropy: torch.Tensor,
        final_entropy: torch.Tensor,
        scale_entropy: torch.Tensor,
        weights: torch.Tensor,
        scale_logits: torch.Tensor,
        alpha: torch.Tensor,
        delta_norm: torch.Tensor,
        offsets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if size_ids is None:
            return {}
        active_sizes = size_ids[active[:, 0], active[:, 1]]
        result: dict[str, torch.Tensor] = {}
        top1 = weights.flatten(1).amax(dim=1)
        center_mass = weights[:, :, 0].sum(dim=1)
        level_mass = weights.sum(dim=2)
        offset_mean = offsets.abs().mean(dim=(1, 2, 3))
        scale_span = scale_logits.amax(dim=1) - scale_logits.amin(dim=1)
        for size_id, size_name in enumerate(("small", "medium", "large")):
            selected = active_sizes == size_id
            count = selected.sum()
            result[f"bqr_{size_name}_queries"] = count.detach().float()
            if not bool(count):
                continue
            for metric, values in (
                ("query_attention_entropy", query_entropy),
                ("final_attention_entropy", final_entropy),
                ("scale_prior_entropy", scale_entropy),
                ("scale_logit_span", scale_span),
                ("top1_attention", top1),
                ("center_attention", center_mass),
                ("gate_mean", alpha.squeeze(-1)),
                ("fusion_delta_norm", delta_norm),
                ("offset_abs_mean", offset_mean),
            ):
                result[f"bqr_{size_name}_{metric}_sum"] = (
                    values[selected].detach().sum()
                )
            for level in range(self.num_levels):
                result[f"bqr_{size_name}_level_{level}_attention_sum"] = (
                    level_mass[selected, level].detach().sum()
                )
        return result

    def forward(
        self,
        queries: torch.Tensor,
        noisy_boxes: torch.Tensor,
        valid_mask: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        size_ids: torch.Tensor | None = None,
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
                "bqr_query_attention_entropy": zero,
                "bqr_final_attention_entropy": zero,
                "bqr_scale_prior_entropy": zero,
                "bqr_scale_logit_span": zero,
                "bqr_center_attention": zero,
                "bqr_top1_attention": zero,
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

            query_logits = self.attention_head(active_queries).reshape(
                -1, self.num_levels, self.points_per_level
            )
            scale_logits = self.compute_scale_prior(
                active_boxes,
                spatial_shapes,
                valid_ratios.float(),
                active[:, 0],
            )
            if self.scale_aware:
                final_logits = query_logits + self.scale_weight * scale_logits[..., None]
            else:
                final_logits = query_logits
            query_weights = query_logits.flatten(1).softmax(dim=-1).reshape_as(query_logits)
            weights = final_logits.flatten(1).softmax(dim=-1).reshape_as(final_logits)
            scale_level_weights = scale_logits.softmax(dim=1)
            region_feature = (sampled * weights[..., None]).sum(dim=(1, 2))
            projected = self.feature_projection(region_feature)
            alpha = self.gate(torch.cat((active_queries, projected), dim=-1)).sigmoid()
            fused_active = active_queries + alpha * projected
            fused = query_float.clone()
            fused[active[:, 0], active[:, 1]] = fused_active

        query_entropy = self._entropy(query_weights)
        final_entropy = self._entropy(weights)
        scale_entropy = self._entropy(scale_level_weights)
        delta_norm = (fused_active - active_queries).norm(dim=-1)
        scale_span = scale_logits.amax(dim=1) - scale_logits.amin(dim=1)
        diagnostics = {
            "bqr_valid_queries": torch.as_tensor(float(len(active)), device=queries.device),
            "bqr_gate_mean": alpha.detach().mean(),
            "bqr_offset_abs_mean": offsets.detach().abs().mean(),
            "bqr_attention_entropy": final_entropy.detach().mean(),
            "bqr_query_attention_entropy": query_entropy.detach().mean(),
            "bqr_final_attention_entropy": final_entropy.detach().mean(),
            "bqr_scale_prior_entropy": scale_entropy.detach().mean(),
            "bqr_scale_logit_span": scale_span.detach().mean(),
            "bqr_center_attention": weights.detach()[:, :, 0].sum(dim=1).mean(),
            "bqr_top1_attention": weights.detach().flatten(1).amax(dim=1).mean(),
            "bqr_region_norm": region_feature.detach().norm(dim=-1).mean(),
            "bqr_fusion_delta_norm": delta_norm.detach().mean(),
        }
        diagnostics.update(
            self._size_diagnostics(
                size_ids,
                active,
                query_entropy,
                final_entropy,
                scale_entropy,
                weights,
                scale_logits,
                alpha,
                delta_norm,
                offsets,
            )
        )
        return fused.to(dtype=original_dtype), diagnostics
