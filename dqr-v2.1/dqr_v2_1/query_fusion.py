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


class ContentAwareRegionGuidedDNFusion(nn.Module):
    """BQR-DN V2 prior attention plus sampled-feature compatibility."""

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        points_per_level: int,
        gate_bias: float,
        attention_dim: int,
        content_scale_init: float,
        attention_temperature: float,
        content_enabled: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.points_per_level = points_per_level
        self.attention_dim = attention_dim
        self.attention_temperature = attention_temperature
        self.content_enabled = content_enabled
        count = num_levels * points_per_level

        # Keep V2-compatible modules in exactly the same creation order.
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
        self._reset_v2_parameters(gate_bias)

        self.query_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, attention_dim, bias=False)
        )
        self.feature_key_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, attention_dim, bias=False)
        )
        initial_raw = math.log(content_scale_init / (1.0 - content_scale_init))
        self.content_scale_raw = nn.Parameter(torch.tensor(initial_raw, dtype=torch.float32))

    def _reset_v2_parameters(self, gate_bias: float) -> None:
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
        nn.init.normal_(self.feature_projection[-1].weight, std=1e-3)
        nn.init.zeros_(self.feature_projection[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    @staticmethod
    def _entropy(weights: torch.Tensor) -> torch.Tensor:
        flat = weights.flatten(1)
        return -(flat * flat.clamp_min(1e-8).log()).sum(dim=1)

    def _size_diagnostics(
        self,
        size_ids: torch.Tensor | None,
        active: torch.Tensor,
        prior_entropy: torch.Tensor,
        final_entropy: torch.Tensor,
        weights: torch.Tensor,
        alpha: torch.Tensor,
        delta_norm: torch.Tensor,
        offsets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if size_ids is None:
            return {}
        active_sizes = size_ids[active[:, 0], active[:, 1]]
        result = {}
        top1 = weights.flatten(1).amax(dim=1)
        offset_mean = offsets.abs().mean(dim=(1, 2, 3))
        for size_id, size_name in enumerate(("small", "medium", "large")):
            selected = active_sizes == size_id
            count = selected.sum()
            result[f"bqr_{size_name}_queries"] = count.detach().float()
            if not bool(count):
                continue
            for metric, values in (
                ("prior_entropy", prior_entropy),
                ("final_entropy", final_entropy),
                ("top1_attention", top1),
                ("gate_mean", alpha.squeeze(-1)),
                ("fusion_delta_norm", delta_norm),
                ("offset_abs_mean", offset_mean),
            ):
                result[f"bqr_{size_name}_{metric}_sum"] = values[selected].detach().sum()
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
                "bqr_content_scale": zero,
                "bqr_prior_entropy": zero,
                "bqr_final_entropy": zero,
                "bqr_content_logit_std": zero,
                "bqr_top1_attention": zero,
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

            prior_logits = self.attention_head(active_queries).reshape(
                -1, self.num_levels, self.points_per_level
            )
            content_scale = self.content_scale_raw.sigmoid()
            if self.content_enabled:
                query_keys = self.query_projection(active_queries)[:, None, None, :]
                feature_keys = self.feature_key_projection(sampled)
                content_logits = (query_keys * feature_keys).sum(dim=-1) / math.sqrt(self.attention_dim)
                final_logits = prior_logits + content_scale * content_logits
            else:
                content_logits = torch.zeros_like(prior_logits)
                final_logits = prior_logits
            prior_weights = prior_logits.flatten(1).softmax(dim=-1).reshape_as(prior_logits)
            weights = (final_logits / self.attention_temperature).flatten(1).softmax(dim=-1).reshape_as(final_logits)
            region_feature = (sampled * weights[..., None]).sum(dim=(1, 2))
            projected = self.feature_projection(region_feature)
            alpha = self.gate(torch.cat((active_queries, projected), dim=-1)).sigmoid()
            fused_active = active_queries + alpha * projected
            fused = query_float.clone()
            fused[active[:, 0], active[:, 1]] = fused_active

        prior_entropy = self._entropy(prior_weights)
        final_entropy = self._entropy(weights)
        delta_norm = (fused_active - active_queries).norm(dim=-1)
        diagnostics = {
            "bqr_valid_queries": torch.as_tensor(float(len(active)), device=queries.device),
            "bqr_gate_mean": alpha.detach().mean(),
            "bqr_offset_abs_mean": offsets.detach().abs().mean(),
            "bqr_attention_entropy": final_entropy.detach().mean(),
            "bqr_region_norm": region_feature.detach().norm(dim=-1).mean(),
            "bqr_fusion_delta_norm": delta_norm.detach().mean(),
            "bqr_content_scale": (content_scale.detach() if self.content_enabled else content_scale.detach() * 0),
            "bqr_prior_entropy": prior_entropy.detach().mean(),
            "bqr_final_entropy": final_entropy.detach().mean(),
            "bqr_content_logit_std": content_logits.detach().std(unbiased=False),
            "bqr_top1_attention": weights.detach().flatten(1).amax(dim=1).mean(),
        }
        diagnostics.update(
            self._size_diagnostics(
                size_ids, active, prior_entropy, final_entropy, weights,
                alpha, delta_norm, offsets,
            )
        )
        return fused.to(dtype=original_dtype), diagnostics
