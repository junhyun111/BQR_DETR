from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig
from .upstream import (
    ensure_dqr_v2_1_imports,
    ensure_dqr_v2_imports,
    ensure_dqr_v3_1_imports,
    ensure_dqr_v3_imports,
    ensure_upstream_imports,
)


ensure_upstream_imports()
from util import box_ops  # noqa: E402


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(layers)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            tensor = F.relu(layer(tensor)) if index < len(self.layers) - 1 else layer(tensor)
        return tensor


@dataclass
class EncoderContext:
    memory: torch.Tensor
    spatial_shapes: torch.Tensor
    level_start_index: torch.Tensor
    valid_ratios: torch.Tensor
    padding_mask: torch.Tensor
    query_features: torch.Tensor


class GTGuidedAuxiliary(nn.Module):
    def __init__(self, hidden_dim: int, num_levels: int, points_per_level: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.points_per_level = points_per_level
        count = num_levels * points_per_level
        self.offset_head = MLP(hidden_dim, hidden_dim, count * 2, 2)
        self.attention_head = MLP(hidden_dim, hidden_dim, count, 2)
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
        self.box_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.offset_head.layers[-1].weight)
        prior = torch.tensor(
            [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]],
            dtype=torch.float32,
        )
        if self.points_per_level != 4:
            angles = torch.arange(self.points_per_level, dtype=torch.float32) * (2 * math.pi / self.points_per_level)
            prior = torch.stack((angles.cos(), angles.sin()), dim=-1) * 0.25
        raw_prior = torch.atanh((prior / 0.5).clamp(-0.999, 0.999))
        self.offset_head.layers[-1].bias.data.copy_(raw_prior.repeat(self.num_levels, 1).flatten())
        nn.init.zeros_(self.attention_head.layers[-1].weight)
        nn.init.zeros_(self.attention_head.layers[-1].bias)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.box_head.layers[-1].weight)
        nn.init.zeros_(self.box_head.layers[-1].bias)

    def _matched_inputs(
        self,
        context: EncoderContext,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = context.query_features.device
        batch_parts: list[torch.Tensor] = []
        query_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            if len(query_indices) == 0:
                continue
            batch_parts.append(torch.full((len(query_indices),), batch_index, dtype=torch.long, device=device))
            query_parts.append(query_indices.to(device=device, dtype=torch.long))
            target_parts.append(targets[batch_index]["boxes"][target_indices.to(targets[batch_index]["boxes"].device)])
        if not batch_parts:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, self.hidden_dim, device=device),
                torch.empty(0, 4, device=device),
            )
        batch_indices = torch.cat(batch_parts)
        query_indices = torch.cat(query_parts)
        query_features = context.query_features[batch_indices, query_indices]
        boxes = torch.cat(target_parts).to(device=device)
        return batch_indices, query_features, boxes

    def _sample_memory(
        self,
        context: EncoderContext,
        batch_indices: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        memory = context.memory.float()
        samples = []
        for level in range(self.num_levels):
            height = int(context.spatial_shapes[level, 0].item())
            width = int(context.spatial_shapes[level, 1].item())
            start = int(context.level_start_index[level].item())
            level_memory = memory[:, start : start + height * width]
            level_map = level_memory.reshape(memory.shape[0], height, width, self.hidden_dim).permute(0, 3, 1, 2)
            object_maps = level_map[batch_indices]
            valid_ratio = context.valid_ratios[batch_indices, level].float()
            grid = points[:, level] * valid_ratio[:, None, :]
            grid = grid.mul(2.0).sub(1.0).reshape(len(batch_indices), 1, self.points_per_level, 2)
            sampled = F.grid_sample(
                object_maps,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            samples.append(sampled.squeeze(2).transpose(1, 2))
        return torch.stack(samples, dim=1)

    def forward(
        self,
        context: EncoderContext,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        normalizer: float,
    ) -> dict[str, torch.Tensor]:
        batch_indices, query_features, gt_boxes = self._matched_inputs(context, targets, indices)
        if len(batch_indices) == 0:
            zero = context.memory.sum() * 0.0
            return {"loss_aux_bbox": zero, "loss_aux_giou": zero, "aux_matches": zero.detach()}

        query_float = query_features.float()
        offsets = self.offset_head(query_float).reshape(
            -1, self.num_levels, self.points_per_level, 2
        )
        offsets = offsets.tanh() * 0.5
        points = gt_boxes[:, None, None, :2].float() + gt_boxes[:, None, None, 2:].float() * offsets
        points = points.clamp(0.0, 1.0)
        sampled = self._sample_memory(context, batch_indices, points)
        weights = self.attention_head(query_float).reshape(
            -1, self.num_levels, self.points_per_level
        )
        weights = weights.flatten(1).softmax(dim=-1).reshape_as(weights)
        region_feature = (sampled * weights[..., None]).sum(dim=(1, 2))
        projected = self.feature_projection(region_feature)
        alpha = self.gate(torch.cat((query_float, projected), dim=-1)).sigmoid()
        fused_query = query_float + alpha * projected
        predicted_boxes = self.box_head(fused_query).sigmoid()

        loss_bbox = F.l1_loss(predicted_boxes, gt_boxes.float(), reduction="none").sum() / normalizer
        giou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(predicted_boxes),
            box_ops.box_cxcywh_to_xyxy(gt_boxes.float()),
        )
        loss_giou = (1.0 - torch.diag(giou)).sum() / normalizer
        return {
            "loss_aux_bbox": loss_bbox,
            "loss_aux_giou": loss_giou,
            "aux_matches": torch.as_tensor(float(len(batch_indices)), device=loss_bbox.device),
            "aux_gate_mean": alpha.detach().mean(),
            "aux_offset_abs_mean": offsets.detach().abs().mean(),
            "aux_attention_entropy": (
                -(weights.flatten(1) * weights.flatten(1).clamp_min(1e-8).log()).sum(dim=1).mean().detach()
            ),
        }


class ResearchModel(nn.Module):
    def __init__(self, detector: nn.Module, config: ExperimentConfig) -> None:
        super().__init__()
        self.detector = detector
        self.method = config.method
        self.auxiliary = None
        self.bqr_bridge = None
        self._capture_enabled = False
        self._encoder_cache: dict[str, torch.Tensor] = {}
        self._decoder_hidden: torch.Tensor | None = None
        self._hook_handles = []
        self._query_features: torch.Tensor | None = None
        if config.method == "gt_guided_aux":
            cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            # Initialize new AUX parameters deterministically without changing
            # the data-augmentation RNG stream shared with the baseline run.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(config.seed + 1_000_003)
                self.auxiliary = GTGuidedAuxiliary(
                    config.hidden_dim,
                    config.num_feature_levels,
                    config.sampling_points_per_level,
                )
            self._hook_handles.append(
                detector.transformer.encoder.register_forward_hook(
                    self._capture_encoder, with_kwargs=True
                )
            )
            self._hook_handles.append(
                detector.transformer.decoder.register_forward_hook(self._capture_decoder)
            )
        elif config.method == "bqr_dn_v2":
            ensure_dqr_v2_imports()
            from dqr_v2 import attach_bqr_dn_v2

            cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(config.seed + 2_000_003)
                self.bqr_bridge = attach_bqr_dn_v2(detector, config)
        elif config.method == "bqr_dn_v2_1":
            ensure_dqr_v2_1_imports()
            from dqr_v2_1 import attach_bqr_dn_v2_1

            cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            # Reuse V2's module seed. The V2-compatible modules are created in
            # the same order, so the only initialized difference is the new
            # content-attention branch.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(config.seed + 2_000_003)
                self.bqr_bridge = attach_bqr_dn_v2_1(detector, config)
        elif config.method == "bqr_dn_v3":
            ensure_dqr_v3_imports()
            from dqr_v3 import attach_bqr_dn_v3

            cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            # Keep the BQR module seed shared across versions and ablations.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(config.seed + 2_000_003)
                self.bqr_bridge = attach_bqr_dn_v3(detector, config)
        elif config.method == "bqr_dn_v3_1":
            ensure_dqr_v3_1_imports()
            from dqr_v3_1 import attach_bqr_dn_v3_1

            cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            # Keep initialization identical to V3; only the scale-prior
            # multiplier differs through the experiment configuration.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(config.seed + 2_000_003)
                self.bqr_bridge = attach_bqr_dn_v3_1(detector, config)

    def _capture_encoder(self, module, args, kwargs, output) -> None:
        del module, args
        if not self._capture_enabled:
            return
        memory = output[0] if isinstance(output, tuple) else output
        self._encoder_cache = {
            "memory": memory,
            "spatial_shapes": kwargs["spatial_shapes"],
            "level_start_index": kwargs["level_start_index"],
            "valid_ratios": kwargs["valid_ratios"],
            "padding_mask": kwargs["key_padding_mask"],
        }

    def _capture_decoder(self, module, args, output) -> None:
        del module, args
        if self._capture_enabled:
            hidden_layers = output[0]
            self._decoder_hidden = hidden_layers[-1]

    def forward(self, samples, targets: list[dict] | None = None) -> dict:
        self._capture_enabled = bool(self.training and self.auxiliary is not None and targets is not None)
        self._encoder_cache = {}
        self._decoder_hidden = None
        self._query_features = None
        if self.bqr_bridge is not None:
            self.bqr_bridge.set_targets(targets if self.training else None)
        try:
            outputs = self.detector(samples, targets)
        finally:
            if self.bqr_bridge is not None:
                self.bqr_bridge.clear_targets()
        if self._capture_enabled:
            if not self._encoder_cache or self._decoder_hidden is None:
                raise RuntimeError("DINO encoder/decoder features were not captured")
            pad_size = int((outputs.get("dn_meta") or {}).get("pad_size", 0))
            self._query_features = self._decoder_hidden[:, pad_size:]
            if self._query_features.shape[1] != outputs["pred_boxes"].shape[1]:
                raise RuntimeError("Captured decoder queries do not align with DINO predictions")
        return outputs

    def training_diagnostics(self) -> dict[str, torch.Tensor]:
        if self.bqr_bridge is None:
            return {}
        return dict(self.bqr_bridge.latest_diagnostics)

    def auxiliary_losses(
        self,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        normalizer: float,
    ) -> dict[str, torch.Tensor]:
        if self.auxiliary is None:
            return {}
        if self._query_features is None:
            raise RuntimeError("Auxiliary loss requires a training forward pass")
        context = EncoderContext(
            query_features=self._query_features,
            **self._encoder_cache,
        )
        return self.auxiliary(context, targets, indices, normalizer)

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


def make_research_model(detector: nn.Module, config: ExperimentConfig) -> ResearchModel:
    return ResearchModel(detector, config)
