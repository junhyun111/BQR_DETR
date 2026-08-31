from __future__ import annotations

import torch
import torch.nn as nn

from .dn_slots import build_valid_dn_mask
from .fusion import CleanGTRegionFusion


class BQRDecoderBridge(nn.Module):
    """Modify only the training-time DN prefix before the official decoder."""

    def __init__(
        self,
        decoder: nn.Module,
        *,
        num_queries: int,
        hidden_dim: int,
        num_levels: int,
        grid_size: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.num_queries = num_queries
        self.fusion = CleanGTRegionFusion(hidden_dim, num_levels, grid_size, alpha)
        self._targets: list[dict] | None = None
        self._diagnostics_enabled = False
        self.latest_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def num_layers(self) -> int:
        return self.decoder.num_layers

    @property
    def bbox_embed(self):
        return self.decoder.bbox_embed

    @bbox_embed.setter
    def bbox_embed(self, value) -> None:
        self.decoder.bbox_embed = value

    @property
    def class_embed(self):
        return self.decoder.class_embed

    @class_embed.setter
    def class_embed(self, value) -> None:
        self.decoder.class_embed = value

    def set_context(
        self, targets: list[dict] | None, *, diagnostics_enabled: bool = False
    ) -> None:
        self._targets = targets
        self._diagnostics_enabled = bool(diagnostics_enabled)
        self.latest_diagnostics = {}

    def clear_context(self) -> None:
        self._targets = None
        self._diagnostics_enabled = False

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        attn_mask=None,
    ):
        pad_size = int(tgt.shape[1] - self.num_queries)
        if self.training and pad_size > 0:
            if self._targets is None:
                raise RuntimeError("BQR training forward requires clean targets")
            valid_mask = build_valid_dn_mask(
                self._targets, pad_size, device=tgt.device
            )
            fused, self.latest_diagnostics = self.fusion(
                tgt[:, :pad_size],
                valid_mask,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_valid_ratios,
                self._targets,
                diagnostics_enabled=self._diagnostics_enabled,
            )
            tgt = torch.cat((fused, tgt[:, pad_size:]), dim=1)
        elif self.training and pad_size == 0:
            tgt = tgt + self.fusion._parameter_zero(tgt)
        return self.decoder(
            tgt,
            reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            src_valid_ratios,
            query_pos=query_pos,
            src_padding_mask=src_padding_mask,
            attn_mask=attn_mask,
        )
