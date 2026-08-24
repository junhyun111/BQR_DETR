from __future__ import annotations

import torch
import torch.nn as nn

from .dn_prepare import build_valid_dn_mask
from .query_fusion import RegionGuidedDNFusion


class BQRDecoderBridge(nn.Module):
    """Pre-condition only DINO's DN prefix, then call its original decoder."""

    def __init__(
        self,
        decoder: nn.Module,
        *,
        num_queries: int,
        hidden_dim: int,
        num_levels: int,
        points_per_level: int,
        gate_bias: float,
        enabled: bool,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.num_queries = num_queries
        self.enabled = enabled
        self.fusion = RegionGuidedDNFusion(
            hidden_dim,
            num_levels,
            points_per_level,
            gate_bias,
        )
        self._targets: list[dict] | None = None
        self.latest_diagnostics: dict[str, torch.Tensor] = {}

    def set_targets(self, targets: list[dict] | None) -> None:
        self._targets = targets
        self.latest_diagnostics = {}

    def clear_targets(self) -> None:
        self._targets = None

    def forward(
        self,
        tgt,
        memory,
        memory_key_padding_mask=None,
        pos=None,
        refpoints_unsigmoid=None,
        level_start_index=None,
        spatial_shapes=None,
        valid_ratios=None,
        tgt_mask=None,
    ):
        pad_size = int(tgt.shape[0] - self.num_queries)
        if self.training and self.enabled and pad_size > 0:
            if self._targets is None:
                raise RuntimeError("BQR-DN training forward requires targets")
            valid_mask = build_valid_dn_mask(
                self._targets,
                pad_size,
                device=tgt.device,
            )
            dn_queries = tgt[:pad_size].transpose(0, 1)
            noisy_boxes = refpoints_unsigmoid[:pad_size].transpose(0, 1).sigmoid()
            fused, self.latest_diagnostics = self.fusion(
                dn_queries,
                noisy_boxes,
                valid_mask,
                memory.transpose(0, 1),
                spatial_shapes,
                level_start_index,
                valid_ratios,
            )
            tgt = torch.cat((fused.transpose(0, 1), tgt[pad_size:]), dim=0)
        return self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            refpoints_unsigmoid=refpoints_unsigmoid,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask,
        )
