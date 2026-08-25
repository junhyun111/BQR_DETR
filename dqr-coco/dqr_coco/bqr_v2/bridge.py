from __future__ import annotations

import torch
import torch.nn as nn

from .dn_slots import build_valid_dn_mask
from .fusion import RegionGuidedDNFusion


class BQRDecoderBridge(nn.Module):
    """Fuse only the training-time DINO DN prefix, then call its decoder."""

    def __init__(
        self,
        decoder: nn.Module,
        *,
        num_queries: int,
        hidden_dim: int,
        num_levels: int,
        points_per_level: int,
        gate_bias: float,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.num_queries = num_queries
        self.fusion = RegionGuidedDNFusion(
            hidden_dim, num_levels, points_per_level, gate_bias
        )
        self._targets: list[dict] | None = None
        self._diagnostics_enabled = False
        self.latest_diagnostics: dict[str, torch.Tensor] = {}

    def set_context(
        self, targets: list[dict] | None, *, diagnostics_enabled: bool = False
    ) -> None:
        self._targets = targets
        self._diagnostics_enabled = diagnostics_enabled
        self.latest_diagnostics = {}

    def clear_context(self) -> None:
        self._targets = None
        self._diagnostics_enabled = False

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
        if self.training and pad_size > 0:
            if self._targets is None:
                raise RuntimeError("BQR-DN V2 training forward requires targets")
            valid_mask = build_valid_dn_mask(
                self._targets, pad_size, device=tgt.device
            )
            fused, self.latest_diagnostics = self.fusion(
                tgt[:pad_size].transpose(0, 1),
                refpoints_unsigmoid[:pad_size].transpose(0, 1).sigmoid(),
                valid_mask,
                memory.transpose(0, 1),
                spatial_shapes,
                level_start_index,
                valid_ratios,
                self._targets,
                diagnostics_enabled=self._diagnostics_enabled,
            )
            tgt = torch.cat((fused.transpose(0, 1), tgt[pad_size:]), dim=0)
        elif self.training and pad_size == 0:
            # A crop can remove every GT box on one DDP rank, in which case
            # official DINO creates no DN prefix. Preserve the exact decoder
            # input while keeping V2 parameters participating with zero grad.
            tgt = tgt + self.fusion._parameter_zero(tgt)
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
