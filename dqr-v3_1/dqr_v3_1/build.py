from __future__ import annotations

from .model import BQRV3DecoderBridge


def attach_bqr_dn_v3_1(detector, config):
    bridge = BQRV3DecoderBridge(
        detector.transformer.decoder,
        num_queries=config.num_queries,
        hidden_dim=config.hidden_dim,
        num_levels=config.num_feature_levels,
        points_per_level=config.bqr_points_per_level,
        gate_bias=config.bqr_gate_bias,
        scale_aware=config.bqr_scale_aware,
        target_cells=config.bqr_target_cells,
        scale_sigma=config.bqr_scale_sigma,
        scale_weight=config.bqr_scale_weight,
        scale_logit_floor=config.bqr_scale_logit_floor,
        enabled=config.bqr_enabled,
    )
    detector.transformer.decoder = bridge
    return bridge
