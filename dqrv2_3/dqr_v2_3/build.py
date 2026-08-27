from __future__ import annotations

from .model import BQRDecoderBridge


def attach_bqr_dn_v2_3(detector, config):
    bridge = BQRDecoderBridge(
        detector.transformer.decoder,
        num_queries=config.num_queries,
        hidden_dim=config.hidden_dim,
        num_levels=config.num_feature_levels,
        points_per_level=config.bqr_points_per_level,
        gate_bias=config.bqr_gate_bias,
        residual_ratio=config.bqr_residual_ratio,
        enabled=config.bqr_enabled,
    )
    detector.transformer.decoder = bridge
    return bridge
