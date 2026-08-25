from __future__ import annotations

from .model import BQRV21DecoderBridge


def attach_bqr_dn_v2_1(detector, config):
    bridge = BQRV21DecoderBridge(
        detector.transformer.decoder,
        num_queries=config.num_queries,
        hidden_dim=config.hidden_dim,
        num_levels=config.num_feature_levels,
        points_per_level=config.bqr_points_per_level,
        gate_bias=config.bqr_gate_bias,
        attention_dim=config.bqr_attention_dim,
        content_scale_init=config.bqr_content_scale_init,
        attention_temperature=config.bqr_attention_temperature,
        content_enabled=config.bqr_content_attention,
        enabled=config.bqr_enabled,
    )
    detector.transformer.decoder = bridge
    return bridge
