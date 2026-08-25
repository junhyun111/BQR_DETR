from __future__ import annotations

import torch

from gt_guided_dino.upstream import ensure_dqr_v2_1_imports, ensure_dqr_v2_imports


ensure_dqr_v2_imports()
ensure_dqr_v2_1_imports()
from dqr_v2.query_fusion import RegionGuidedDNFusion  # noqa: E402
from dqr_v2_1.dn_prepare import build_dn_size_ids  # noqa: E402
from dqr_v2_1.model import BQRV21DecoderBridge  # noqa: E402
from dqr_v2_1.query_fusion import ContentAwareRegionGuidedDNFusion  # noqa: E402


def _fusion_inputs(hidden_dim: int = 8):
    return (
        torch.randn(2, 4, hidden_dim, requires_grad=True),
        torch.rand(2, 4, 4),
        torch.tensor([[True, True, True, True], [True, False, True, False]]),
        torch.randn(2, 5, hidden_dim, requires_grad=True),
        torch.tensor([[2, 2], [1, 1]]),
        torch.tensor([0, 4]),
        torch.ones(2, 2, 2),
    )


def test_dn_size_ids_follow_repeated_dino_slots():
    targets = [
        {
            "labels": torch.tensor([1, 2, 3]),
            "area": torch.tensor([100.0, 2_000.0, 20_000.0]),
        },
        {"labels": torch.tensor([4]), "area": torch.tensor([200.0])},
    ]
    size_ids = build_dn_size_ids(targets, pad_size=6, device=torch.device("cpu"))
    assert size_ids.tolist() == [[0, 1, 2, 0, 1, 2], [0, -1, -1, 0, -1, -1]]


def test_content_attention_has_nonzero_initial_gradient_and_diagnostics():
    module = ContentAwareRegionGuidedDNFusion(
        hidden_dim=8,
        num_levels=2,
        points_per_level=4,
        gate_bias=-2.0,
        attention_dim=4,
        content_scale_init=0.05,
        attention_temperature=1.0,
        content_enabled=True,
    )
    inputs = _fusion_inputs()
    size_ids = torch.tensor([[0, 1, 2, 0], [0, -1, 1, -1]])
    fused, diagnostics = module(*inputs, size_ids)
    assert torch.allclose(diagnostics["bqr_content_scale"], torch.tensor(0.05), atol=1e-6)
    assert diagnostics["bqr_content_logit_std"] > 0
    assert diagnostics["bqr_small_queries"] == 3
    fused.square().mean().backward()
    assert module.content_scale_raw.grad is not None
    assert float(module.content_scale_raw.grad.abs()) > 0
    assert module.query_projection[-1].weight.grad is not None
    assert float(module.query_projection[-1].weight.grad.abs().sum()) > 0
    assert module.feature_key_projection[-1].weight.grad is not None
    assert float(module.feature_key_projection[-1].weight.grad.abs().sum()) > 0


def test_disabling_content_branch_exactly_matches_v2_fusion():
    torch.manual_seed(1234)
    v2 = RegionGuidedDNFusion(8, 2, 4, -2.0)
    torch.manual_seed(1234)
    v21 = ContentAwareRegionGuidedDNFusion(
        hidden_dim=8,
        num_levels=2,
        points_per_level=4,
        gate_bias=-2.0,
        attention_dim=4,
        content_scale_init=0.05,
        attention_temperature=1.0,
        content_enabled=False,
    )
    inputs = _fusion_inputs()
    output_v2, _ = v2(*inputs)
    output_v21, _ = v21(*inputs)
    assert torch.equal(output_v2, output_v21)


class _CaptureDecoder(torch.nn.Module):
    def forward(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["tgt"], kwargs["refpoints_unsigmoid"]


def test_v21_bridge_preserves_normal_queries_reference_and_attention_mask():
    decoder = _CaptureDecoder()
    bridge = BQRV21DecoderBridge(
        decoder,
        num_queries=2,
        hidden_dim=8,
        num_levels=2,
        points_per_level=4,
        gate_bias=-2.0,
        attention_dim=4,
        content_scale_init=0.05,
        attention_temperature=1.0,
        content_enabled=True,
        enabled=True,
    )
    bridge.train()
    bridge.set_targets(
        [{"labels": torch.tensor([1]), "area": torch.tensor([100.0])}]
    )
    tgt = torch.randn(4, 1, 8)
    reference = torch.randn(4, 1, 4)
    attention_mask = torch.zeros(4, 4, dtype=torch.bool)
    normal_before = tgt[2:].clone()
    bridge(
        tgt=tgt,
        memory=torch.randn(5, 1, 8),
        memory_key_padding_mask=torch.zeros(1, 5, dtype=torch.bool),
        pos=torch.randn(5, 1, 8),
        refpoints_unsigmoid=reference,
        level_start_index=torch.tensor([0, 4]),
        spatial_shapes=torch.tensor([[2, 2], [1, 1]]),
        valid_ratios=torch.ones(1, 2, 2),
        tgt_mask=attention_mask,
    )
    assert torch.equal(decoder.kwargs["tgt"][2:], normal_before)
    assert decoder.kwargs["refpoints_unsigmoid"] is reference
    assert decoder.kwargs["tgt_mask"] is attention_mask
