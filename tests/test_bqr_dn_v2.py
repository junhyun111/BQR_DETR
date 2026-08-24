import torch

from gt_guided_dino.upstream import ensure_dqr_v2_imports


ensure_dqr_v2_imports()
from dqr_v2.dn_prepare import build_valid_dn_mask  # noqa: E402
from dqr_v2.model import BQRDecoderBridge  # noqa: E402
from dqr_v2.query_fusion import RegionGuidedDNFusion  # noqa: E402


def test_valid_dn_mask_matches_dino_group_padding():
    targets = [
        {"labels": torch.tensor([1, 2])},
        {"labels": torch.tensor([3])},
    ]
    mask = build_valid_dn_mask(targets, pad_size=8, device=torch.device("cpu"))
    assert mask.tolist() == [
        [True, True, True, True, True, True, True, True],
        [True, False, True, False, True, False, True, False],
    ]


def test_bqr_fusion_is_differentiable_and_preserves_padding():
    module = RegionGuidedDNFusion(
        hidden_dim=8,
        num_levels=2,
        points_per_level=4,
        gate_bias=-2.0,
    )
    queries = torch.randn(2, 4, 8, requires_grad=True)
    noisy_boxes = torch.rand(2, 4, 4)
    valid_mask = torch.tensor(
        [[True, True, True, True], [True, False, True, False]]
    )
    memory = torch.randn(2, 5, 8, requires_grad=True)
    fused, diagnostics = module(
        queries,
        noisy_boxes,
        valid_mask,
        memory,
        torch.tensor([[2, 2], [1, 1]]),
        torch.tensor([0, 4]),
        torch.ones(2, 2, 2),
    )
    assert torch.equal(fused[1, ~valid_mask[1]], queries[1, ~valid_mask[1]])
    fused.square().mean().backward()
    assert memory.grad is not None and float(memory.grad.abs().sum()) > 0
    assert queries.grad is not None and float(queries.grad.abs().sum()) > 0
    assert float(diagnostics["bqr_valid_queries"]) == 6.0


class _CaptureDecoder(torch.nn.Module):
    def forward(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["tgt"], kwargs["refpoints_unsigmoid"]


def test_decoder_bridge_changes_only_dn_content_and_keeps_reference():
    decoder = _CaptureDecoder()
    bridge = BQRDecoderBridge(
        decoder,
        num_queries=2,
        hidden_dim=8,
        num_levels=2,
        points_per_level=4,
        gate_bias=-2.0,
        enabled=True,
    )
    bridge.train()
    bridge.set_targets([{"labels": torch.tensor([1])}])
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
