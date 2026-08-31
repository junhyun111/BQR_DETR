from __future__ import annotations

import torch

from bqr_dn.bqr.bridge import BQRDecoderBridge


class EchoDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 1
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, *args, **kwargs):
        return tgt


def _inputs(pad_size=6, num_queries=4):
    torch.manual_seed(7)
    tgt = torch.randn(2, pad_size + num_queries, 8, requires_grad=True)
    memory = torch.randn(2, 8, 8, requires_grad=True)
    shapes = torch.tensor([[2, 3], [1, 2]], dtype=torch.long)
    starts = torch.tensor([0, 6], dtype=torch.long)
    ratios = torch.ones(2, 2, 2)
    references = torch.rand(2, pad_size + num_queries, 4)
    targets = [
        {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor(
                [[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.1]]
            ),
            "area": torch.tensor([2000.0, 300.0]),
        },
        {
            "labels": torch.tensor([3]),
            "boxes": torch.tensor([[0.6, 0.5, 0.2, 0.2]]),
            "area": torch.tensor([900.0]),
        },
    ]
    return tgt, memory, shapes, starts, ratios, references, targets


def test_bridge_changes_only_valid_dn_prefix_and_has_gradients():
    bridge = BQRDecoderBridge(
        EchoDecoder(),
        num_queries=4,
        hidden_dim=8,
        num_levels=2,
        grid_size=2,
        alpha=0.25,
    ).train()
    tgt, memory, shapes, starts, ratios, references, targets = _inputs()
    bridge.set_context(targets, diagnostics_enabled=True)
    output = bridge(tgt, references, memory, shapes, starts, ratios)
    assert torch.equal(output[:, 6:], tgt[:, 6:])
    assert not torch.equal(output[0, :6], tgt[0, :6])
    # Padded slots for the one-object image remain exactly unchanged.
    assert torch.equal(output[1, [1, 3, 5]], tgt[1, [1, 3, 5]])
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in bridge.fusion.parameters())
    assert bridge.latest_diagnostics["bqr_valid_queries"] == 9


def test_eval_is_exact_passthrough_without_targets():
    bridge = BQRDecoderBridge(
        EchoDecoder(),
        num_queries=4,
        hidden_dim=8,
        num_levels=2,
        grid_size=2,
        alpha=0.25,
    ).eval()
    tgt, memory, shapes, starts, ratios, references, _ = _inputs(
        pad_size=0, num_queries=4
    )
    output = bridge(tgt, references, memory, shapes, starts, ratios)
    assert torch.equal(output, tgt)
    assert bridge.latest_diagnostics == {}


def test_zero_dn_prefix_keeps_bqr_parameters_in_training_graph():
    bridge = BQRDecoderBridge(
        EchoDecoder(),
        num_queries=4,
        hidden_dim=8,
        num_levels=2,
        grid_size=2,
        alpha=0.25,
    ).train()
    empty = [
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}
        for _ in range(2)
    ]
    tgt, memory, shapes, starts, ratios, references, _ = _inputs(
        pad_size=0, num_queries=4
    )
    bridge.set_context(empty)
    output = bridge(tgt, references, memory, shapes, starts, ratios)
    output.sum().backward()
    assert torch.equal(output, tgt)
    assert all(parameter.grad is not None for parameter in bridge.fusion.parameters())
