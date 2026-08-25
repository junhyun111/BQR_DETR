from __future__ import annotations

import torch
import torch.nn.functional as F

from dqr_coco.bqr_v2.bridge import BQRDecoderBridge
from dqr_coco.bqr_v2.fusion import RegionGuidedDNFusion
from dqr_coco.bqr_v2.region_sampler import MultiScaleRegionSampler


class _EchoDecoder(torch.nn.Module):
    def forward(self, **kwargs):
        return kwargs["tgt"]


def _reference_sample(
    memory, spatial_shapes, level_start_index, valid_ratios, batch_indices, points
):
    hidden = memory.shape[-1]
    samples = []
    for level in range(len(spatial_shapes)):
        height, width = (int(value) for value in spatial_shapes[level])
        start = int(level_start_index[level])
        level_map = memory[:, start : start + height * width].reshape(
            memory.shape[0], height, width, hidden
        ).permute(0, 3, 1, 2)
        object_maps = level_map[batch_indices]
        grid = points[:, level] * valid_ratios[batch_indices, level, None, :]
        grid = grid.mul(2).sub(1).reshape(len(batch_indices), 1, points.shape[2], 2)
        sampled = F.grid_sample(
            object_maps, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        samples.append(sampled.squeeze(2).transpose(1, 2))
    return torch.stack(samples, dim=1)


def test_batch_aware_sampler_matches_feature_map_duplication():
    torch.manual_seed(3)
    shapes = torch.tensor([[2, 3], [1, 2]], dtype=torch.long)
    starts = torch.tensor([0, 6], dtype=torch.long)
    memory = torch.randn(2, 8, 4, requires_grad=True)
    ratios = torch.ones(2, 2, 2)
    batch_indices = torch.tensor([0, 0, 1, 1, 1])
    points = torch.rand(5, 2, 4, 2, requires_grad=True)
    sampler = MultiScaleRegionSampler(4, 2, 4)
    actual = sampler(memory, shapes, starts, ratios, batch_indices, points)
    expected = _reference_sample(
        memory, shapes, starts, ratios, batch_indices, points
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_empty_dn_batch_keeps_all_parameters_in_gradient_graph():
    fusion = RegionGuidedDNFusion(8, 2, 4, -2.0)
    queries = torch.randn(2, 4, 8, requires_grad=True)
    noisy_boxes = torch.rand(2, 4, 4)
    valid = torch.zeros(2, 4, dtype=torch.bool)
    memory = torch.randn(2, 8, 8, requires_grad=True)
    shapes = torch.tensor([[2, 3], [1, 2]], dtype=torch.long)
    starts = torch.tensor([0, 6], dtype=torch.long)
    ratios = torch.ones(2, 2, 2)
    targets = [
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
    ]
    output, diagnostics = fusion(
        queries,
        noisy_boxes,
        valid,
        memory,
        shapes,
        starts,
        ratios,
        targets,
        diagnostics_enabled=True,
    )
    output.sum().backward()
    assert diagnostics == {}
    assert torch.equal(output, queries)
    assert all(parameter.grad is not None for parameter in fusion.parameters())


def test_zero_length_dn_prefix_is_ddp_safe():
    bridge = BQRDecoderBridge(
        _EchoDecoder(),
        num_queries=4,
        hidden_dim=8,
        num_levels=2,
        points_per_level=4,
        gate_bias=-2.0,
    ).train()
    bridge.set_context(
        [
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        ]
    )
    tgt = torch.randn(4, 2, 8, requires_grad=True)
    output = bridge(
        tgt=tgt,
        memory=torch.randn(8, 2, 8),
        refpoints_unsigmoid=torch.randn(4, 2, 4),
        spatial_shapes=torch.tensor([[2, 3], [1, 2]]),
        level_start_index=torch.tensor([0, 6]),
        valid_ratios=torch.ones(2, 2, 2),
    )
    output.sum().backward()
    assert torch.equal(output, tgt)
    assert all(parameter.grad is not None for parameter in bridge.fusion.parameters())


def test_diagnostics_include_scale_level_mass_and_inside_ratio():
    torch.manual_seed(9)
    fusion = RegionGuidedDNFusion(8, 2, 4, -2.0)
    queries = torch.randn(2, 8, 8, requires_grad=True)
    noisy_boxes = torch.tensor(
        [[[0.5, 0.5, 0.4, 0.4]] * 8, [[0.5, 0.5, 0.2, 0.2]] * 8],
        dtype=torch.float32,
    )
    valid = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0]],
        dtype=torch.bool,
    )
    memory = torch.randn(2, 8, 8, requires_grad=True)
    shapes = torch.tensor([[2, 3], [1, 2]], dtype=torch.long)
    starts = torch.tensor([0, 6], dtype=torch.long)
    ratios = torch.ones(2, 2, 2)
    targets = [
        {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.1, 0.1]]),
            "area": torch.tensor([12_000.0, 500.0]),
        },
        {
            "labels": torch.tensor([3]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "area": torch.tensor([2_500.0]),
        },
    ]
    output, diagnostics = fusion(
        queries,
        noisy_boxes,
        valid,
        memory,
        shapes,
        starts,
        ratios,
        targets,
        diagnostics_enabled=True,
    )
    assert output.shape == queries.shape
    assert diagnostics["bqr_valid_queries"] == 12
    for size in ("small", "medium", "large"):
        count = diagnostics[f"bqr_{size}_queries"]
        if count > 0:
            mass = sum(
                diagnostics[f"bqr_{size}_level{level}_attention_sum"]
                for level in range(2)
            )
            assert torch.allclose(mass / count, torch.tensor(1.0), atol=1e-6)
            ratio = diagnostics[f"bqr_{size}_gt_inside_sum"] / count
            assert 0 <= ratio <= 1
