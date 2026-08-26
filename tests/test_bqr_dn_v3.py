from __future__ import annotations

import torch
import torch.nn.functional as F

from gt_guided_dino.upstream import ensure_dqr_v3_imports


ensure_dqr_v3_imports()
from dqr_v3.model import BQRV3DecoderBridge  # noqa: E402
from dqr_v3.query_fusion import ScaleAwareRegionGuidedDNFusion  # noqa: E402
from dqr_v3.region_sampler import MultiScaleRegionSampler  # noqa: E402


def _module(*, scale_aware: bool = True, scale_weight: float = 1.0):
    return ScaleAwareRegionGuidedDNFusion(
        hidden_dim=8,
        num_levels=4,
        points_per_level=5,
        gate_bias=-2.0,
        scale_aware=scale_aware,
        target_cells=4.0,
        scale_sigma=0.8,
        scale_weight=scale_weight,
        scale_logit_floor=-4.0,
    )


def _fusion_inputs():
    queries = torch.randn(2, 4, 8, requires_grad=True)
    noisy_boxes = torch.rand(2, 4, 4)
    noisy_boxes[..., 2:] = noisy_boxes[..., 2:] * 0.6 + 0.05
    valid_mask = torch.tensor(
        [[True, True, True, True], [True, False, True, False]]
    )
    memory = torch.randn(2, 8, 8, requires_grad=True)
    spatial_shapes = torch.tensor([[2, 2], [2, 1], [1, 1], [1, 1]])
    level_start_index = torch.tensor([0, 4, 6, 7])
    valid_ratios = torch.ones(2, 4, 2)
    size_ids = torch.tensor([[0, 1, 2, 0], [0, -1, 1, -1]])
    return (
        queries,
        noisy_boxes,
        valid_mask,
        memory,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        size_ids,
    )


def _per_query_sample_reference(
    memory,
    spatial_shapes,
    level_start_index,
    valid_ratios,
    batch_indices,
    points,
):
    samples = []
    hidden_dim = memory.shape[-1]
    for level in range(len(spatial_shapes)):
        height, width = spatial_shapes[level].tolist()
        start = int(level_start_index[level])
        level_map = memory[:, start : start + height * width].reshape(
            memory.shape[0], height, width, hidden_dim
        ).permute(0, 3, 1, 2)
        object_maps = level_map[batch_indices]
        grid = points[:, level] * valid_ratios[batch_indices, level, None, :]
        grid = grid.mul(2.0).sub(1.0).reshape(
            len(batch_indices), 1, points.shape[2], 2
        )
        sampled = F.grid_sample(
            object_maps,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        samples.append(sampled.squeeze(2).transpose(1, 2))
    return torch.stack(samples, dim=1)


def test_region_sampler_batches_grids_without_repeating_feature_maps(monkeypatch):
    torch.manual_seed(7)
    batch_size, hidden_dim, points_per_level = 4, 3, 5
    spatial_shapes = torch.tensor([[3, 4], [2, 2], [1, 2], [1, 1]])
    level_sizes = spatial_shapes.prod(dim=1)
    level_start_index = torch.cat(
        (torch.zeros(1, dtype=torch.long), level_sizes.cumsum(0)[:-1])
    )
    memory = torch.randn(batch_size, int(level_sizes.sum()), hidden_dim)
    valid_ratios = torch.rand(batch_size, 4, 2) * 0.4 + 0.6
    # Deliberately ungrouped, uneven, and with image 1 having no active query.
    batch_indices = torch.tensor([2, 0, 3, 2, 0, 2])
    points = torch.rand(len(batch_indices), 4, points_per_level, 2)

    expected = _per_query_sample_reference(
        memory,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        batch_indices,
        points,
    )

    import dqr_v3.region_sampler as region_sampler_module

    original_grid_sample = region_sampler_module.F.grid_sample
    input_batch_sizes = []
    grid_shapes = []

    def capture_shapes(input_map, grid, **kwargs):
        input_batch_sizes.append(input_map.shape[0])
        grid_shapes.append(tuple(grid.shape))
        return original_grid_sample(input_map, grid, **kwargs)

    monkeypatch.setattr(region_sampler_module.F, "grid_sample", capture_shapes)
    actual = MultiScaleRegionSampler(hidden_dim, 4, points_per_level)(
        memory,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        batch_indices,
        points,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert input_batch_sizes == [batch_size] * 4
    assert grid_shapes == [(batch_size, 3, points_per_level, 2)] * 4


def test_five_point_initialization_is_center_plus_corners():
    module = _module()
    offsets = module.offset_head(torch.zeros(1, 8)).reshape(1, 4, 5, 2)
    offsets = offsets.tanh() * 0.5
    expected = torch.tensor(
        [
            [0.00, 0.00],
            [-0.25, -0.25],
            [0.25, -0.25],
            [-0.25, 0.25],
            [0.25, 0.25],
        ]
    )
    assert torch.allclose(offsets[0], expected.expand(4, -1, -1), atol=1e-6)


def test_scale_prior_moves_from_fine_to_coarse_levels_with_object_size():
    module = _module()
    boxes = torch.tensor(
        [
            [0.5, 0.5, 0.05, 0.05],
            [0.5, 0.5, 0.50, 0.50],
        ]
    )
    spatial_shapes = torch.tensor([[64, 64], [32, 32], [16, 16], [8, 8]])
    valid_ratios = torch.ones(1, 4, 2)
    logits = module.compute_scale_prior(
        boxes, spatial_shapes, valid_ratios, torch.tensor([0, 0])
    )
    assert int(logits[0].argmax()) == 0
    assert int(logits[1].argmax()) == 3
    assert torch.equal(logits.amax(dim=1), torch.zeros(2))
    assert float(logits.min()) >= -4.0


def test_scale_prior_matches_width_height_order_and_valid_shape():
    module = _module()
    boxes = torch.tensor([[0.5, 0.5, 0.10, 0.20]])
    spatial_shapes = torch.tensor([[20, 40], [10, 20], [5, 10], [2, 5]])
    valid_ratios = torch.ones(1, 4, 2)
    logits = module.compute_scale_prior(
        boxes, spatial_shapes, valid_ratios, torch.tensor([0])
    )
    # At level 0: width=0.1*40=4 and height=0.2*20=4 cells.
    assert torch.isclose(logits[0, 0], torch.tensor(0.0))
    assert int(logits.argmax(dim=1)) == 0


def test_disabling_scale_prior_matches_zero_scale_weight():
    torch.manual_seed(123)
    disabled = _module(scale_aware=False, scale_weight=1.0)
    torch.manual_seed(123)
    zero_weight = _module(scale_aware=True, scale_weight=0.0)
    inputs = _fusion_inputs()
    disabled_output, disabled_diagnostics = disabled(*inputs)
    zero_output, zero_diagnostics = zero_weight(*inputs)
    assert torch.equal(disabled_output, zero_output)
    assert torch.equal(
        disabled_diagnostics["bqr_final_attention_entropy"],
        zero_diagnostics["bqr_final_attention_entropy"],
    )


def test_v3_fusion_is_differentiable_preserves_padding_and_logs_level_mass():
    module = _module()
    inputs = _fusion_inputs()
    queries, _, valid_mask, memory, *_ = inputs
    fused, diagnostics = module(*inputs)
    assert torch.equal(fused[1, ~valid_mask[1]], queries[1, ~valid_mask[1]])
    assert float(diagnostics["bqr_valid_queries"]) == 6.0
    for size_name, count in (("small", 3.0), ("medium", 2.0), ("large", 1.0)):
        assert float(diagnostics[f"bqr_{size_name}_queries"]) == count
        total_level_mass = sum(
            diagnostics[f"bqr_{size_name}_level_{level}_attention_sum"]
            for level in range(4)
        )
        assert torch.allclose(total_level_mass, torch.tensor(count), atol=1e-6)
    fused.square().mean().backward()
    assert memory.grad is not None and float(memory.grad.abs().sum()) > 0
    assert queries.grad is not None and float(queries.grad.abs().sum()) > 0


class _CaptureDecoder(torch.nn.Module):
    def forward(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["tgt"], kwargs["refpoints_unsigmoid"]


def test_v3_bridge_changes_only_dn_content_and_keeps_reference():
    decoder = _CaptureDecoder()
    bridge = BQRV3DecoderBridge(
        decoder,
        num_queries=2,
        hidden_dim=8,
        num_levels=4,
        points_per_level=5,
        gate_bias=-2.0,
        scale_aware=True,
        target_cells=4.0,
        scale_sigma=0.8,
        scale_weight=1.0,
        scale_logit_floor=-4.0,
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
        memory=torch.randn(8, 1, 8),
        memory_key_padding_mask=torch.zeros(1, 8, dtype=torch.bool),
        pos=torch.randn(8, 1, 8),
        refpoints_unsigmoid=reference,
        level_start_index=torch.tensor([0, 4, 6, 7]),
        spatial_shapes=torch.tensor([[2, 2], [2, 1], [1, 1], [1, 1]]),
        valid_ratios=torch.ones(1, 4, 2),
        tgt_mask=attention_mask,
    )
    assert torch.equal(decoder.kwargs["tgt"][2:], normal_before)
    assert decoder.kwargs["refpoints_unsigmoid"] is reference
    assert decoder.kwargs["tgt_mask"] is attention_mask
