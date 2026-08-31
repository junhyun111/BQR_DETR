from __future__ import annotations

import torch
import torch.nn.functional as F

from bqr_dn.bqr.region_sampler import FixedBoxRegionSampler


def _reference_sample(
    sampler, memory, spatial_shapes, starts, ratios, batch_indices, boxes
):
    points = sampler.points_for_boxes(boxes)
    samples = []
    for level in range(len(spatial_shapes)):
        height, width = (int(value) for value in spatial_shapes[level])
        start = int(starts[level])
        level_map = memory[:, start : start + height * width].reshape(
            memory.shape[0], height, width, memory.shape[-1]
        ).permute(0, 3, 1, 2)
        object_maps = level_map[batch_indices]
        grid = points * ratios[batch_indices, level, None, :]
        grid = grid.mul(2).sub(1).reshape(len(batch_indices), 1, -1, 2)
        sampled = F.grid_sample(
            object_maps,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        samples.append(sampled.squeeze(2).transpose(1, 2))
    return torch.stack(samples, dim=1)


def test_batch_aware_sampler_matches_feature_map_duplication():
    torch.manual_seed(3)
    sampler = FixedBoxRegionSampler(4, 2, 2)
    shapes = torch.tensor([[2, 3], [1, 2]], dtype=torch.long)
    starts = torch.tensor([0, 6], dtype=torch.long)
    memory = torch.randn(2, 8, 4, requires_grad=True)
    ratios = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[0.8, 0.9], [0.8, 0.9]]])
    batch_indices = torch.tensor([0, 0, 1, 1, 1])
    boxes = torch.tensor(
        [
            [0.5, 0.5, 0.4, 0.4],
            [0.3, 0.3, 0.2, 0.1],
            [0.5, 0.5, 0.2, 0.2],
            [0.7, 0.4, 0.1, 0.3],
            [0.2, 0.8, 0.2, 0.1],
        ]
    )
    actual = sampler(memory, shapes, starts, ratios, batch_indices, boxes)
    expected = _reference_sample(
        sampler, memory, shapes, starts, ratios, batch_indices, boxes
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_two_by_two_points_cover_box_interior_not_only_center():
    sampler = FixedBoxRegionSampler(8, 4, 2)
    box = torch.tensor([[0.5, 0.5, 0.4, 0.2]])
    points = sampler.points_for_boxes(box)[0]
    assert points.unique(dim=0).shape[0] == 4
    assert torch.allclose(points.mean(0), box[0, :2])
    assert points[:, 0].min() > 0.3 and points[:, 0].max() < 0.7
    assert points[:, 1].min() > 0.4 and points[:, 1].max() < 0.6
