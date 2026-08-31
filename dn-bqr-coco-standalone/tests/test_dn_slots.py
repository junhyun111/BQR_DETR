from __future__ import annotations

import torch

from bqr_dn.bqr.dn_slots import active_clean_metadata, build_valid_dn_mask


def _targets():
    return [
        {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor(
                [[0.2, 0.3, 0.1, 0.2], [0.7, 0.6, 0.3, 0.4]]
            ),
            "area": torch.tensor([100.0, 2000.0]),
        },
        {
            "labels": torch.tensor([3]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "area": torch.tensor([500.0]),
        },
    ]


def test_dn_valid_mask_matches_grouped_padding():
    mask = build_valid_dn_mask(_targets(), 6, device=torch.device("cpu"))
    assert mask.tolist() == [
        [True, True, True, True, True, True],
        [True, False, True, False, True, False],
    ]


def test_active_slots_repeat_clean_boxes_per_dn_group():
    targets = _targets()
    mask = build_valid_dn_mask(targets, 6, device=torch.device("cpu"))
    active = mask.nonzero(as_tuple=False)
    boxes, areas = active_clean_metadata(targets, active, 6)
    expected_first = torch.stack(
        [targets[0]["boxes"][0], targets[0]["boxes"][1]] * 3
    )
    assert torch.equal(boxes[:6], expected_first)
    assert areas.shape == (9,)


def test_empty_targets_produce_empty_valid_mask():
    targets = [
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}
    ]
    assert not build_valid_dn_mask(targets, 0, device=torch.device("cpu")).any()
