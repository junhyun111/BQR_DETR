from __future__ import annotations

import torch


def _single_pad(targets: list[dict] | None) -> int:
    return max((len(target["labels"]) for target in targets or []), default=0)


def build_valid_dn_mask(
    targets: list[dict] | None,
    pad_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct DINO's valid repeated DN slots, leaving padding inactive."""
    batch_size = len(targets) if targets is not None else 0
    single_pad = _single_pad(targets)
    if batch_size == 0 or pad_size == 0 or single_pad == 0:
        return torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)
    if pad_size % single_pad != 0:
        raise RuntimeError(
            f"DINO DN pad size {pad_size} is not divisible by max target count {single_pad}"
        )
    counts = torch.as_tensor(
        [len(target["labels"]) for target in targets], dtype=torch.long, device=device
    )
    slot_in_group = torch.arange(pad_size, device=device) % single_pad
    return slot_in_group.unsqueeze(0) < counts.unsqueeze(1)


def build_dn_size_ids(
    targets: list[dict] | None,
    pad_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map repeated DN slots to COCO-style size bins for detached diagnostics."""
    batch_size = len(targets) if targets is not None else 0
    result = torch.full((batch_size, pad_size), -1, dtype=torch.long, device=device)
    single_pad = _single_pad(targets)
    if batch_size == 0 or pad_size == 0 or single_pad == 0:
        return result
    if pad_size % single_pad != 0:
        raise RuntimeError(
            f"DINO DN pad size {pad_size} is not divisible by max target count {single_pad}"
        )
    slot_in_group = torch.arange(pad_size, device=device) % single_pad
    for batch_index, target in enumerate(targets or []):
        count = len(target["labels"])
        if count == 0:
            continue
        if "area" in target:
            area = target["area"].detach().to(device=device, dtype=torch.float32)
        else:
            height, width = target["size"].to(device=device, dtype=torch.float32)
            area = (
                target["boxes"][:, 2:].detach().prod(dim=-1).to(device)
                * height
                * width
            )
        size_id = torch.where(area < 32**2, 0, torch.where(area < 96**2, 1, 2))
        valid = slot_in_group < count
        result[batch_index, valid] = size_id[slot_in_group[valid]]
    return result
