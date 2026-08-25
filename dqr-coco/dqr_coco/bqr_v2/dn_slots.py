from __future__ import annotations

import torch


def build_valid_dn_mask(
    targets: list[dict], pad_size: int, *, device: torch.device
) -> torch.Tensor:
    batch_size = len(targets)
    if batch_size == 0 or pad_size == 0:
        return torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)
    counts = torch.as_tensor(
        [len(target["labels"]) for target in targets],
        dtype=torch.long,
        device=device,
    )
    single_pad = int(counts.max().item())
    if single_pad == 0:
        return torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)
    if pad_size % single_pad:
        raise RuntimeError(
            f"DINO DN pad size {pad_size} is not divisible by {single_pad}"
        )
    slots = torch.arange(pad_size, device=device) % single_pad
    return slots.unsqueeze(0) < counts.unsqueeze(1)


def active_slot_metadata(
    targets: list[dict], active: torch.Tensor, pad_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clean GT boxes, transformed areas and positive-DN flags."""
    device = active.device
    counts = [len(target["labels"]) for target in targets]
    single_pad = max(counts, default=0)
    if single_pad == 0 or pad_size % single_pad:
        raise RuntimeError("Cannot reconstruct DINO DN slot metadata")
    target_indices = active[:, 1] % single_pad
    block_indices = active[:, 1] // single_pad
    positive = block_indices.remainder(2).eq(0)

    boxes = []
    areas = []
    for row, target_index in zip(active[:, 0].tolist(), target_indices.tolist()):
        target = targets[row]
        box = target["boxes"][target_index].to(device=device, dtype=torch.float32)
        boxes.append(box)
        if "area" in target:
            area = target["area"][target_index].to(device=device, dtype=torch.float32)
        else:
            height, width = target["size"].to(device=device, dtype=torch.float32)
            area = box[2] * width * box[3] * height
        areas.append(area)
    return torch.stack(boxes), torch.stack(areas), positive
