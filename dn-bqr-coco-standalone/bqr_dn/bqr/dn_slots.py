from __future__ import annotations

import torch


def build_valid_dn_mask(
    targets: list[dict], pad_size: int, *, device: torch.device
) -> torch.Tensor:
    """Reconstruct the padded DN layout made by official ``prepare_for_dn``."""
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
            f"DN pad size {pad_size} is not divisible by max target count {single_pad}"
        )
    slots = torch.arange(pad_size, device=device) % single_pad
    return slots.unsqueeze(0) < counts.unsqueeze(1)


def active_clean_metadata(
    targets: list[dict], active: torch.Tensor, pad_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map active padded DN slots to clean augmented GT boxes and areas."""
    counts = [len(target["labels"]) for target in targets]
    single_pad = max(counts, default=0)
    if single_pad == 0 or pad_size % single_pad:
        raise RuntimeError("Cannot reconstruct DN slot-to-target mapping")
    target_indices = active[:, 1].remainder(single_pad)
    boxes: list[torch.Tensor] = []
    areas: list[torch.Tensor] = []
    for batch_index, target_index in zip(
        active[:, 0].tolist(), target_indices.tolist()
    ):
        target = targets[batch_index]
        boxes.append(target["boxes"][target_index].to(active.device, torch.float32))
        if "area" in target:
            area = target["area"][target_index].to(active.device, torch.float32)
        else:
            height, width = target["size"].to(active.device, torch.float32)
            box = boxes[-1]
            area = box[2] * width * box[3] * height
        areas.append(area)
    return torch.stack(boxes), torch.stack(areas)
