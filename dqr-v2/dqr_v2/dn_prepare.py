from __future__ import annotations

import torch


def build_valid_dn_mask(
    targets: list[dict] | None,
    pad_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct DINO's valid DN slots without changing its noise process."""
    batch_size = len(targets) if targets is not None else 0
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
    if pad_size % single_pad != 0:
        raise RuntimeError(
            f"DINO DN pad size {pad_size} is not divisible by max target count {single_pad}"
        )
    slot_in_group = torch.arange(pad_size, device=device) % single_pad
    return slot_in_group.unsqueeze(0) < counts.unsqueeze(1)
