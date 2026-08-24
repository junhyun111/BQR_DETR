from __future__ import annotations

import torch
from dataclasses import replace

from .build import build_experiment
from .config import smoke_config
from .data import build_loaders, move_targets_to_device
from .optimizer import build_optimizer


def run_smoke(method: str = "gt_guided_aux", device_name: str | None = None) -> dict:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = smoke_config(method=method, output_root="artifacts/smoke")
    train_loader, _ = build_loaders(config)
    model, criterion, _ = build_experiment(
        config, device, use_common_initialization=False
    )
    optimizer, _ = build_optimizer(model, config)
    samples, targets = next(iter(train_loader))
    samples = samples.to(device)
    targets = move_targets_to_device(targets, device)
    model.train()
    outputs = model(samples, targets)
    losses = criterion(model, outputs, targets)
    total = criterion.weighted_loss(losses)
    total.backward()
    optimizer.step()
    result = {
        "device": str(device),
        "method": method,
        "loss": float(total.detach()),
        "queries": int(outputs["pred_boxes"].shape[1]),
        "finite": bool(torch.isfinite(total)),
        "aux_matches": float(losses.get("aux_matches", torch.tensor(0.0))),
    }
    model.close()
    return result


def run_official_shape_smoke(method: str = "gt_guided_aux", image_size: int = 256) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Official-shape smoke test requires CUDA")
    device = torch.device("cuda")
    config = replace(
        smoke_config(method=method, output_root="artifacts/smoke_official_shape"),
        train_limit=1,
        num_queries=900,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        dn_number=100,
        train_scales=(image_size,),
        train_max_size=max(384, int(image_size * 1.67)),
        val_size=image_size,
        val_max_size=max(384, int(image_size * 1.67)),
    )
    train_loader, _ = build_loaders(config)
    model, criterion, _ = build_experiment(config, device, use_common_initialization=False)
    optimizer, _ = build_optimizer(model, config)
    samples, targets = next(iter(train_loader))
    samples = samples.to(device)
    targets = move_targets_to_device(targets, device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(samples, targets)
        losses = criterion(model, outputs, targets)
        total = criterion.weighted_loss(losses)
    total.backward()
    optimizer.step()
    result = {
        "device": str(device),
        "method": method,
        "loss": float(total.detach()),
        "queries": int(outputs["pred_boxes"].shape[1]),
        "image_size": image_size,
        "finite": bool(torch.isfinite(total)),
        "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "aux_matches": float(losses.get("aux_matches", torch.tensor(0.0))),
    }
    model.close()
    return result


@torch.no_grad()
def run_baseline_parity_smoke() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_config = smoke_config(method="baseline", output_root="artifacts/parity_smoke")
    loader, _ = build_loaders(baseline_config)
    samples, _ = next(iter(loader))
    samples = samples.to(device)

    baseline, _, _ = build_experiment(baseline_config, device)
    baseline.eval()
    baseline_outputs = baseline(samples, None)
    logits = baseline_outputs["pred_logits"].detach().cpu()
    boxes = baseline_outputs["pred_boxes"].detach().cpu()
    baseline.close()
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    proposed_config = replace(baseline_config, method="gt_guided_aux")
    proposed, _, _ = build_experiment(proposed_config, device)
    proposed.eval()
    proposed_outputs = proposed(samples, None)
    logit_difference = float((logits - proposed_outputs["pred_logits"].detach().cpu()).abs().max())
    box_difference = float((boxes - proposed_outputs["pred_boxes"].detach().cpu()).abs().max())
    capture_was_disabled = proposed._query_features is None and not proposed._encoder_cache
    proposed.close()
    return {
        "max_logit_difference": logit_difference,
        "max_box_difference": box_difference,
        "aux_capture_disabled_at_inference": capture_was_disabled,
    }
