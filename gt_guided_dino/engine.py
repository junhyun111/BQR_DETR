from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .build import build_experiment
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, seed_everything
from .data import build_loaders, move_targets_to_device
from .optimizer import build_optimizer


def _make_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler(device.type, enabled=enabled)


def _autocast_dtype(precision: str):
    return torch.float16 if precision == "fp16" else torch.bfloat16


def _weighted_loss_contributions(criterion, losses: dict[str, torch.Tensor]) -> dict[str, float]:
    """Group all weighted decoder and DN terms by objective for logging."""
    grouped = {
        "weighted_ce_total": 0.0,
        "weighted_bbox_total": 0.0,
        "weighted_giou_total": 0.0,
    }
    for name, weight in criterion.weight_dict.items():
        if name not in losses:
            continue
        if name.startswith("loss_ce"):
            key = "weighted_ce_total"
        elif name.startswith("loss_bbox"):
            key = "weighted_bbox_total"
        elif name.startswith("loss_giou"):
            key = "weighted_giou_total"
        else:
            continue
        grouped[key] += float(losses[name].detach() * weight)
    return grouped


def train(
    config: ExperimentConfig,
    *,
    resume: bool | str | Path = True,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.precision != "fp32" and device.type != "cuda":
        raise RuntimeError("The configured mixed precision requires a CUDA GPU")
    seed_everything(config.seed)
    train_loader, _ = build_loaders(config)
    model, criterion, _ = build_experiment(config, device)
    optimizer, scheduler = build_optimizer(model, config)
    scaler = _make_scaler(device, config.precision == "fp16")

    start_epoch = 0
    history: list[dict] = []
    resume_path = None
    if isinstance(resume, (str, Path)) and str(resume).lower() != "auto":
        resume_path = Path(resume).resolve()
    elif resume and config.latest_checkpoint.is_file():
        resume_path = config.latest_checkpoint
    if resume_path is not None:
        start_epoch, history = load_checkpoint(
            resume_path,
            config,
            model,
            optimizer,
            scheduler,
            scaler,
        )

    model.train()
    for epoch in range(start_epoch, config.epochs):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        batches = 0
        optimizer_steps = 0
        iterator = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}", leave=False)
        for batch_index, (samples, targets) in enumerate(iterator):
            samples = samples.to(device)
            targets = move_targets_to_device(targets, device)
            with torch.autocast(
                device_type=device.type,
                dtype=_autocast_dtype(config.precision),
                enabled=config.precision != "fp32",
            ):
                outputs = model(samples, targets)
                loss_dict = criterion(model, outputs, targets)
                loss = criterion.weighted_loss(loss_dict)
                scaled_loss = loss / config.accumulation_steps

            scaler.scale(scaled_loss).backward()
            final_batch = batch_index + 1 == len(train_loader)
            update = (batch_index + 1) % config.accumulation_steps == 0 or final_batch
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            batches += 1
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for name, value in _weighted_loss_contributions(criterion, loss_dict).items():
                totals[name] = totals.get(name, 0.0) + value
            for name, value in loss_dict.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
            iterator.set_postfix(loss=f"{totals['loss'] / batches:.3f}")

        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "lr": lr_used,
            "loss": totals["loss"] / max(batches, 1),
            "batches": batches,
            "optimizer_steps": optimizer_steps,
            "seconds": time.perf_counter() - started,
            "peak_cuda_mb": (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else math.nan
            ),
        }
        for name in (
            "loss_ce", "loss_bbox", "loss_giou", "loss_aux_bbox", "loss_aux_giou",
            "aux_gate_mean", "aux_offset_abs_mean", "aux_attention_entropy",
            "loss_ce_dn", "loss_bbox_dn", "loss_giou_dn",
            "bqr_valid_queries", "bqr_gate_mean", "bqr_offset_abs_mean",
            "bqr_attention_entropy", "bqr_region_norm", "bqr_fusion_delta_norm",
            "bqr_content_scale", "bqr_prior_entropy", "bqr_final_entropy",
            "bqr_content_logit_std", "bqr_top1_attention",
            "weighted_ce_total", "weighted_bbox_total", "weighted_giou_total",
        ):
            if name in totals:
                row[name] = totals[name] / max(batches, 1)
        for size_name in ("small", "medium", "large"):
            count_key = f"bqr_{size_name}_queries"
            count = totals.get(count_key, 0.0)
            row[count_key] = count
            if count <= 0:
                continue
            for metric in (
                "prior_entropy", "final_entropy", "top1_attention",
                "gate_mean", "fusion_delta_norm", "offset_abs_mean",
            ):
                sum_key = f"bqr_{size_name}_{metric}_sum"
                if sum_key in totals:
                    row[f"bqr_{size_name}_{metric}"] = totals[sum_key] / count
        history.append(row)
        save_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            history=history,
        )
        print(
            f"epoch={epoch + 1:02d} loss={row['loss']:.4f} "
            f"lr={row['lr']:.2e} seconds={row['seconds']:.1f}"
        )
    model.close()
    return history
