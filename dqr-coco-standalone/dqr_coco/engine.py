from __future__ import annotations

import math
import time
from contextlib import nullcontext
from itertools import islice

import torch
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import load_training_state, resolve_resume_path, save_training_state
from .config import ExperimentConfig
from .data import DataBundle, move_batch
from .distributed import (
    DistributedContext,
    all_gather_objects,
    all_reduce_sum,
    unwrap_model,
)
from .evaluation import _float_tree, evaluate
from .logging_utils import append_jsonl, write_run_metadata


def _make_scaler(config: ExperimentConfig):
    enabled = config.precision == "fp16" and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(config: ExperimentConfig, device: torch.device):
    enabled = device.type == "cuda" and config.precision != "fp32"
    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _weighted_loss(criterion, losses: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = [
        losses[name] * weight
        for name, weight in criterion.weight_dict.items()
        if name in losses
    ]
    if not terms:
        raise RuntimeError("Official DINO criterion produced no weighted losses")
    return torch.stack(terms).sum()


class EpochAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def add(self, values: dict[str, torch.Tensor], *, count: int = 1) -> None:
        for name, value in values.items():
            scalar = value.detach().float()
            self.sums[name] = self.sums.get(name, torch.zeros_like(scalar)) + scalar * count
            self.counts[name] = self.counts.get(name, 0) + count

    def add_sums(self, values: dict[str, torch.Tensor]) -> None:
        for name, value in values.items():
            scalar = value.detach().float()
            self.sums[name] = self.sums.get(name, torch.zeros_like(scalar)) + scalar
            self.counts[name] = 1

    def gather(self, context: DistributedContext) -> tuple[dict[str, float], dict[str, int]]:
        names = sorted(self.sums)
        values = (
            torch.stack([self.sums[name] for name in names]).cpu().tolist()
            if names
            else []
        )
        local_sums = dict(zip(names, values))
        payloads = all_gather_objects((local_sums, self.counts), context)
        total_sums: dict[str, float] = {}
        total_counts: dict[str, int] = {}
        for sums, counts in payloads:
            for name, value in sums.items():
                total_sums[name] = total_sums.get(name, 0.0) + value
            for name, value in counts.items():
                # Diagnostic *_sum fields are literal sums; their count is not
                # used. Ordinary losses use counts to form a micro-batch mean.
                total_counts[name] = total_counts.get(name, 0) + value
        return total_sums, total_counts


def _finalize_train_metrics(
    accumulator: EpochAccumulator,
    context: DistributedContext,
    *,
    epoch_seconds: float,
    optimizer_steps: int,
    skipped_steps: int,
    peak_memory_mb: float,
) -> dict[str, float]:
    sums, counts = accumulator.gather(context)
    row: dict[str, float] = {}
    for name, total in sums.items():
        if name.startswith("diag_"):
            continue
        count = counts.get(name, 0)
        if count:
            row[name] = total / count

    valid = sums.get("diag_bqr_valid_queries", 0.0)
    base = {
        "bqr_gate": "diag_bqr_gate_sum",
        "bqr_offset": "diag_bqr_offset_abs_sum",
        "bqr_entropy": "diag_bqr_attention_entropy_sum",
        "bqr_region_norm": "diag_bqr_region_norm_sum",
        "bqr_fusion_delta": "diag_bqr_fusion_delta_norm_sum",
    }
    for output_name, sum_name in base.items():
        row[output_name] = sums.get(sum_name, 0.0) / valid if valid else float("nan")
    row["bqr_valid_queries"] = valid
    for size_name in ("small", "medium", "large"):
        size_count = sums.get(f"diag_bqr_{size_name}_queries", 0.0)
        row[f"bqr_{size_name}_queries"] = size_count
        row[f"bqr_{size_name}_gt_inside_ratio"] = (
            sums.get(f"diag_bqr_{size_name}_gt_inside_sum", 0.0) / size_count
            if size_count
            else float("nan")
        )
        for level in range(4):
            row[f"bqr_{size_name}_level{level}_attention_mass"] = (
                sums.get(f"diag_bqr_{size_name}_level{level}_attention_sum", 0.0)
                / size_count
                if size_count
                else float("nan")
            )
        for polarity in ("positive", "negative"):
            polarity_count = sums.get(
                f"diag_bqr_{size_name}_{polarity}_queries", 0.0
            )
            row[f"bqr_{size_name}_{polarity}_gt_inside_ratio"] = (
                sums.get(
                    f"diag_bqr_{size_name}_{polarity}_gt_inside_sum", 0.0
                )
                / polarity_count
                if polarity_count
                else float("nan")
            )
    row.update(
        {
            "epoch_train_time_seconds": epoch_seconds,
            "optimizer_steps": optimizer_steps,
            "amp_skipped_steps": skipped_steps,
            "peak_gpu_memory_mb": peak_memory_mb,
        }
    )
    return row


def train_one_epoch(
    config: ExperimentConfig,
    context: DistributedContext,
    model,
    criterion,
    optimizer,
    data: DataBundle,
    scaler,
    *,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    if data.train_loader is None or data.train_sampler is None:
        raise RuntimeError("Training data loader was not built")
    model.train()
    criterion.train()
    if hasattr(data.train_sampler, "set_epoch"):
        data.train_sampler.set_epoch(epoch)
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
        torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    accumulator = EpochAccumulator()
    optimizer_steps = 0
    skipped_steps = 0
    iterator = iter(data.train_loader)

    while True:
        window = list(islice(iterator, config.accumulation_steps))
        if not window:
            break
        window_size = len(window)
        local_boxes = sum(
            len(target["labels"])
            for _, targets in window
            for target in targets
        )
        global_boxes = torch.tensor(
            float(local_boxes), device=context.device, dtype=torch.float32
        )
        all_reduce_sum(global_boxes, context)
        # Emulate world_size * accumulation_steps virtual ranks. Criterion
        # losses are still divided by window_size before backward.
        num_boxes = torch.clamp(
            global_boxes / (context.world_size * window_size), min=1.0
        )
        optimizer.zero_grad(set_to_none=True)
        local_nonfinite = torch.zeros((), device=context.device, dtype=torch.int32)

        for micro_index, (samples, targets) in enumerate(window):
            final_micro = micro_index + 1 == window_size
            diagnostics = (
                final_micro
                and config.diagnostics_every > 0
                and (global_step + 1) % config.diagnostics_every == 0
                and config.method == "bqr_dn_v2"
            )
            unwrap_model(model).set_diagnostics_enabled(diagnostics)
            sync_context = (
                nullcontext()
                if final_micro or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            samples, targets = move_batch(samples, targets, context.device)
            with sync_context:
                with _autocast(config, context.device):
                    outputs = model(samples, targets)
                # Matcher and all detection losses run in FP32 for AMP safety.
                losses = criterion(
                    _float_tree(outputs),
                    targets,
                    num_boxes_override=num_boxes,
                )
                total_loss = _weighted_loss(criterion, losses)
                local_nonfinite = torch.maximum(
                    local_nonfinite, (~torch.isfinite(total_loss.detach())).int()
                )
                scaler.scale(total_loss / window_size).backward()

            raw = {
                name: value
                for name, value in losses.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            raw["train_total_loss"] = total_loss
            for name in (
                "loss_ce",
                "loss_bbox",
                "loss_giou",
                "loss_ce_dn",
                "loss_bbox_dn",
                "loss_giou_dn",
            ):
                if name in losses and name in criterion.weight_dict:
                    raw[f"weighted_{name}"] = losses[name] * criterion.weight_dict[name]
            accumulator.add(raw)
            if diagnostics:
                accumulator.add_sums(
                    {
                        f"diag_{name}": value
                        for name, value in unwrap_model(model)
                        .training_diagnostics()
                        .items()
                    }
                )

        bad = local_nonfinite
        if context.distributed:
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bad.item():
            raise FloatingPointError(
                f"Non-finite loss detected at epoch {epoch + 1}, step {global_step + 1}"
            )

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip
        )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < previous_scale:
            skipped_steps += 1
        optimizer_steps += 1
        global_step += 1
        accumulator.add({"grad_norm": grad_norm})
        if context.is_main and (global_step % 100 == 0 or optimizer_steps == 1):
            print(
                f"epoch={epoch + 1:02d} step={global_step} "
                f"loss={float(total_loss.detach()):.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
        peak_memory = torch.cuda.max_memory_allocated(context.device) / 1024**2
    else:
        peak_memory = 0.0
    elapsed = time.perf_counter() - started
    timing = torch.tensor(
        [elapsed, peak_memory], dtype=torch.float64, device=context.device
    )
    if context.distributed:
        torch.distributed.all_reduce(timing, op=torch.distributed.ReduceOp.MAX)
    metrics = _finalize_train_metrics(
        accumulator,
        context,
        epoch_seconds=float(timing[0]),
        optimizer_steps=optimizer_steps,
        skipped_steps=skipped_steps,
        peak_memory_mb=float(timing[1]),
    )
    return metrics, global_step


def train(
    config: ExperimentConfig,
    context: DistributedContext,
    model,
    criterion,
    postprocessors: dict,
    optimizer,
    scheduler,
    data: DataBundle,
    *,
    resume: str | None,
) -> list[dict]:
    write_run_metadata(config, context)
    scaler = _make_scaler(config)
    start_epoch = 0
    global_step = 0
    best_ap = float("-inf")
    history: list[dict] = []
    resume_path = resolve_resume_path(config, resume)
    if resume is None and resume_path is None and config.latest_checkpoint.is_file():
        raise FileExistsError(
            f"Run already contains a checkpoint: {config.latest_checkpoint}. "
            "Use --resume auto or choose a different --run-name."
        )
    if resume_path is not None:
        start_epoch, global_step, best_ap, history = load_training_state(
            resume_path,
            config,
            context,
            model,
            optimizer,
            scheduler,
            scaler,
        )
        if context.is_main:
            print(f"Resumed from {resume_path} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, config.epochs):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics, global_step = train_one_epoch(
            config,
            context,
            model,
            criterion,
            optimizer,
            data,
            scaler,
            epoch=epoch,
            global_step=global_step,
        )
        val_metrics = evaluate(config, context, model, postprocessors, data)
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "lr": learning_rate,
            **train_metrics,
            **val_metrics,
        }
        history.append(row)
        current_ap = float(val_metrics["AP"])
        is_best = current_ap > best_ap
        if is_best:
            best_ap = current_ap
        scheduler.step()
        save_training_state(
            config,
            context,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            global_step=global_step,
            best_ap=best_ap,
            history=history,
            is_best=is_best,
            is_final=epoch + 1 == config.epochs,
        )
        if context.is_main:
            append_jsonl(config.run_dir / "metrics.jsonl", row)
            print(
                f"epoch={epoch + 1:02d} AP={row['AP']:.4f} "
                f"AP50={row['AP50']:.4f} AP75={row['AP75']:.4f} "
                f"train={row['epoch_train_time_seconds'] / 60:.1f}m "
                f"val={row['val_time_seconds'] / 60:.1f}m",
                flush=True,
            )
        context.barrier()
    return history
