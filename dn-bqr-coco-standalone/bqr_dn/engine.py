from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch

from .checkpoint import (
    load_training_state,
    resolve_resume_path,
    save_training_state,
)
from .config import ExperimentConfig
from .data import DataBundle, move_batch
from .distributed import DistributedContext, all_gather_objects, unwrap_model
from .evaluation import autocast_context, evaluate, float_tree
from .logging_utils import append_jsonl, write_run_metadata


class TensorAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def update(self, values: dict[str, torch.Tensor | float]) -> None:
        for name, value in values.items():
            tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
            tensor = tensor.float()
            self.sums[name] = self.sums.get(name, tensor.new_zeros(())) + tensor
            self.counts[name] = self.counts.get(name, 0) + 1

    def local_payload(self) -> dict[str, tuple[float, int]]:
        return {
            name: (float(value.item()), self.counts[name])
            for name, value in self.sums.items()
        }


def _merge_mean_payloads(payloads: list[dict]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for payload in payloads:
        for name, (value, count) in payload.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + count
    return {name: sums[name] / max(counts[name], 1) for name in sums}


def _merge_diagnostics(payloads: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for payload in payloads:
        for name, (value, count) in payload.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + count
    result: dict[str, float] = {}
    valid = totals.get("bqr_valid_queries", 0.0)
    if valid > 0:
        result["bqr_valid_queries"] = valid
        for source, target in (
            ("bqr_gate_sum", "bqr_gate_mean"),
            ("bqr_region_norm_sum", "bqr_region_norm_mean"),
            ("bqr_fusion_delta_norm_sum", "bqr_fusion_delta_norm_mean"),
        ):
            if source in totals:
                result[target] = totals[source] / valid
        for name, value in totals.items():
            if name.startswith("bqr_level") and name.endswith("_norm_sum"):
                result[name.removesuffix("_sum") + "_mean"] = value / valid
            elif name.startswith("bqr_") and name.endswith("_queries"):
                result[name] = value
    if "bqr_grad_norm" in totals:
        result["bqr_grad_norm"] = totals["bqr_grad_norm"] / max(
            counts["bqr_grad_norm"], 1
        )
    return result


def _make_scaler(config: ExperimentConfig):
    enabled = config.precision == "fp16" and torch.cuda.is_available()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _bqr_grad_norm(model) -> torch.Tensor | None:
    raw_model = unwrap_model(model)
    parameters = raw_model.bqr_parameters()
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return None
    total = gradients[0].new_zeros(())
    for gradient in gradients:
        total = total + gradient.detach().float().pow(2).sum()
    return total.sqrt()


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
    if data.train_loader is None:
        raise RuntimeError("Training data loader is unavailable")
    if isinstance(data.train_sampler, torch.utils.data.DistributedSampler):
        data.train_sampler.set_epoch(epoch)

    model.train()
    criterion.train()
    raw_model = unwrap_model(model)
    loss_accumulator = TensorAccumulator()
    diagnostic_accumulator = TensorAccumulator()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    skipped_steps = 0
    local_images = 0
    total_micro_batches = len(data.train_loader)

    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
        torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    window_bad = torch.zeros((), dtype=torch.int32, device=context.device)

    for micro_index, (samples, targets) in enumerate(data.train_loader):
        window_start = (micro_index // config.accumulation_steps) * config.accumulation_steps
        window_size = min(
            config.accumulation_steps, total_micro_batches - window_start
        )
        window_position = micro_index - window_start + 1
        should_sync = window_position == window_size
        diagnostics_enabled = (
            config.method == "bqr"
            and config.diagnostics_every > 0
            and should_sync
            and (global_step + 1) % config.diagnostics_every == 0
        )
        raw_model.set_diagnostics_enabled(diagnostics_enabled)
        samples, targets = move_batch(samples, targets, context.device)
        local_images += len(targets)

        sync_context = (
            model.no_sync()
            if context.distributed and not should_sync
            else nullcontext()
        )
        with sync_context:
            with autocast_context(config, context.device):
                outputs, mask_dict = model(samples, targets)
            outputs = float_tree(outputs)
            mask_dict = float_tree(mask_dict)
            loss_dict = criterion(outputs, targets, mask_dict)
            weighted = {
                name: value * criterion.weight_dict[name]
                for name, value in loss_dict.items()
                if name in criterion.weight_dict
            }
            total_loss = sum(weighted.values())
            window_bad.add_(
                (~torch.isfinite(total_loss.detach())).to(torch.int32)
            ).clamp_(max=1)
            scaler.scale(total_loss / window_size).backward()

        loss_accumulator.update(
            {
                "train_loss": total_loss,
                **{f"train_{name}": value for name, value in loss_dict.items()},
            }
        )

        if not should_sync:
            continue

        if context.distributed:
            torch.distributed.all_reduce(
                window_bad, op=torch.distributed.ReduceOp.MAX
            )
        if window_bad.item():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1}, optimizer step {global_step + 1}"
            )

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip
        )
        loss_accumulator.update({"train_grad_norm": grad_norm})

        if diagnostics_enabled:
            diagnostics = raw_model.training_diagnostics()
            if diagnostics:
                diagnostic_accumulator.update(diagnostics)
            bqr_norm = _bqr_grad_norm(model)
            if bqr_norm is not None:
                diagnostic_accumulator.update({"bqr_grad_norm": bqr_norm})

        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        step_skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
        if step_skipped:
            skipped_steps += 1
        else:
            optimizer_steps += 1
            global_step += 1
        optimizer.zero_grad(set_to_none=True)
        window_bad.zero_()

        if context.is_main and (
            global_step == 1 or global_step % 50 == 0 or micro_index + 1 == total_micro_batches
        ):
            memory = (
                torch.cuda.memory_allocated(context.device) / 1024**3
                if context.device.type == "cuda"
                else 0.0
            )
            print(
                f"epoch={epoch + 1:02d} step={global_step} "
                f"micro={micro_index + 1}/{total_micro_batches} "
                f"loss={float(total_loss.detach()):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} mem={memory:.1f}GB",
                flush=True,
            )

    raw_model.set_diagnostics_enabled(False)
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
        peak_memory_mb = torch.cuda.max_memory_allocated(context.device) / 1024**2
    else:
        peak_memory_mb = 0.0
    elapsed = time.perf_counter() - started

    loss_payloads = all_gather_objects(loss_accumulator.local_payload(), context)
    diagnostic_payloads = all_gather_objects(
        diagnostic_accumulator.local_payload(), context
    )
    metrics = _merge_mean_payloads(loss_payloads)
    metrics.update(_merge_diagnostics(diagnostic_payloads))

    timing = torch.tensor(
        [elapsed, peak_memory_mb, float(local_images), float(optimizer_steps), float(skipped_steps)],
        dtype=torch.float64,
        device=context.device,
    )
    if context.distributed:
        maximum = timing[:2].clone()
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        totals = timing[2:].clone()
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        timing = torch.cat((maximum, totals))
    metrics.update(
        epoch_train_time_seconds=float(timing[0].item()),
        peak_memory_mb=float(timing[1].item()),
        train_images=int(timing[2].item()),
        optimizer_steps=int(timing[3].item() / context.world_size),
        skipped_steps=int(timing[4].item() / context.world_size),
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
    write_run_metadata(
        config, context, subset_manifest=data.subset_manifest
    )
    scaler = _make_scaler(config)
    start_epoch = 0
    global_step = 0
    best_ap = float("-inf")
    history: list[dict] = []
    resume_path = resolve_resume_path(config, resume)
    if resume is None and config.latest_checkpoint.is_file():
        raise FileExistsError(
            f"Run already contains {config.latest_checkpoint}; use --resume auto or a new run name"
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
            print(
                f"Resumed {resume_path} at epoch {start_epoch + 1}, step {global_step}",
                flush=True,
            )

    if (
        config.stop_after_epoch is not None
        and start_epoch >= config.stop_after_epoch
    ):
        if context.is_main:
            print(
                f"Run has already reached planned stop epoch {config.stop_after_epoch}",
                flush=True,
            )
        return history

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
        epoch_number = epoch + 1
        should_evaluate = config.should_evaluate(epoch_number)
        val_metrics = (
            evaluate(config, context, model, postprocessors, data)
            if should_evaluate
            else {}
        )
        row = {
            "epoch": epoch_number,
            "global_step": global_step,
            "lr": learning_rate,
            "evaluated": should_evaluate,
            **train_metrics,
            **val_metrics,
        }
        history.append(row)

        is_best = False
        if should_evaluate:
            current_ap = float(val_metrics["AP"])
            is_best = current_ap > best_ap
            if is_best:
                best_ap = current_ap

        scheduler.step()
        is_final = epoch_number == config.epochs
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
            is_final=is_final,
            is_milestone=should_evaluate,
        )
        if context.is_main:
            append_jsonl(config.run_dir / "metrics.jsonl", row)
            evaluation_text = (
                f" AP={row['AP']:.4f} AP75={row['AP75']:.4f}"
                if should_evaluate
                else " AP=skipped"
            )
            print(
                f"epoch={epoch_number:02d}{evaluation_text} "
                f"train={row['epoch_train_time_seconds'] / 60:.1f}m",
                flush=True,
            )
        context.barrier()
        if config.stop_after_epoch is not None and epoch_number >= config.stop_after_epoch:
            if context.is_main and not is_final:
                print(
                    f"Stopped at planned checkpoint epoch {epoch_number}; resume with --resume auto",
                    flush=True,
                )
            break
    return history
