from __future__ import annotations

import math
import sys
import time
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .checkpoint import (
    load_training_state,
    resolve_resume_path,
    save_training_state,
)
from .config import ExperimentConfig
from .data import DataBundle, move_batch
from .distributed import (
    DistributedContext,
    all_gather_objects,
    unwrap_model,
)
from .evaluation import evaluate
from .logging_utils import (
    append_jsonl,
    write_run_metadata,
)


def _make_scaler(config: ExperimentConfig):
    """
    FP16 only uses GradScaler.

    BF16 has FP32-like exponent range and normally does not need
    dynamic loss scaling.
    """
    enabled = (
        config.precision == "fp16"
        and torch.cuda.is_available()
    )

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


def _autocast(
    config: ExperimentConfig,
    device: torch.device,
):
    enabled = (
        device.type == "cuda"
        and config.precision != "fp32"
    )

    if config.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def _weighted_loss(
    criterion,
    losses: dict[str, torch.Tensor],
) -> torch.Tensor:
    terms = [
        losses[name] * weight
        for name, weight
        in criterion.weight_dict.items()
        if name in losses
    ]

    if not terms:
        raise RuntimeError(
            "Official DINO criterion produced no weighted losses"
        )

    return torch.stack(terms).sum()


class EpochAccumulator:
    """
    Store only the metrics actually needed for experiment analysis.

    Scalars stay on GPU during the epoch and are transferred to CPU
    once at epoch end. This avoids .item() synchronization for every
    individual loss on every micro-batch.
    """

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def add(
        self,
        values: dict[str, torch.Tensor],
        *,
        count: int = 1,
    ) -> None:
        for name, value in values.items():
            if not torch.is_tensor(value):
                continue

            if value.numel() != 1:
                continue

            scalar = value.detach().float()

            if name in self.sums:
                self.sums[name] = (
                    self.sums[name]
                    + scalar * count
                )
            else:
                self.sums[name] = (
                    scalar * count
                )

            self.counts[name] = (
                self.counts.get(name, 0)
                + count
            )

    def add_sums(
        self,
        values: dict[str, torch.Tensor],
    ) -> None:
        for name, value in values.items():
            if not torch.is_tensor(value):
                continue

            if value.numel() != 1:
                continue

            scalar = value.detach().float()

            if name in self.sums:
                self.sums[name] = (
                    self.sums[name]
                    + scalar
                )
            else:
                self.sums[name] = scalar

            self.counts[name] = 1

    def gather(
        self,
        context: DistributedContext,
    ) -> tuple[
        dict[str, float],
        dict[str, int],
    ]:
        names = sorted(
            self.sums
        )

        if names:
            packed = torch.stack(
                [
                    self.sums[name]
                    for name in names
                ]
            )

            # Single synchronization for metric transfer.
            values = packed.cpu().tolist()
        else:
            values = []

        local_sums = dict(
            zip(
                names,
                values,
            )
        )

        payloads = all_gather_objects(
            (
                local_sums,
                self.counts,
            ),
            context,
        )

        total_sums: dict[str, float] = {}
        total_counts: dict[str, int] = {}

        for sums, counts in payloads:
            for name, value in sums.items():
                total_sums[name] = (
                    total_sums.get(
                        name,
                        0.0,
                    )
                    + value
                )

            for name, value in counts.items():
                total_counts[name] = (
                    total_counts.get(
                        name,
                        0,
                    )
                    + value
                )

        return (
            total_sums,
            total_counts,
        )


def _finalize_train_metrics(
    accumulator: EpochAccumulator,
    context: DistributedContext,
    *,
    epoch_seconds: float,
    optimizer_steps: int,
    skipped_steps: int,
    peak_memory_mb: float,
) -> dict[str, float]:
    sums, counts = accumulator.gather(
        context
    )

    row: dict[str, float] = {}

    for name, total in sums.items():
        if name.startswith(
            "diag_"
        ):
            continue

        count = counts.get(
            name,
            0,
        )

        if count:
            row[name] = (
                total / count
            )

    valid = sums.get(
        "diag_bqr_valid_queries",
        0.0,
    )

    base = {
        "bqr_gate":
            "diag_bqr_gate_sum",
        "bqr_offset":
            "diag_bqr_offset_abs_sum",
        "bqr_entropy":
            "diag_bqr_attention_entropy_sum",
        "bqr_region_norm":
            "diag_bqr_region_norm_sum",
        "bqr_fusion_delta":
            "diag_bqr_fusion_delta_norm_sum",
    }

    for output_name, sum_name in base.items():
        if valid:
            row[output_name] = (
                sums.get(
                    sum_name,
                    0.0,
                )
                / valid
            )
        else:
            row[output_name] = (
                float("nan")
            )

    row[
        "bqr_valid_queries"
    ] = valid

    for size_name in (
        "small",
        "medium",
        "large",
    ):
        size_count = sums.get(
            f"diag_bqr_{size_name}_queries",
            0.0,
        )

        row[
            f"bqr_{size_name}_queries"
        ] = size_count

        if size_count:
            row[
                f"bqr_{size_name}_gt_inside_ratio"
            ] = (
                sums.get(
                    f"diag_bqr_{size_name}_gt_inside_sum",
                    0.0,
                )
                / size_count
            )
        else:
            row[
                f"bqr_{size_name}_gt_inside_ratio"
            ] = float("nan")

        for level in range(4):
            if size_count:
                value = (
                    sums.get(
                        f"diag_bqr_{size_name}_level"
                        f"{level}_attention_sum",
                        0.0,
                    )
                    / size_count
                )
            else:
                value = float("nan")

            row[
                f"bqr_{size_name}_level"
                f"{level}_attention_mass"
            ] = value

        for polarity in (
            "positive",
            "negative",
        ):
            polarity_count = sums.get(
                f"diag_bqr_{size_name}_"
                f"{polarity}_queries",
                0.0,
            )

            if polarity_count:
                value = (
                    sums.get(
                        f"diag_bqr_{size_name}_"
                        f"{polarity}_gt_inside_sum",
                        0.0,
                    )
                    / polarity_count
                )
            else:
                value = float("nan")

            row[
                f"bqr_{size_name}_{polarity}"
                f"_gt_inside_ratio"
            ] = value

    row.update(
        {
            "epoch_train_time_seconds":
                epoch_seconds,
            "optimizer_steps":
                optimizer_steps,
            "amp_skipped_steps":
                skipped_steps,
            "peak_gpu_memory_mb":
                peak_memory_mb,
        }
    )

    return row


def _selected_loss_metrics(
    losses: dict[str, torch.Tensor],
    criterion,
    total_loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Avoid launching dozens of tiny accumulation kernels for every
    auxiliary loss. Keep the metrics needed for comparison/analysis.
    """

    raw: dict[str, torch.Tensor] = {
        "train_total_loss":
            total_loss
    }

    selected = (
        "loss_ce",
        "loss_bbox",
        "loss_giou",
        "loss_ce_dn",
        "loss_bbox_dn",
        "loss_giou_dn",
    )

    for name in selected:
        value = losses.get(
            name
        )

        if (
            value is not None
            and torch.is_tensor(value)
            and value.numel() == 1
        ):
            raw[name] = value

            if name in criterion.weight_dict:
                raw[
                    f"weighted_{name}"
                ] = (
                    value
                    * criterion.weight_dict[
                        name
                    ]
                )

    return raw


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
) -> tuple[
    dict[str, float],
    int,
]:
    if (
        data.train_loader is None
        or data.train_sampler is None
    ):
        raise RuntimeError(
            "Training data loader was not built"
        )

    model.train()
    criterion.train()

    if hasattr(
        data.train_sampler,
        "set_epoch",
    ):
        data.train_sampler.set_epoch(
            epoch
        )

    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            context.device
        )

        # Synchronize only at the epoch timing boundary.
        torch.cuda.synchronize(
            context.device
        )

    started = time.perf_counter()

    accumulator = EpochAccumulator()

    optimizer_steps = 0
    skipped_steps = 0

    accumulation_steps = (
        config.accumulation_steps
    )

    total_micro_batches = len(
        data.train_loader
    )

    # Only complete accumulation windows are used.
    #
    # batch 4, accumulation 4:
    # every optimizer update receives exactly 16 images.
    usable_micro_batches = (
        total_micro_batches
        - (
            total_micro_batches
            % accumulation_steps
        )
    )

    total_optimizer_steps = (
        usable_micro_batches
        // accumulation_steps
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    progress = None

    if context.is_main:
        progress = tqdm(
            total=total_optimizer_steps,
            desc=f"Epoch {epoch + 1:02d}",
            file=sys.stdout,
            disable=False,

            # Fixed width is more reliable through
            # docker logs -f than dynamic terminal probing.
            dynamic_ncols=False,
            ncols=150,

            leave=True,
            mininterval=1.0,
            maxinterval=5.0,
            smoothing=0.1,

            bar_format=(
                "{desc}: "
                "{percentage:6.2f}%|{bar}| "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )

    micro_in_window = 0
    last_total_loss = None
    last_loss_value = None

    for micro_batch_index, (
        samples,
        targets,
    ) in enumerate(
        data.train_loader
    ):
        if (
            micro_batch_index
            >= usable_micro_batches
        ):
            break

        micro_in_window += 1

        final_micro = (
            micro_in_window
            == accumulation_steps
        )

        next_global_step = (
            global_step + 1
        )

        diagnostics = (
            final_micro
            and config.method
            == "bqr_dn_v2"
            and config.diagnostics_every > 0
            and (
                next_global_step
                % config.diagnostics_every
                == 0
            )
        )

        unwrap_model(
            model
        ).set_diagnostics_enabled(
            diagnostics
        )

        # Batch is moved immediately after DataLoader yields it.
        # There is no accumulation-window list buffering.
        samples, targets = move_batch(
            samples,
            targets,
            context.device,
        )

        if (
            isinstance(
                model,
                DistributedDataParallel,
            )
            and not final_micro
        ):
            sync_context = (
                model.no_sync()
            )
        else:
            sync_context = (
                nullcontext()
            )

        with sync_context:
            # Match official DINO AMP structure:
            # forward + criterion stay in the autocast region.
            with _autocast(
                config,
                context.device,
            ):
                outputs = model(
                    samples,
                    targets,
                )

                losses = criterion(
                    outputs,
                    targets,
                )

                total_loss = _weighted_loss(
                    criterion,
                    losses,
                )

            last_total_loss = (
                total_loss
            )

            backward_loss = (
                total_loss
                / accumulation_steps
            )

            scaler.scale(
                backward_loss
            ).backward()

        accumulator.add(
            _selected_loss_metrics(
                losses,
                criterion,
                total_loss,
            )
        )

        if diagnostics:
            diagnostic_values = (
                unwrap_model(
                    model
                ).training_diagnostics()
            )

            accumulator.add_sums(
                {
                    f"diag_{name}":
                        value
                    for name, value
                    in diagnostic_values.items()
                }
            )

        # Drop references to large decoder output structures ASAP.
        del outputs
        del losses
        del backward_loss

        if not final_micro:
            continue

        if scaler.is_enabled():
            scaler.unscale_(
                optimizer
            )

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip,
            )
        )

        previous_scale = (
            scaler.get_scale()
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        step_skipped = (
            scaler.is_enabled()
            and (
                scaler.get_scale()
                < previous_scale
            )
        )

        if step_skipped:
            skipped_steps += 1
        else:
            optimizer_steps += 1
            global_step += 1

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulator.add(
            {
                "grad_norm":
                    grad_norm
            }
        )

        micro_in_window = 0

        # Progress itself updates every optimizer window.
        if progress is not None:
            progress.update(1)

        # Only synchronize for scalar loss reporting every 50
        # optimizer updates. No .item() on every iteration.
        should_refresh_stats = (
            optimizer_steps == 1
            or (
                optimizer_steps > 0
                and optimizer_steps % 50 == 0
            )
        )

        if (
            should_refresh_stats
            and last_total_loss is not None
        ):
            last_loss_value = float(
                last_total_loss.detach()
            )

            if not math.isfinite(
                last_loss_value
            ):
                if progress is not None:
                    progress.close()

                raise FloatingPointError(
                    "Non-finite loss detected at "
                    f"epoch {epoch + 1}, "
                    f"step {global_step}"
                )

            if context.device.type == "cuda":
                memory_gb = (
                    torch.cuda.memory_allocated(
                        context.device
                    )
                    / 1024**3
                )

                peak_gb = (
                    torch.cuda.max_memory_allocated(
                        context.device
                    )
                    / 1024**3
                )
            else:
                memory_gb = 0.0
                peak_gb = 0.0

            if progress is not None:
                postfix = {
                    "loss":
                        f"{last_loss_value:.3f}",
                    "lr":
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                    "step":
                        global_step,
                    "mem":
                        f"{memory_gb:.1f}G",
                    "peak":
                        f"{peak_gb:.1f}G",
                }

                if scaler.is_enabled():
                    postfix[
                        "scale"
                    ] = (
                        f"{scaler.get_scale():.0f}"
                    )

                    postfix[
                        "skips"
                    ] = (
                        skipped_steps
                    )

                progress.set_postfix(
                    postfix,
                    refresh=True,
                )

    if progress is not None:
        progress.close()

    if context.device.type == "cuda":
        # Epoch-end timing boundary.
        torch.cuda.synchronize(
            context.device
        )

        peak_memory = (
            torch.cuda.max_memory_allocated(
                context.device
            )
            / 1024**2
        )
    else:
        peak_memory = 0.0

    elapsed = (
        time.perf_counter()
        - started
    )

    timing = torch.tensor(
        [
            elapsed,
            peak_memory,
        ],
        dtype=torch.float64,
        device=context.device,
    )

    if context.distributed:
        torch.distributed.all_reduce(
            timing,
            op=torch.distributed.ReduceOp.MAX,
        )

    metrics = _finalize_train_metrics(
        accumulator,
        context,
        epoch_seconds=float(
            timing[0]
        ),
        optimizer_steps=optimizer_steps,
        skipped_steps=skipped_steps,
        peak_memory_mb=float(
            timing[1]
        ),
    )

    return (
        metrics,
        global_step,
    )


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
        config,
        context,
    )

    scaler = _make_scaler(
        config
    )

    start_epoch = 0
    global_step = 0
    best_ap = float("-inf")

    history: list[dict] = []

    resume_path = (
        resolve_resume_path(
            config,
            resume,
        )
    )

    if (
        resume is None
        and resume_path is None
        and config.latest_checkpoint.is_file()
    ):
        raise FileExistsError(
            "Run already contains a checkpoint: "
            f"{config.latest_checkpoint}. "
            "Use a new --run-name or remove "
            "the old run directory."
        )

    if resume_path is not None:
        (
            start_epoch,
            global_step,
            best_ap,
            history,
        ) = load_training_state(
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
                f"Resumed from {resume_path} "
                f"at epoch {start_epoch + 1}",
                flush=True,
            )

    for epoch in range(
        start_epoch,
        config.epochs,
    ):
        learning_rate = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        (
            train_metrics,
            global_step,
        ) = train_one_epoch(
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

        val_metrics = evaluate(
            config,
            context,
            model,
            postprocessors,
            data,
        )

        row = {
            "epoch":
                epoch + 1,
            "global_step":
                global_step,
            "lr":
                learning_rate,
            **train_metrics,
            **val_metrics,
        }

        history.append(
            row
        )

        current_ap = float(
            val_metrics["AP"]
        )

        is_best = (
            current_ap
            > best_ap
        )

        if is_best:
            best_ap = (
                current_ap
            )

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
            is_final=(
                epoch + 1
                == config.epochs
            ),
        )

        if context.is_main:
            append_jsonl(
                config.run_dir
                / "metrics.jsonl",
                row,
            )

            print(
                f"epoch={epoch + 1:02d} "
                f"AP={row['AP']:.4f} "
                f"AP50={row['AP50']:.4f} "
                f"AP75={row['AP75']:.4f} "
                f"train="
                f"{row['epoch_train_time_seconds'] / 60:.1f}m "
                f"val="
                f"{row['val_time_seconds'] / 60:.1f}m",
                flush=True,
            )

        context.barrier()

    return history