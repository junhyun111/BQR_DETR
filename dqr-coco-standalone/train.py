from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from dqr_coco.config import ExperimentConfig
from dqr_coco.data import build_data
from dqr_coco.distributed import (
    cleanup_distributed,
    initialize_distributed,
    require_compiled_msda,
)
from dqr_coco.engine import train
from dqr_coco.model import build_experiment
from dqr_coco.optimizer import build_optimizer


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in ("none", "full"):
        return None
    return _positive_int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train official DINO R50-4scale or BQR-DN V2 on COCO 2017."
    )
    parser.add_argument(
        "--method",
        choices=("baseline", "bqr_dn_v2"),
        default=os.environ.get("METHOD", "bqr_dn_v2"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("COCO_ROOT", "data/coco"))
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "artifacts"))
    )
    parser.add_argument(
        "--torch-cache", type=Path, default=Path(os.environ.get("TORCH_HOME", ".cache/torch"))
    )
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME") or None)
    parser.add_argument("--epochs", type=_positive_int, default=int(os.environ.get("EPOCHS", "12")))
    parser.add_argument("--batch-size", type=_positive_int, default=int(os.environ.get("BATCH_SIZE", "2")))
    parser.add_argument(
        "--accumulation-steps", type=_positive_int, default=int(os.environ.get("ACCUMULATION_STEPS", "4"))
    )
    parser.add_argument(
        "--target-global-batch-size", type=_positive_int, default=int(os.environ.get("TARGET_GLOBAL_BATCH_SIZE", "16"))
    )
    parser.add_argument("--eval-batch-size", type=_positive_int, default=int(os.environ.get("EVAL_BATCH_SIZE", "2")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "8")))
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default=os.environ.get("PRECISION", "fp16")
    )
    parser.add_argument("--diagnostics-every", type=int, default=int(os.environ.get("DIAGNOSTICS_EVERY", "100")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--train-limit", type=_optional_positive_int, default=None)
    parser.add_argument("--val-limit", type=_optional_positive_int, default=None)
    parser.add_argument("--resume", default=os.environ.get("RESUME", "auto"))
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        torch_cache=args.torch_cache,
        method=args.method,
        seed=args.seed,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        target_global_batch_size=args.target_global_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
        diagnostics_every=args.diagnostics_every,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        deterministic=args.deterministic,
    )
    context = initialize_distributed(allow_cpu=args.allow_cpu)
    try:
        config.validate_world_size(context.world_size)
        require_compiled_msda(context.device)
        if context.is_main:
            print(
                f"method={config.method} world_size={context.world_size} "
                f"batch_per_gpu={config.batch_size} accumulation={config.accumulation_steps} "
                f"effective_batch={config.target_global_batch_size} precision={config.precision}",
                flush=True,
            )
        data = build_data(config, context, training=True)
        model, criterion, postprocessors = build_experiment(config, context)
        optimizer, scheduler = build_optimizer(model, config)
        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        resume = (
            None
            if args.resume is None or str(args.resume).lower() in ("none", "false", "no")
            else args.resume
        )
        train(
            config,
            context,
            model,
            criterion,
            postprocessors,
            optimizer,
            scheduler,
            data,
            resume=resume,
        )
        context.barrier()
    finally:
        cleanup_distributed(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
