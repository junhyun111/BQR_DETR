from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bqr_dn.config import ExperimentConfig
from bqr_dn.data import build_data
from bqr_dn.distributed import (
    cleanup_distributed,
    initialize_distributed,
    require_compiled_msda,
    seed_everything,
)
from bqr_dn.engine import train
from bqr_dn.model import build_experiment
from bqr_dn.optimizer import build_optimizer


def _epochs(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated epoch numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one evaluation epoch is required")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Train official DN-Deformable-DETR baseline or clean-GT BQR"
    )
    result.add_argument("--method", choices=("baseline", "bqr"), default="bqr")
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--torch-cache", type=Path, required=True)
    result.add_argument("--run-name", default=None)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--stop-after-epoch", type=int, default=None)
    result.add_argument("--eval-epochs", type=_epochs, default=(5, 10, 15, 20))
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--accumulation-steps", type=int, default=4)
    result.add_argument("--target-global-batch-size", type=int, default=16)
    result.add_argument("--eval-batch-size", type=int, default=2)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--persistent-workers", action="store_true")
    result.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--backbone-lr", type=float, default=1e-5)
    result.add_argument("--lr-drop-epoch", type=int, default=16)
    result.add_argument("--train-size", type=int, default=10_000)
    result.add_argument("--subset-seed", type=int, default=42)
    result.add_argument("--val-limit", type=int, default=None)
    result.add_argument("--diagnostics-every", type=int, default=100)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--resume", default="auto")
    result.add_argument("--deterministic", action="store_true")
    result.add_argument("--allow-cpu", action="store_true")
    result.add_argument("--enc-layers", type=int, default=6, help=argparse.SUPPRESS)
    result.add_argument("--dec-layers", type=int, default=6, help=argparse.SUPPRESS)
    result.add_argument("--num-queries", type=int, default=300, help=argparse.SUPPRESS)
    result.add_argument("--dn-scalar", type=int, default=5, help=argparse.SUPPRESS)
    return result


def main() -> None:
    args = parser().parse_args()
    context = initialize_distributed(allow_cpu=args.allow_cpu)
    try:
        config = ExperimentConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            torch_cache=args.torch_cache,
            method=args.method,
            seed=args.seed,
            run_name=args.run_name or None,
            epochs=args.epochs,
            stop_after_epoch=args.stop_after_epoch,
            eval_epochs=args.eval_epochs,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            target_global_batch_size=args.target_global_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            precision=args.precision,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            lr_drop_epoch=args.lr_drop_epoch,
            train_size=args.train_size,
            subset_seed=args.subset_seed,
            val_limit=args.val_limit,
            diagnostics_every=args.diagnostics_every,
            deterministic=args.deterministic,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            num_queries=args.num_queries,
            dn_scalar=args.dn_scalar,
        )
        actual_global_batch = (
            config.batch_size * config.accumulation_steps * context.world_size
        )
        if actual_global_batch != config.target_global_batch_size:
            raise ValueError(
                "Effective global batch mismatch: "
                f"{config.batch_size} x {config.accumulation_steps} x "
                f"{context.world_size} = {actual_global_batch}, expected "
                f"{config.target_global_batch_size}"
            )
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        seed_everything(config.seed, context.rank, config.deterministic)
        require_compiled_msda(context.device)
        model, criterion, postprocessors = build_experiment(config, context)
        data = build_data(config, context, training=True)
        optimizer, scheduler = build_optimizer(model, config)
        if context.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                output_device=context.local_rank if context.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        if context.is_main:
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            print(
                f"method={config.method} upstream=ff3902a world_size={context.world_size} "
                f"batch_per_gpu={config.batch_size} accumulation={config.accumulation_steps} "
                f"effective_batch={actual_global_batch} precision={config.precision} "
                f"train_size={config.train_size} parameters={parameter_count:,}",
                flush=True,
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
            resume=args.resume,
        )
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
