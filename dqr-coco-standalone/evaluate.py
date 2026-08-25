from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from dqr_coco.checkpoint import load_training_state
from dqr_coco.config import ExperimentConfig
from dqr_coco.data import build_data
from dqr_coco.distributed import (
    cleanup_distributed,
    initialize_distributed,
    require_compiled_msda,
)
from dqr_coco.evaluation import evaluate
from dqr_coco.model import build_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a dqr-coco checkpoint on val2017.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("COCO_ROOT", "data/coco")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "8")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "2")))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=os.environ.get("INFERENCE_PRECISION", "fp16"))
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(
        checkpoint["config"],
        data_root=args.data_root,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
        val_limit=args.val_limit,
    )
    context = initialize_distributed(allow_cpu=args.allow_cpu)
    try:
        require_compiled_msda(context.device)
        data = build_data(config, context, training=False)
        model, _, postprocessors = build_experiment(
            config, context, use_common_initialization=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics = evaluate(config, context, model, postprocessors, data)
        if context.is_main:
            suffix = "" if args.val_limit is None else f"_val{args.val_limit}"
            output = checkpoint_path.parent.parent / f"evaluation{suffix}.json"
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    {"checkpoint": str(checkpoint_path), "metrics": metrics},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, output)
            print(json.dumps(metrics, indent=2, sort_keys=True))
        context.barrier()
    finally:
        cleanup_distributed(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
