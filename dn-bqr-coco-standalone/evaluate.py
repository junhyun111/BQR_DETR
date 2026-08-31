from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from bqr_dn.config import ExperimentConfig
from bqr_dn.data import build_data
from bqr_dn.distributed import (
    cleanup_distributed,
    initialize_distributed,
    require_compiled_msda,
)
from bqr_dn.evaluation import evaluate
from bqr_dn.model import build_experiment
from bqr_dn.upstream import upstream_commit


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DN-BQR checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/artifacts"))
    parser.add_argument("--torch-cache", type=Path, default=Path("/workspace/torch-cache"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("upstream_commit") != upstream_commit():
        raise ValueError("Checkpoint upstream commit mismatch")
    allowed = {field.name for field in fields(ExperimentConfig)}
    values = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in allowed
    }
    values.update(
        data_root=args.data_root,
        output_root=args.output_root,
        torch_cache=args.torch_cache,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        stop_after_epoch=None,
    )
    if args.precision is not None:
        values["precision"] = args.precision
    config = ExperimentConfig(**values)
    context = initialize_distributed()
    try:
        require_compiled_msda(context.device)
        model, _, postprocessors = build_experiment(
            config, context, use_common_initialization=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        data = build_data(config, context, training=False)
        metrics = evaluate(config, context, model, postprocessors, data)
        if context.is_main:
            print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
