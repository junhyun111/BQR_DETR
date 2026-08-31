from __future__ import annotations

import time

import torch

from .config import ExperimentConfig
from .data import DataBundle, move_batch
from .distributed import DistributedContext, all_reduce_sum
from .upstream import ensure_upstream_imports


ensure_upstream_imports()
from datasets.coco_eval import CocoEvaluator  # noqa: E402


COCO_STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_S",
    "AP_M",
    "AP_L",
    "AR1",
    "AR10",
    "AR100",
    "AR_S",
    "AR_M",
    "AR_L",
)


def autocast_context(config: ExperimentConfig, device: torch.device):
    enabled = device.type == "cuda" and config.precision != "fp32"
    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def float_tree(value):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: float_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [float_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(float_tree(item) for item in value)
    return value


@torch.no_grad()
def evaluate(
    config: ExperimentConfig,
    context: DistributedContext,
    model,
    postprocessors: dict,
    data: DataBundle,
) -> dict[str, float]:
    model.eval()
    evaluator = CocoEvaluator(data.coco_api, ["bbox"])
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    local_images = 0
    for samples, targets in data.val_loader:
        samples, targets = move_batch(samples, targets, context.device)
        with autocast_context(config, context.device):
            outputs, _ = model(samples)
        outputs = float_tree(outputs)
        original_sizes = torch.stack([target["orig_size"] for target in targets])
        results = postprocessors["bbox"](outputs, original_sizes)
        evaluator.update(
            {
                int(target["image_id"].item()): {
                    name: tensor.detach().cpu() for name, tensor in result.items()
                }
                for target, result in zip(targets, results)
            }
        )
        local_images += len(targets)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    if context.is_main:
        evaluator.summarize()
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    elapsed = torch.tensor(
        time.perf_counter() - started, dtype=torch.float64, device=context.device
    )
    if context.distributed:
        torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
    image_count = torch.tensor(float(local_images), device=context.device)
    all_reduce_sum(image_count, context)
    stats = evaluator.coco_eval["bbox"].stats
    metrics = {
        name: float(stats[index]) for index, name in enumerate(COCO_STAT_NAMES)
    }
    metrics["val_time_seconds"] = float(elapsed.item())
    metrics["val_images"] = int(image_count.item())
    return metrics
