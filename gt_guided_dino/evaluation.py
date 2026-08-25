from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .build import build_experiment
from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .data import VOCDetectionDataset, build_loaders, move_targets_to_device
from .metrics import evaluate_voc_predictions


@torch.no_grad()
def evaluate_model(
    model,
    postprocessors: dict,
    val_loader,
    dataset: VOCDetectionDataset,
    device: torch.device,
    *,
    precision: str = "fp32",
) -> dict:
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed-precision evaluation requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    model.eval()
    predictions = []
    started = time.perf_counter()
    for samples, targets in tqdm(val_loader, desc="VOC2007 validation", leave=False):
        samples = samples.to(device)
        device_targets = move_targets_to_device(targets, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
            outputs = model(samples, None)
        float_outputs = {
            **outputs,
            "pred_logits": outputs["pred_logits"].float(),
            "pred_boxes": outputs["pred_boxes"].float(),
        }
        original_sizes = torch.stack([target["orig_size"] for target in device_targets])
        batch_predictions = postprocessors["bbox"](float_outputs, original_sizes)
        for target, prediction in zip(device_targets, batch_predictions):
            predictions.append(
                {
                    "image_id": int(target["image_id"].item()),
                    "scores": prediction["scores"].detach().cpu().tolist(),
                    "labels": prediction["labels"].detach().cpu().tolist(),
                    "boxes": prediction["boxes"].detach().cpu().tolist(),
                }
            )
    metrics = evaluate_voc_predictions(predictions, dataset)
    metrics["images"] = len(dataset)
    metrics["seconds"] = time.perf_counter() - started
    return {"metrics": metrics, "predictions": predictions}


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    data_root: str | Path | None = None,
    precision: str = "fp32",
    val_limit: int | None = None,
) -> dict:
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    overrides = {}
    if data_root is not None:
        overrides["data_root"] = Path(data_root)
    if val_limit is not None:
        overrides["val_limit"] = val_limit
    config = ExperimentConfig.from_dict(checkpoint["config"], **overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = build_loaders(config)
    dataset = val_loader.dataset
    model, _, postprocessors = build_experiment(
        config, device, use_common_initialization=False
    )
    load_checkpoint(path, config, model)
    result = evaluate_model(
        model,
        postprocessors,
        val_loader,
        dataset,
        device,
        precision=precision,
    )
    # A limited validation pass is a diagnostic and must never replace the
    # full-validation cache consumed by the comparison notebooks.
    suffix = "" if val_limit is None else f"_val{val_limit}"
    output_path = config.run_dir / f"evaluation{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    model.close()
    return result
