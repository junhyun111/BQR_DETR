"""Class-score-free localization evaluation for DINO predictions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from .build import build_experiment
from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .data import build_loaders, move_targets_to_device


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    top_left, bottom_right = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((top_left + bottom_right) / 2, bottom_right - top_left), dim=-1)


def _box_iou_and_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU and generalized IoU for xyxy boxes."""
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp_min(1e-7)

    enclosing_top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_area = (enclosing_bottom_right - enclosing_top_left).clamp_min(0).prod(dim=-1)
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-7)
    return iou, giou


def _target_boxes_normalized(target: dict) -> torch.Tensor:
    """Return non-difficult target boxes in DINO-normalized cxcywh format."""
    boxes = target["boxes"]
    if "difficult" in target:
        boxes = boxes[~target["difficult"].bool()]
    # The vendored DINO Normalize transform has already converted targets from
    # absolute xyxy coordinates to normalized cxcywh coordinates.
    return boxes


def match_localization_only(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hungarian matching using only box geometry, never labels or scores."""
    if len(target_boxes) == 0:
        empty = torch.empty(0, dtype=torch.int64, device=predicted_boxes.device)
        return empty, empty
    predicted_xyxy = _cxcywh_to_xyxy(predicted_boxes)
    target_xyxy = _cxcywh_to_xyxy(target_boxes)
    _, pairwise_giou = _box_iou_and_giou(predicted_xyxy, target_xyxy)
    pairwise_l1 = torch.cdist(predicted_boxes, target_boxes, p=1)
    cost = l1_weight * pairwise_l1 + giou_weight * (1.0 - pairwise_giou)
    prediction_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(prediction_indices, dtype=torch.int64, device=predicted_boxes.device),
        torch.as_tensor(target_indices, dtype=torch.int64, device=predicted_boxes.device),
    )


def summarize_matched_boxes(records: dict[str, list[torch.Tensor]]) -> dict[str, float | int]:
    """Aggregate per-matched-box metrics into JSON-safe scalar values."""
    if not records["iou"]:
        return {"matched_boxes": 0}
    values = {name: torch.cat(parts).float() for name, parts in records.items()}
    return {
        "matched_boxes": int(values["iou"].numel()),
        "mean_matched_iou": float(values["iou"].mean()),
        "median_matched_iou": float(values["iou"].median()),
        "mean_matched_giou": float(values["giou"].mean()),
        "bbox_l1_error": float(values["l1"].mean()),
        "center_error": float(values["center"].mean()),
        "width_height_error": float(values["width_height"].mean()),
        "matched_iou50": float((values["iou"] >= 0.5).float().mean()),
        "matched_iou75": float((values["iou"] >= 0.75).float().mean()),
    }


@torch.no_grad()
def evaluate_localization_only(
    checkpoint_path: str | Path,
    *,
    data_root: str | Path | None = None,
    precision: str = "fp32",
    val_limit: int | None = None,
) -> dict:
    """Evaluate all queries with localization-only Hungarian matching.

    The model receives no GT boxes. GT is used only after inference to pair
    each non-difficult object with one of the 900 predicted boxes. Class
    logits, labels and confidence scores are deliberately unused.
    """
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    overrides = {}
    if data_root is not None:
        overrides["data_root"] = Path(data_root)
    if val_limit is not None:
        overrides["val_limit"] = val_limit
    config = ExperimentConfig.from_dict(payload["config"], **overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed-precision localization evaluation requires CUDA")
    _, val_loader = build_loaders(config)
    model, _, _ = build_experiment(config, device, use_common_initialization=False)
    load_checkpoint(path, config, model)
    model.eval()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    records: dict[str, list[torch.Tensor]] = {
        "iou": [], "giou": [], "l1": [], "center": [], "width_height": [],
    }
    started = time.perf_counter()
    for samples, targets in tqdm(val_loader, desc="Localization-only validation", leave=False):
        samples = samples.to(device)
        targets = move_targets_to_device(targets, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
            outputs = model(samples, None)
        for predicted_boxes, target in zip(outputs["pred_boxes"].float(), targets, strict=True):
            target_boxes = _target_boxes_normalized(target).float()
            prediction_indices, target_indices = match_localization_only(predicted_boxes, target_boxes)
            if len(target_indices) == 0:
                continue
            predicted = predicted_boxes[prediction_indices]
            matched_target = target_boxes[target_indices]
            iou, giou = _box_iou_and_giou(_cxcywh_to_xyxy(predicted), _cxcywh_to_xyxy(matched_target))
            diagonal = torch.arange(len(predicted), device=device)
            records["iou"].append(iou[diagonal, diagonal].cpu())
            records["giou"].append(giou[diagonal, diagonal].cpu())
            records["l1"].append((predicted - matched_target).abs().mean(dim=-1).cpu())
            records["center"].append(
                torch.linalg.vector_norm(predicted[:, :2] - matched_target[:, :2], dim=-1).cpu()
            )
            records["width_height"].append(
                (predicted[:, 2:] - matched_target[:, 2:]).abs().mean(dim=-1).cpu()
            )
    model.close()
    result = summarize_matched_boxes(records)
    result.update(
        {
            "evaluation": "all-query localization-only Hungarian matching",
            "images": len(val_loader.dataset),
            "seconds": time.perf_counter() - started,
            "matching_cost": "5.0 * L1(cx,cy,w,h) + 2.0 * (1 - GIoU)",
        }
    )
    # A limited run is a smoke check and must not overwrite the full-validation cache.
    if val_limit is None:
        output_path = config.run_dir / "localization_only.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
