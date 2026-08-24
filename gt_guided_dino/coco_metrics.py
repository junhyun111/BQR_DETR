"""COCO-style detection metrics for VOC predictions.

VOC2007 natively defines AP@0.5 only.  This adapter evaluates the existing
VOC validation annotations with the standard COCO IoU, area and recall grids,
so experiments can also compare AP, AP50, AP75, AP by object size and AR.
VOC ``difficult`` objects are represented as COCO crowd annotations and are
therefore ignored when a detection overlaps them.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import numpy as np

from .config import VOC_CLASSES


COCO_STAT_NAMES = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)


def _xyxy_to_xywh(box: list[float]) -> list[float] | None:
    left, top, right, bottom = (float(value) for value in box)
    width = max(right - left, 0.0)
    height = max(bottom - top, 0.0)
    if width <= 0.0 or height <= 0.0:
        return None
    return [left, top, width, height]


def _build_coco_ground_truth(dataset) -> Any:
    from pycocotools.coco import COCO

    annotations = []
    images = []
    annotation_id = 1
    for image_id in dataset.image_ids:
        numeric_id = int(image_id)
        record = dataset.get_ground_truth(numeric_id)
        images.append(
            {"id": numeric_id, "width": record["width"], "height": record["height"]}
        )
        for object_ in record["objects"]:
            bbox = _xyxy_to_xywh(object_["box"])
            if bbox is None:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": numeric_id,
                    "category_id": int(object_["label"]) + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": int(bool(object_["difficult"])),
                }
            )
            annotation_id += 1

    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": class_name}
            for index, class_name in enumerate(VOC_CLASSES)
        ],
    }
    coco.createIndex()
    return coco


def _build_coco_detections(predictions: list[dict]) -> list[dict]:
    detections = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        for score, label, box in zip(
            prediction["scores"], prediction["labels"], prediction["boxes"], strict=True
        ):
            bbox = _xyxy_to_xywh(box)
            if bbox is not None:
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "bbox": bbox,
                        "score": float(score),
                    }
                )
    return detections


def compute_coco_metrics(predictions: list[dict], dataset, *, verbose: bool = False) -> dict:
    """Return standard COCO-style summary statistics and per-class AP.

    ``predictions`` is the prediction list saved by :func:`evaluate_checkpoint`.
    The returned AP/AR values are fractions in ``[0, 1]``, not percentages.
    """
    from pycocotools.cocoeval import COCOeval

    coco_gt = _build_coco_ground_truth(dataset)
    detections = _build_coco_detections(predictions)
    if not detections:
        return {
            **{name: float("nan") for name in COCO_STAT_NAMES},
            "AP_by_class": {class_name: float("nan") for class_name in VOC_CLASSES},
            "AP50_by_class": {class_name: float("nan") for class_name in VOC_CLASSES},
        }
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = sorted(int(image_id) for image_id in dataset.image_ids)
    evaluator.params.catIds = list(range(1, len(VOC_CLASSES) + 1))
    evaluator.params.maxDets = [1, 10, 100]
    if verbose:
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    else:
        with redirect_stdout(StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

    precision = evaluator.eval["precision"]  # [IoU, recall, class, area, maxDet]
    ap_by_class = {}
    ap50_by_class = {}
    for class_index, class_name in enumerate(VOC_CLASSES):
        ap_values = precision[:, :, class_index, 0, -1]
        ap_values = ap_values[ap_values > -1]
        ap_by_class[class_name] = float(np.mean(ap_values)) if ap_values.size else float("nan")
        ap50_values = precision[0, :, class_index, 0, -1]
        ap50_values = ap50_values[ap50_values > -1]
        ap50_by_class[class_name] = float(np.mean(ap50_values)) if ap50_values.size else float("nan")

    summary = {
        name: float(value) if value >= 0 else float("nan")
        for name, value in zip(COCO_STAT_NAMES, evaluator.stats, strict=True)
    }
    return {
        **summary,
        "AP_by_class": ap_by_class,
        "AP50_by_class": ap50_by_class,
    }
