from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import VOC_CLASSES


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float64)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    area_box = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return intersection / np.maximum(area_box + areas - intersection, 1e-12)


def voc_ap(recall: np.ndarray, precision: np.ndarray, *, use_07_metric: bool) -> float:
    if use_07_metric:
        return float(
            sum(
                np.max(precision[recall >= threshold]) if np.any(recall >= threshold) else 0.0
                for threshold in np.arange(0.0, 1.1, 0.1)
            )
            / 11.0
        )
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def evaluate_voc_predictions(
    predictions: list[dict],
    dataset,
    *,
    iou_threshold: float = 0.5,
) -> dict:
    ground_truth: dict[int, dict[int, dict]] = {}
    positives = np.zeros(len(VOC_CLASSES), dtype=np.int64)
    for image_id in dataset.image_ids:
        annotation = dataset.get_ground_truth(image_id)
        by_class: dict[int, dict] = {}
        for class_index in range(len(VOC_CLASSES)):
            objects = [obj for obj in annotation["objects"] if obj["label"] == class_index]
            boxes = np.asarray([obj["box"] for obj in objects], dtype=np.float64).reshape(-1, 4)
            difficult = np.asarray([bool(obj["difficult"]) for obj in objects], dtype=bool)
            by_class[class_index] = {"boxes": boxes, "difficult": difficult}
            positives[class_index] += int((~difficult).sum())
        ground_truth[int(image_id)] = by_class

    detections: dict[int, list[tuple[int, float, np.ndarray]]] = defaultdict(list)
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        for score, label, box in zip(prediction["scores"], prediction["labels"], prediction["boxes"]):
            detections[int(label)].append((image_id, float(score), np.asarray(box, dtype=np.float64)))

    ap07 = {}
    ap_integral = {}
    for class_index, class_name in enumerate(VOC_CLASSES):
        ranked = sorted(detections[class_index], key=lambda item: item[1], reverse=True)
        matched = {
            image_id: np.zeros(len(values[class_index]["boxes"]), dtype=bool)
            for image_id, values in ground_truth.items()
        }
        true_positive = np.zeros(len(ranked), dtype=np.float64)
        false_positive = np.zeros(len(ranked), dtype=np.float64)
        ignored = np.zeros(len(ranked), dtype=bool)
        for index, (image_id, _, box) in enumerate(ranked):
            record = ground_truth[image_id][class_index]
            overlaps = _iou(box, record["boxes"])
            if len(overlaps) == 0 or float(overlaps.max()) < iou_threshold:
                false_positive[index] = 1.0
                continue
            best = int(overlaps.argmax())
            if record["difficult"][best]:
                ignored[index] = True
            elif not matched[image_id][best]:
                true_positive[index] = 1.0
                matched[image_id][best] = True
            else:
                false_positive[index] = 1.0
        true_positive = true_positive[~ignored]
        false_positive = false_positive[~ignored]
        true_positive = np.cumsum(true_positive)
        false_positive = np.cumsum(false_positive)
        recall = true_positive / max(int(positives[class_index]), 1)
        precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
        ap07[class_name] = voc_ap(recall, precision, use_07_metric=True)
        ap_integral[class_name] = voc_ap(recall, precision, use_07_metric=False)

    return {
        "voc07_map50": float(np.mean(list(ap07.values()))),
        "voc_map50_integral": float(np.mean(list(ap_integral.values()))),
        "ap50_voc07_by_class": ap07,
        "ap50_integral_by_class": ap_integral,
    }

