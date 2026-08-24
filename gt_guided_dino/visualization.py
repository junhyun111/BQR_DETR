from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .config import VOC_CLASSES


def plot_history(history_path: str | Path):
    path = Path(history_path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [float(row["loss"]) for row in rows], marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Weighted loss")
    axes[1].plot(epochs, [float(row["lr"]) for row in rows], marker="o")
    axes[1].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    figure.tight_layout()
    return figure


def plot_predictions(
    predictions: list[dict],
    data_root: str | Path,
    *,
    count: int = 4,
    score_threshold: float = 0.4,
):
    selected = predictions[:count]
    figure, axes = plt.subplots(len(selected), 1, figsize=(10, 7 * len(selected)), squeeze=False)
    for axis, prediction in zip(axes[:, 0], selected):
        image_id = int(prediction["image_id"])
        image = Image.open(Path(data_root) / "JPEGImages" / f"{image_id:06d}.jpg").convert("RGB")
        axis.imshow(image)
        for score, label, box in zip(prediction["scores"], prediction["labels"], prediction["boxes"]):
            if score < score_threshold:
                continue
            left, top, right, bottom = box
            axis.add_patch(
                patches.Rectangle(
                    (left, top), right - left, bottom - top,
                    fill=False, edgecolor="lime", linewidth=2,
                )
            )
            axis.text(
                left, top, f"{VOC_CLASSES[int(label)]} {score:.2f}",
                color="black", fontsize=9,
                bbox={"facecolor": "lime", "alpha": 0.75, "pad": 2},
            )
        axis.set_title(f"VOC2007 {image_id:06d}")
        axis.axis("off")
    figure.tight_layout()
    return figure

