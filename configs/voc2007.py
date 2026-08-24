from pathlib import Path

from gt_guided_dino import ExperimentConfig


BASELINE = ExperimentConfig(
    data_root=Path("VOC2007"),
    method="baseline",
)

PROPOSED = ExperimentConfig(
    data_root=Path("VOC2007"),
    method="gt_guided_aux",
)

