from pathlib import Path

from gt_guided_dino.config import ExperimentConfig
from gt_guided_dino.data import VOCDetectionDataset, build_transforms, prepare_train_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_voc_split_and_deterministic_subset(tmp_path):
    config = ExperimentConfig(data_root=ROOT / "VOC2007", output_root=tmp_path)
    first = prepare_train_manifest(config)
    second = prepare_train_manifest(config)
    assert first == second
    assert len(first) == 1000
    assert len(set(first)) == 1000


def test_voc_sample_has_normalized_boxes(tmp_path):
    config = ExperimentConfig(data_root=ROOT / "VOC2007", output_root=tmp_path, train_limit=2)
    image_ids = prepare_train_manifest(config)
    dataset = VOCDetectionDataset(
        config.data_root,
        image_ids,
        transforms=build_transforms(config, train=False),
        exclude_difficult=True,
    )
    image, target = dataset[0]
    assert image.ndim == 3
    assert target["boxes"].shape[-1] == 4
    assert bool(((target["boxes"] >= 0) & (target["boxes"] <= 1)).all())

