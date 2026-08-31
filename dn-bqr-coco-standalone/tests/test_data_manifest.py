from __future__ import annotations

import json

import torch

from bqr_dn.config import ExperimentConfig
from bqr_dn.data import prepare_subset_manifest
from bqr_dn.distributed import DistributedContext


class FakeCoco:
    def __init__(self, image_ids):
        self.imgToAnns = {
            image_id: [
                {
                    "category_id": image_id % 3 + 1,
                    "area": float(image_id * 500),
                    "iscrowd": 0,
                }
            ]
            for image_id in image_ids
        }


class FakeDataset:
    def __init__(self, image_ids):
        self.ids = image_ids
        self.coco = FakeCoco(image_ids)


def test_subset_manifest_is_seeded_random_and_reused(tmp_path):
    annotation_dir = tmp_path / "coco" / "annotations"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "instances_train2017.json").write_text(
        json.dumps({"images": list(range(6))}), encoding="utf-8"
    )
    config = ExperimentConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "out",
        torch_cache=tmp_path / "cache",
        train_size=3,
        subset_seed=42,
        epochs=1,
        eval_epochs=(1,),
        lr_drop_epoch=1,
    )
    context = DistributedContext(0, 1, 0, torch.device("cpu"), False)
    dataset = FakeDataset([1, 2, 3, 4, 5, 6])
    first = prepare_subset_manifest(config, context, dataset)
    second = prepare_subset_manifest(config, context, dataset)
    assert first == second
    assert first["image_ids"] == [1, 5, 6]
    assert first["image_ids"] != [1, 2, 3]
    assert first["distribution"]["object_count"] == 3
