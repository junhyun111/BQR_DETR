from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    Subset,
)

from .config import ExperimentConfig
from .distributed import DistributedContext
from .upstream import ensure_upstream_imports


ensure_upstream_imports()
from datasets import build_dataset, get_coco_api_from_dataset  # noqa: E402
from util.misc import collate_fn  # noqa: E402


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without the duplicate padding of DistributedSampler."""

    def __init__(self, dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size


@dataclass
class DataBundle:
    train_loader: DataLoader | None
    val_loader: DataLoader
    train_sampler: Sampler | None
    coco_api: object
    subset_manifest: dict | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _distribution_summary(coco, image_ids: list[int]) -> dict:
    category_counts: dict[str, int] = {}
    size_counts = {"small": 0, "medium": 0, "large": 0}
    objects = 0
    for image_id in image_ids:
        for annotation in coco.imgToAnns.get(image_id, []):
            if annotation.get("iscrowd", 0):
                continue
            objects += 1
            category = str(annotation["category_id"])
            category_counts[category] = category_counts.get(category, 0) + 1
            area = float(annotation.get("area", 0.0))
            if area < 32.0**2:
                size_counts["small"] += 1
            elif area < 96.0**2:
                size_counts["medium"] += 1
            else:
                size_counts["large"] += 1
    return {
        "object_count": objects,
        "category_counts": category_counts,
        "size_counts": size_counts,
    }


def _validate_manifest(
    manifest: dict,
    *,
    annotation_sha256: str,
    train_size: int,
    subset_seed: int,
) -> None:
    expected = {
        "annotation_sha256": annotation_sha256,
        "train_size": train_size,
        "subset_seed": subset_seed,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: manifest={old!r}, expected={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(f"COCO subset manifest mismatch: {details}")
    image_ids = manifest.get("image_ids")
    if not isinstance(image_ids, list) or len(image_ids) != train_size:
        raise ValueError("COCO subset manifest contains an invalid image ID list")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("COCO subset manifest contains duplicate image IDs")


def prepare_subset_manifest(
    config: ExperimentConfig,
    context: DistributedContext,
    train_dataset,
) -> dict:
    annotation_path = config.data_root / "annotations" / "instances_train2017.json"
    annotation_sha256 = _sha256(annotation_path)
    path = config.subset_manifest_path
    if context.is_main and not path.is_file():
        available_ids = sorted(int(image_id) for image_id in train_dataset.ids)
        if config.train_size > len(available_ids):
            raise ValueError(
                f"Requested {config.train_size} training images, but COCO has {len(available_ids)}"
            )
        sampled = random.Random(config.subset_seed).sample(
            available_ids, config.train_size
        )
        image_ids = sorted(sampled)
        manifest = {
            "format_version": 1,
            "dataset": "coco_train2017",
            "annotation_path": "annotations/instances_train2017.json",
            "annotation_sha256": annotation_sha256,
            "train_size": config.train_size,
            "subset_seed": config.subset_seed,
            "sampling": "random.sample(sorted(image_ids), n)",
            "image_ids": image_ids,
            "distribution": _distribution_summary(train_dataset.coco, image_ids),
        }
        _atomic_json(path, manifest)
    context.barrier()
    if not path.is_file():
        raise FileNotFoundError(f"Subset manifest was not created: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(
        manifest,
        annotation_sha256=annotation_sha256,
        train_size=config.train_size,
        subset_seed=config.subset_seed,
    )
    dataset_ids = {int(image_id): index for index, image_id in enumerate(train_dataset.ids)}
    missing = [image_id for image_id in manifest["image_ids"] if image_id not in dataset_ids]
    if missing:
        raise ValueError(f"Subset manifest refers to missing COCO image IDs: {missing[:5]}")
    return manifest


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_common(
    config: ExperimentConfig, context: DistributedContext
) -> dict:
    common = {
        "num_workers": config.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    if config.num_workers > 0:
        common["prefetch_factor"] = 2
    return common


def _validate_coco_root(config: ExperimentConfig) -> None:
    required = (
        config.data_root / "train2017",
        config.data_root / "val2017",
        config.data_root / "annotations" / "instances_train2017.json",
        config.data_root / "annotations" / "instances_val2017.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing COCO paths:\n" + "\n".join(missing))


def build_data(
    config: ExperimentConfig,
    context: DistributedContext,
    *,
    training: bool = True,
) -> DataBundle:
    _validate_coco_root(config)
    args = config.official_args(context.device)
    val_dataset = build_dataset("val", args)
    if config.val_limit is not None:
        val_dataset = Subset(val_dataset, range(min(config.val_limit, len(val_dataset))))

    manifest = None
    train_dataset = None
    if training:
        full_train_dataset = build_dataset("train", args)
        manifest = prepare_subset_manifest(config, context, full_train_dataset)
        id_to_index = {
            int(image_id): index for index, image_id in enumerate(full_train_dataset.ids)
        }
        indices = [id_to_index[image_id] for image_id in manifest["image_ids"]]
        train_dataset = Subset(full_train_dataset, indices)

    if context.distributed and train_dataset is not None:
        train_sampler: Sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
    elif train_dataset is not None:
        generator = torch.Generator().manual_seed(config.seed)
        train_sampler = RandomSampler(train_dataset, generator=generator)
    else:
        train_sampler = None

    if context.distributed:
        val_sampler: Sampler = DistributedEvalSampler(
            val_dataset, context.rank, context.world_size
        )
    else:
        val_sampler = SequentialSampler(val_dataset)

    common = _loader_common(config, context)
    train_loader = None
    if train_dataset is not None and train_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=BatchSampler(
                train_sampler, batch_size=config.batch_size, drop_last=True
            ),
            **common,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        coco_api=get_coco_api_from_dataset(val_dataset),
        subset_manifest=manifest,
    )


def _move_value(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    return value


def move_batch(samples, targets: list[dict], device: torch.device):
    return samples.to(device), [_move_value(target, device) for target in targets]
