from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Sampler, SequentialSampler, Subset

from .config import ExperimentConfig
from .distributed import DistributedContext
from .upstream import ensure_upstream_imports


ensure_upstream_imports()
from datasets import build_dataset, get_coco_api_from_dataset  # noqa: E402
from util.misc import collate_fn  # noqa: E402


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding duplicate images."""

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


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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

    train_dataset = build_dataset("train", args) if training else None
    if train_dataset is not None and config.train_limit is not None:
        train_dataset = Subset(train_dataset, range(min(config.train_limit, len(train_dataset))))

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
        train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    else:
        train_sampler = None

    if context.distributed:
        val_sampler: Sampler = DistributedEvalSampler(
            val_dataset, context.rank, context.world_size
        )
    else:
        val_sampler = SequentialSampler(val_dataset)

    common = {
        "num_workers": config.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    if config.num_workers > 0:
        common["prefetch_factor"] = 2
    train_loader = None
    if train_dataset is not None and train_sampler is not None:
        train_batch_sampler = BatchSampler(
            train_sampler, batch_size=config.batch_size, drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
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
    )


def move_batch(samples, targets: list[dict], device: torch.device):
    samples = samples.to(device, non_blocking=True)
    moved = [
        {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]
    return samples, moved
