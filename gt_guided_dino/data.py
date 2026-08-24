from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .config import ExperimentConfig, VOC_CLASSES
from .upstream import ensure_upstream_imports


ensure_upstream_imports()
from datasets import transforms as T  # noqa: E402
from util.misc import collate_fn  # noqa: E402


CLASS_TO_INDEX = {name: index for index, name in enumerate(VOC_CLASSES)}


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"VOC split file not found: {path}")
    return [line.strip().split()[0] for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def prepare_train_manifest(config: ExperimentConfig) -> list[str]:
    all_ids = _read_split(config.data_root / "ImageSets" / "Main" / "train.txt")
    if config.train_limit >= len(all_ids):
        selected = all_ids
    else:
        selected = sorted(random.Random(config.seed).sample(all_ids, config.train_limit))
    path = config.split_manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(selected) + "\n"
    if path.is_file():
        existing = path.read_text(encoding="ascii")
        if existing != content:
            raise ValueError(f"Existing split manifest does not match the configured seed/limit: {path}")
    else:
        path.write_text(content, encoding="ascii")
    return selected


def build_transforms(config: ExperimentConfig, train: bool):
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if not train:
        return T.Compose([T.RandomResize([config.val_size], max_size=config.val_max_size), normalize])

    resize = T.RandomResize(list(config.train_scales), max_size=config.train_max_size)
    if config.use_random_crop:
        resize_or_crop = T.RandomSelect(
            resize,
            T.Compose(
                [
                    T.RandomResize(list(config.crop_resize_scales)),
                    T.RandomSizeCrop(config.crop_min_size, config.crop_max_size),
                    resize,
                ]
            ),
        )
    else:
        resize_or_crop = resize
    return T.Compose([T.RandomHorizontalFlip(), resize_or_crop, normalize])


class VOCDetectionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_ids: list[str],
        *,
        transforms,
        exclude_difficult: bool,
    ) -> None:
        self.root = Path(root)
        self.image_ids = list(image_ids)
        self.transforms = transforms
        self.exclude_difficult = exclude_difficult
        self._annotation_cache: dict[str, dict] = {}
        for directory in ("JPEGImages", "Annotations"):
            if not (self.root / directory).is_dir():
                raise FileNotFoundError(f"Missing VOC directory: {self.root / directory}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _parse_annotation(self, image_id: str) -> dict:
        cached = self._annotation_cache.get(image_id)
        if cached is not None:
            return cached
        path = self.root / "Annotations" / f"{image_id}.xml"
        root = ET.parse(path).getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"Annotation has no size node: {path}")
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        objects = []
        for node in root.findall("object"):
            name = node.findtext("name", "").strip().lower()
            if name not in CLASS_TO_INDEX:
                raise ValueError(f"Unknown VOC class {name!r} in {path}")
            box = node.find("bndbox")
            if box is None:
                continue
            # VOC coordinates are 1-based and inclusive. Convert to continuous
            # zero-based xyxy coordinates while preserving inclusive extent.
            xmin = float(box.findtext("xmin", "1")) - 1.0
            ymin = float(box.findtext("ymin", "1")) - 1.0
            xmax = float(box.findtext("xmax", "1"))
            ymax = float(box.findtext("ymax", "1"))
            xmin = min(max(xmin, 0.0), float(width))
            ymin = min(max(ymin, 0.0), float(height))
            xmax = min(max(xmax, xmin), float(width))
            ymax = min(max(ymax, ymin), float(height))
            objects.append(
                {
                    "label": CLASS_TO_INDEX[name],
                    "box": [xmin, ymin, xmax, ymax],
                    "difficult": int(node.findtext("difficult", "0")),
                }
            )
        parsed = {"width": width, "height": height, "objects": objects}
        self._annotation_cache[image_id] = parsed
        return parsed

    def get_ground_truth(self, image_id: str | int) -> dict:
        return self._parse_annotation(f"{int(image_id):06d}")

    def class_counts(self) -> dict[str, int]:
        counts: Counter[int] = Counter()
        for image_id in self.image_ids:
            for obj in self._parse_annotation(image_id)["objects"]:
                if not obj["difficult"]:
                    counts[obj["label"]] += 1
        return {name: counts[index] for index, name in enumerate(VOC_CLASSES)}

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        annotation = self._parse_annotation(image_id)
        image_path = self.root / "JPEGImages" / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        selected = [
            obj
            for obj in annotation["objects"]
            if not (self.exclude_difficult and obj["difficult"])
        ]
        boxes = torch.as_tensor([obj["box"] for obj in selected], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor([obj["label"] for obj in selected], dtype=torch.int64)
        difficult = torch.as_tensor([obj["difficult"] for obj in selected], dtype=torch.bool)
        area = (
            (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
            * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": torch.zeros(len(selected), dtype=torch.int64),
            "image_id": torch.tensor(int(image_id), dtype=torch.int64),
            "orig_size": torch.tensor([annotation["height"], annotation["width"]], dtype=torch.int64),
            "size": torch.tensor([annotation["height"], annotation["width"]], dtype=torch.int64),
        }
        if not self.exclude_difficult:
            target["difficult"] = difficult
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def build_datasets(config: ExperimentConfig) -> tuple[VOCDetectionDataset, VOCDetectionDataset]:
    train_ids = prepare_train_manifest(config)
    val_ids = _read_split(config.data_root / "ImageSets" / "Main" / "val.txt")
    if config.val_limit is not None:
        val_ids = val_ids[: config.val_limit]
    train_dataset = VOCDetectionDataset(
        config.data_root,
        train_ids,
        transforms=build_transforms(config, train=True),
        exclude_difficult=True,
    )
    val_dataset = VOCDetectionDataset(
        config.data_root,
        val_ids,
        transforms=build_transforms(config, train=False),
        exclude_difficult=False,
    )
    return train_dataset, val_dataset


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def build_loaders(config: ExperimentConfig):
    train_dataset, val_dataset = build_datasets(config)
    generator = torch.Generator().manual_seed(config.seed)
    loader_options = {
        "batch_size": config.batch_size,
        "collate_fn": collate_fn,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset, generator=generator),
        drop_last=False,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        drop_last=False,
        **loader_options,
    )
    return train_loader, val_loader


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [
        {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]
