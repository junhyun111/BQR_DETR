from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch

from .upstream import PROJECT_ROOT, UPSTREAM_ROOT, ensure_upstream_imports


Method = Literal[
    "baseline", "gt_guided_aux", "bqr_dn_v2", "bqr_dn_v2_1", "bqr_dn_v3"
]
Precision = Literal["fp32", "fp16", "bf16"]
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path = PROJECT_ROOT / "VOC2007"
    output_root: Path = PROJECT_ROOT / "artifacts"
    torch_cache: Path = PROJECT_ROOT / "weights" / "torch"
    method: Method = "baseline"
    seed: int = 42
    train_limit: int = 1000
    val_limit: int | None = None
    epochs: int = 12
    batch_size: int = 1
    accumulation_steps: int = 4
    num_workers: int = 0
    precision: Precision = "fp16"
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    lr_drop_epoch: int = 11
    lr_drop_gamma: float = 0.1
    save_every: int = 1

    # Official DINO R50 4-scale defaults.
    num_queries: int = 900
    hidden_dim: int = 256
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    num_feature_levels: int = 4
    dn_number: int = 100

    # Official DINO augmentation defaults.
    train_scales: tuple[int, ...] = (
        480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800,
    )
    train_max_size: int = 1333
    crop_resize_scales: tuple[int, ...] = (400, 500, 600)
    crop_min_size: int = 384
    crop_max_size: int = 600
    use_random_crop: bool = True
    val_size: int = 800
    val_max_size: int = 1333

    # Proposed path.
    aux_weight: float = 1.0
    aux_l1_coef: float = 5.0
    aux_giou_coef: float = 2.0
    sampling_points_per_level: int = 4

    # BQR-DN V2 replaces only the content of DINO's training-time DN queries.
    bqr_enabled: bool = True
    bqr_dn_weight: float = 1.0
    # V2/V2.1 resolve this to 4, while V3 resolves it to 5.
    bqr_points_per_level: int | None = None
    bqr_gate_bias: float = -2.0

    # BQR-DN V2.1 adds sampled-feature-aware attention to the V2 prior.
    bqr_content_attention: bool = True
    bqr_attention_dim: int = 64
    bqr_content_scale_init: float = 0.05
    bqr_attention_temperature: float = 1.0

    # BQR-DN V3 adds a parameter-free scale prior to five-point sampling.
    bqr_scale_aware: bool = True
    bqr_target_cells: float = 4.0
    bqr_scale_sigma: float = 0.8
    bqr_scale_weight: float = 1.0
    bqr_scale_logit_floor: float = -4.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).resolve())
        object.__setattr__(self, "output_root", Path(self.output_root).resolve())
        object.__setattr__(self, "torch_cache", Path(self.torch_cache).resolve())
        if self.bqr_points_per_level is None:
            object.__setattr__(
                self,
                "bqr_points_per_level",
                5 if self.method == "bqr_dn_v3" else 4,
            )
        if self.method not in (
            "baseline", "gt_guided_aux", "bqr_dn_v2", "bqr_dn_v2_1", "bqr_dn_v3"
        ):
            raise ValueError(f"Unknown method: {self.method}")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"Unknown precision: {self.precision}")
        positive = {
            "train_limit": self.train_limit,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "lr": self.lr,
            "backbone_lr": self.backbone_lr,
            "num_queries": self.num_queries,
            "sampling_points_per_level": self.sampling_points_per_level,
            "bqr_points_per_level": self.bqr_points_per_level,
            "bqr_dn_weight": self.bqr_dn_weight,
            "bqr_attention_dim": self.bqr_attention_dim,
            "bqr_attention_temperature": self.bqr_attention_temperature,
            "bqr_target_cells": self.bqr_target_cells,
            "bqr_scale_sigma": self.bqr_scale_sigma,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.val_limit is not None and self.val_limit <= 0:
            raise ValueError("val_limit must be positive when set")
        if not 0.0 < self.bqr_content_scale_init < 1.0:
            raise ValueError("bqr_content_scale_init must be strictly between 0 and 1")
        if self.bqr_scale_weight < 0:
            raise ValueError("bqr_scale_weight must be non-negative")
        if self.bqr_scale_logit_floor > 0:
            raise ValueError("bqr_scale_logit_floor must be non-positive")
        if self.method == "bqr_dn_v3":
            if self.num_feature_levels != 4:
                raise ValueError("BQR-DN V3 requires num_feature_levels=4")
            if self.bqr_points_per_level != 5:
                raise ValueError("BQR-DN V3 requires bqr_points_per_level=5")
        if self.lr_drop_epoch >= self.epochs:
            raise ValueError("lr_drop_epoch must be earlier than the final epoch")

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.method / f"seed_{self.seed}"

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def history_path(self) -> Path:
        return self.run_dir / "history.csv"

    @property
    def split_manifest(self) -> Path:
        return self.output_root / "splits" / f"voc2007_train_seed{self.seed}_n{self.train_limit}.txt"

    @property
    def model_recipe(self) -> dict:
        return {
            "backbone": "resnet50",
            "feature_levels": self.num_feature_levels,
            "queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "encoder_layers": self.enc_layers,
            "decoder_layers": self.dec_layers,
            "feedforward_dim": self.dim_feedforward,
            "dn_number": self.dn_number,
            "num_classes": len(VOC_CLASSES),
        }

    @property
    def model_recipe_hash(self) -> str:
        payload = json.dumps(self.model_recipe, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    @property
    def initialization_path(self) -> Path:
        return self.output_root / "initializations" / f"dino_r50_{self.model_recipe_hash}_seed{self.seed}.pt"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["data_root"] = str(self.data_root)
        result["output_root"] = str(self.output_root)
        result["torch_cache"] = str(self.torch_cache)
        return result

    @classmethod
    def from_dict(cls, values: dict, **overrides) -> "ExperimentConfig":
        valid = {field.name for field in fields(cls)}
        payload = {key: value for key, value in values.items() if key in valid}
        payload.update(overrides)
        return cls(**payload)

    def official_args(self, device: torch.device) -> SimpleNamespace:
        ensure_upstream_imports()
        from util.slconfig import SLConfig

        official = SLConfig.fromfile(str(UPSTREAM_ROOT / "config" / "DINO" / "DINO_4scale.py"))
        args = SimpleNamespace(**dict(official._cfg_dict))
        overrides = {
            "device": str(device),
            "dataset_file": "voc",
            "num_classes": len(VOC_CLASSES),
            "dn_labelbook_size": len(VOC_CLASSES),
            "backbone": "resnet50",
            "return_interm_indices": [1, 2, 3],
            "num_queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "enc_layers": self.enc_layers,
            "dec_layers": self.dec_layers,
            "dim_feedforward": self.dim_feedforward,
            "num_feature_levels": self.num_feature_levels,
            "dn_number": self.dn_number,
            "lr": self.lr,
            "lr_backbone": self.backbone_lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "lr_drop": self.lr_drop_epoch,
            "batch_size": self.batch_size,
            "masks": False,
            "frozen_weights": None,
        }
        for name, value in overrides.items():
            setattr(args, name, value)
        return args


def smoke_config(method: Method = "gt_guided_aux", **overrides) -> ExperimentConfig:
    config = ExperimentConfig(
        method=method,
        train_limit=4,
        val_limit=4,
        epochs=2,
        lr_drop_epoch=1,
        accumulation_steps=1,
        precision="fp32",
        num_queries=20,
        # Official DINO's sine embedding is fixed to the 256-d model width.
        hidden_dim=256,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=512,
        dn_number=4,
        train_scales=(64,),
        train_max_size=96,
        crop_resize_scales=(64,),
        crop_min_size=48,
        crop_max_size=64,
        use_random_crop=False,
        val_size=64,
        val_max_size=96,
    )
    return replace(config, **overrides)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
