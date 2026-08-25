from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch

from .upstream import PACKAGE_ROOT, UPSTREAM_ROOT, ensure_upstream_imports, upstream_commit


Method = Literal["baseline", "bqr_dn_v2"]
Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path = PACKAGE_ROOT / "data" / "coco"
    output_root: Path = PACKAGE_ROOT / "artifacts"
    torch_cache: Path = PACKAGE_ROOT / ".cache" / "torch"
    method: Method = "bqr_dn_v2"
    seed: int = 42
    run_name: str | None = None

    epochs: int = 12
    batch_size: int = 2
    accumulation_steps: int = 4
    target_global_batch_size: int = 16
    eval_batch_size: int = 2
    num_workers: int = 8
    precision: Precision = "fp16"
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    lr_drop_epoch: int = 11
    lr_drop_gamma: float = 0.1
    diagnostics_every: int = 100
    train_limit: int | None = None
    val_limit: int | None = None
    deterministic: bool = False

    # Official DINO R50 4-scale 12-epoch recipe.
    num_classes: int = 91
    num_queries: int = 900
    hidden_dim: int = 256
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    nheads: int = 8
    num_feature_levels: int = 4
    enc_n_points: int = 4
    dec_n_points: int = 4
    dn_number: int = 100
    dn_box_noise_scale: float = 0.4
    dn_label_noise_ratio: float = 0.5

    # BQR-DN V2. These do not alter any official DINO loss weight.
    bqr_points_per_level: int = 4
    bqr_gate_bias: float = -2.0

    def __post_init__(self) -> None:
        for name in ("data_root", "output_root", "torch_cache"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.method not in ("baseline", "bqr_dn_v2"):
            raise ValueError(f"Unsupported method: {self.method}")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"Unsupported precision: {self.precision}")
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "target_global_batch_size": self.target_global_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "lr": self.lr,
            "backbone_lr": self.backbone_lr,
            "num_queries": self.num_queries,
            "dn_number": self.dn_number,
            "bqr_points_per_level": self.bqr_points_per_level,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Settings must be positive: {', '.join(invalid)}")
        if self.num_workers < 0 or self.diagnostics_every < 0:
            raise ValueError("num_workers and diagnostics_every must be non-negative")
        for name in ("train_limit", "val_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")

    @property
    def detector_recipe(self) -> dict:
        return {
            "upstream_commit": upstream_commit(),
            "backbone": "resnet50",
            "num_classes": self.num_classes,
            "feature_levels": self.num_feature_levels,
            "queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "encoder_layers": self.enc_layers,
            "decoder_layers": self.dec_layers,
            "feedforward_dim": self.dim_feedforward,
            "heads": self.nheads,
            "enc_points": self.enc_n_points,
            "dec_points": self.dec_n_points,
            "dn_number": self.dn_number,
            "dn_box_noise": self.dn_box_noise_scale,
            "dn_label_noise": self.dn_label_noise_ratio,
        }

    @property
    def comparison_recipe(self) -> dict:
        return {
            **self.detector_recipe,
            "epochs": self.epochs,
            "batch_size_per_gpu": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "target_global_batch_size": self.target_global_batch_size,
            "precision": self.precision,
            "lr": self.lr,
            "backbone_lr": self.backbone_lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "lr_drop_epoch": self.lr_drop_epoch,
            "seed": self.seed,
        }

    @staticmethod
    def _fingerprint(payload: dict, length: int = 12) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:length]

    @property
    def detector_fingerprint(self) -> str:
        return self._fingerprint(self.detector_recipe)

    @property
    def comparison_fingerprint(self) -> str:
        return self._fingerprint(self.comparison_recipe)

    @property
    def resolved_run_name(self) -> str:
        return self.run_name or (
            f"coco_gb{self.target_global_batch_size}_{self.precision}_"
            f"b{self.batch_size}a{self.accumulation_steps}_{self.comparison_fingerprint[:8]}"
        )

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.method / f"seed_{self.seed}" / self.resolved_run_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def final_checkpoint(self) -> Path:
        return self.checkpoint_dir / "final.pt"

    @property
    def history_path(self) -> Path:
        return self.run_dir / "history.csv"

    @property
    def initialization_path(self) -> Path:
        return (
            self.output_root
            / "initializations"
            / f"dino_r50_4scale_seed{self.seed}_{self.detector_fingerprint}.pt"
        )

    def validate_world_size(self, world_size: int) -> None:
        effective = self.batch_size * world_size * self.accumulation_steps
        if effective != self.target_global_batch_size:
            raise ValueError(
                "Effective global batch mismatch: "
                f"{self.batch_size} x {world_size} x {self.accumulation_steps} "
                f"= {effective}, expected {self.target_global_batch_size}"
            )

    def official_args(self, device: torch.device) -> SimpleNamespace:
        ensure_upstream_imports()
        from util.slconfig import SLConfig

        official = SLConfig.fromfile(str(UPSTREAM_ROOT / "config" / "DINO" / "DINO_4scale.py"))
        args = SimpleNamespace(**dict(official._cfg_dict))
        overrides = {
            "device": str(device),
            "dataset_file": "coco",
            "coco_path": str(self.data_root),
            "num_classes": self.num_classes,
            "dn_labelbook_size": self.num_classes,
            "backbone": "resnet50",
            "return_interm_indices": [1, 2, 3],
            "num_queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "enc_layers": self.enc_layers,
            "dec_layers": self.dec_layers,
            "dim_feedforward": self.dim_feedforward,
            "nheads": self.nheads,
            "num_feature_levels": self.num_feature_levels,
            "enc_n_points": self.enc_n_points,
            "dec_n_points": self.dec_n_points,
            "dn_number": self.dn_number,
            "dn_box_noise_scale": self.dn_box_noise_scale,
            "dn_label_noise_ratio": self.dn_label_noise_ratio,
            "lr": self.lr,
            "lr_backbone": self.backbone_lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "lr_drop": self.lr_drop_epoch,
            "batch_size": self.batch_size,
            "masks": False,
            "frozen_weights": None,
            "cache_mode": False,
        }
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def as_dict(self) -> dict:
        payload = asdict(self)
        for name in ("data_root", "output_root", "torch_cache"):
            payload[name] = str(payload[name])
        payload["resolved_run_name"] = self.resolved_run_name
        payload["comparison_fingerprint"] = self.comparison_fingerprint
        payload["detector_fingerprint"] = self.detector_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: dict, **overrides) -> "ExperimentConfig":
        valid = {field.name for field in fields(cls)}
        values = {key: value for key, value in payload.items() if key in valid}
        values.update(overrides)
        return cls(**values)
