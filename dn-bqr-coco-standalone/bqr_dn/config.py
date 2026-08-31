from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch

from .upstream import PACKAGE_ROOT, upstream_commit, upstream_source_fingerprint


Method = Literal["baseline", "bqr"]
Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path = PACKAGE_ROOT / "data" / "coco"
    output_root: Path = PACKAGE_ROOT / "artifacts"
    torch_cache: Path = PACKAGE_ROOT / ".cache" / "torch"

    method: Method = "bqr"
    seed: int = 42
    run_name: str | None = None

    epochs: int = 20
    stop_after_epoch: int | None = None
    eval_epochs: tuple[int, ...] = (5, 10, 15, 20)
    batch_size: int = 2
    accumulation_steps: int = 4
    target_global_batch_size: int = 16
    eval_batch_size: int = 2
    num_workers: int = 8
    persistent_workers: bool = False
    precision: Precision = "fp16"

    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    lr_drop_epoch: int = 16
    lr_drop_gamma: float = 0.1

    train_size: int = 10_000
    subset_seed: int = 42
    val_limit: int | None = None
    diagnostics_every: int = 100
    deterministic: bool = False

    # Official DN-DAB-Deformable-DETR R50 recipe.
    num_classes: int = 91
    num_queries: int = 300
    hidden_dim: int = 256
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    nheads: int = 8
    dropout: float = 0.0
    num_feature_levels: int = 4
    enc_n_points: int = 4
    dec_n_points: int = 4
    dn_scalar: int = 5
    dn_label_noise_scale: float = 0.2
    dn_box_noise_scale: float = 0.4

    # Minimal clean-GT BQR PoC.
    bqr_grid_size: int = 2
    bqr_alpha: float = 0.25

    def __post_init__(self) -> None:
        for name in ("data_root", "output_root", "torch_cache"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        object.__setattr__(self, "eval_epochs", tuple(sorted(set(self.eval_epochs))))

        if self.method not in ("baseline", "bqr"):
            raise ValueError(f"Unsupported method: {self.method}")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"Unsupported precision: {self.precision}")
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "target_global_batch_size": self.target_global_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "train_size": self.train_size,
            "num_queries": self.num_queries,
            "dn_scalar": self.dn_scalar,
            "bqr_grid_size": self.bqr_grid_size,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Settings must be positive: {', '.join(invalid)}")
        if self.num_workers < 0 or self.diagnostics_every < 0:
            raise ValueError("num_workers and diagnostics_every must be non-negative")
        if self.val_limit is not None and self.val_limit <= 0:
            raise ValueError("val_limit must be positive")
        if self.stop_after_epoch is not None and not (
            1 <= self.stop_after_epoch <= self.epochs
        ):
            raise ValueError("stop_after_epoch must be within the configured schedule")
        if not all(1 <= epoch <= self.epochs for epoch in self.eval_epochs):
            raise ValueError("eval_epochs must be within the configured schedule")
        if not 0.0 <= self.bqr_alpha <= 1.0:
            raise ValueError("bqr_alpha must be in [0, 1]")

    @property
    def effective_batch_size(self) -> int:
        # world size is validated at runtime; this is the expected two-GPU recipe.
        return self.batch_size * self.accumulation_steps * 2

    @property
    def detector_recipe(self) -> dict:
        return {
            "upstream_commit": upstream_commit(),
            "upstream_source_fingerprint": upstream_source_fingerprint(),
            "backbone": "resnet50",
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "enc_layers": self.enc_layers,
            "dec_layers": self.dec_layers,
            "dim_feedforward": self.dim_feedforward,
            "nheads": self.nheads,
            "dropout": self.dropout,
            "num_feature_levels": self.num_feature_levels,
            "enc_n_points": self.enc_n_points,
            "dec_n_points": self.dec_n_points,
            "dn_scalar": self.dn_scalar,
            "dn_label_noise_scale": self.dn_label_noise_scale,
            "dn_box_noise_scale": self.dn_box_noise_scale,
        }

    @property
    def comparison_recipe(self) -> dict:
        return {
            **self.detector_recipe,
            "epochs": self.epochs,
            "eval_epochs": self.eval_epochs,
            "target_global_batch_size": self.target_global_batch_size,
            "precision": self.precision,
            "lr": self.lr,
            "backbone_lr": self.backbone_lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "lr_drop_epoch": self.lr_drop_epoch,
            "lr_drop_gamma": self.lr_drop_gamma,
            "train_size": self.train_size,
            "subset_seed": self.subset_seed,
            "seed": self.seed,
        }

    @property
    def method_recipe(self) -> dict:
        recipe = {**self.comparison_recipe, "method": self.method}
        if self.method == "bqr":
            recipe.update(
                bqr_grid_size=self.bqr_grid_size,
                bqr_alpha=self.bqr_alpha,
            )
        return recipe

    @staticmethod
    def _fingerprint(value: dict) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def detector_fingerprint(self) -> str:
        return self._fingerprint(self.detector_recipe)

    @property
    def comparison_fingerprint(self) -> str:
        return self._fingerprint(self.comparison_recipe)

    @property
    def method_fingerprint(self) -> str:
        return self._fingerprint(self.method_recipe)

    @property
    def resolved_run_name(self) -> str:
        return self.run_name or f"coco{self.train_size}_e{self.epochs}"

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
    def subset_manifest_path(self) -> Path:
        return (
            self.output_root
            / "_shared"
            / "subsets"
            / f"coco_train2017_n{self.train_size}_seed{self.subset_seed}.json"
        )

    @property
    def initialization_path(self) -> Path:
        return (
            self.output_root
            / "_shared"
            / "initialization"
            / f"dn_deformable_{self.detector_fingerprint}_seed{self.seed}.pt"
        )

    def should_evaluate(self, epoch: int) -> bool:
        return epoch in self.eval_epochs

    def as_dict(self) -> dict:
        result = asdict(self)
        for name in ("data_root", "output_root", "torch_cache"):
            result[name] = str(result[name])
        result["eval_epochs"] = list(self.eval_epochs)
        result["detector_fingerprint"] = self.detector_fingerprint
        result["comparison_fingerprint"] = self.comparison_fingerprint
        result["method_fingerprint"] = self.method_fingerprint
        return result

    def official_args(self, device: torch.device) -> SimpleNamespace:
        return SimpleNamespace(
            dataset_file="coco",
            coco_path=str(self.data_root),
            coco_panoptic_path=None,
            device=str(device),
            backbone="resnet50",
            lr_backbone=self.backbone_lr,
            dilation=False,
            position_embedding="sine",
            pe_temperatureH=20,
            pe_temperatureW=20,
            masks=False,
            frozen_weights=None,
            num_feature_levels=self.num_feature_levels,
            hidden_dim=self.hidden_dim,
            nheads=self.nheads,
            enc_layers=self.enc_layers,
            dec_layers=self.dec_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            num_queries=self.num_queries,
            num_patterns=0,
            random_refpoints_xy=False,
            two_stage=False,
            dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points,
            transformer_activation="relu",
            aux_loss=True,
            use_dn=True,
            set_cost_class=2.0,
            set_cost_bbox=5.0,
            set_cost_giou=2.0,
            cls_loss_coef=1.0,
            bbox_loss_coef=5.0,
            giou_loss_coef=2.0,
            mask_loss_coef=1.0,
            dice_loss_coef=1.0,
            focal_alpha=0.25,
            fix_size=False,
            strong_aug=False,
        )
