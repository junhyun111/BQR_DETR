from __future__ import annotations

import os
import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from .bqr_v2 import BQRDecoderBridge
from .config import ExperimentConfig
from .distributed import DistributedContext, seed_everything
from .upstream import ensure_upstream_imports, upstream_commit


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()[:16]


def build_official_components(
    config: ExperimentConfig,
    device: torch.device,
    *,
    pretrained_backbone: bool,
):
    torch.hub.set_dir(str(config.torch_cache))
    ensure_upstream_imports()
    import models  # noqa: F401
    import models.dino.backbone as backbone_module
    from models.registry import MODULE_BUILD_FUNCS

    original = backbone_module.is_main_process
    backbone_module.is_main_process = lambda: pretrained_backbone
    try:
        builder = MODULE_BUILD_FUNCS.get("dino")
        if builder is None:
            raise RuntimeError("Official DINO builder was not registered")
        detector, criterion, postprocessors = builder(config.official_args(device))
    finally:
        backbone_module.is_main_process = original
    return detector.to(device), criterion.to(device), postprocessors


def ensure_common_initialization(
    config: ExperimentConfig, context: DistributedContext
) -> dict:
    path = config.initialization_path
    if context.is_main and not path.is_file():
        # Detector initialization is method-independent and must not inherit a
        # rank-offset seed. V2 parameters are initialized separately later.
        seed_everything(config.seed, rank=0, deterministic=config.deterministic)
        detector, _, _ = build_official_components(
            config, context.device, pretrained_backbone=True
        )
        detector_state = _cpu_state_dict(detector)
        _atomic_save(
            {
                "format_version": 1,
                "upstream_commit": upstream_commit(),
                "seed": config.seed,
                "detector_recipe": config.detector_recipe,
                "detector_fingerprint": config.detector_fingerprint,
                "initialization_fingerprint": _state_dict_fingerprint(detector_state),
                "model_state_dict": detector_state,
            },
            path,
        )
        del detector
        if context.device.type == "cuda":
            torch.cuda.empty_cache()
    context.barrier()
    if not path.is_file():
        raise FileNotFoundError(f"Common detector initialization was not created: {path}")
    initialization = torch.load(path, map_location="cpu", weights_only=False)
    if initialization.get("upstream_commit") != upstream_commit():
        raise ValueError("Initialization uses a different official DINO commit")
    if initialization.get("detector_fingerprint") != config.detector_fingerprint:
        raise ValueError("Initialization detector recipe does not match this run")
    if not initialization.get("initialization_fingerprint"):
        raise ValueError("Initialization is missing its weight fingerprint")
    if int(initialization.get("seed", -1)) != config.seed:
        raise ValueError("Initialization seed does not match this run")
    return initialization


class ResearchModel(nn.Module):
    def __init__(self, detector: nn.Module, config: ExperimentConfig) -> None:
        super().__init__()
        self.detector = detector
        self.method = config.method
        # The bridge is registered exactly once under detector.transformer.
        # Keep this convenience reference outside nn.Module registration so
        # checkpoints do not duplicate the complete original decoder.
        object.__setattr__(self, "bqr_bridge", None)
        self._diagnostics_enabled = False
        if config.method == "bqr_dn_v2":
            cuda_devices = (
                [torch.cuda.current_device()] if torch.cuda.is_available() else []
            )
            # Adding V2 must not advance the detector/data random stream.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(config.seed + 2_000_003)
                bridge = BQRDecoderBridge(
                    detector.transformer.decoder,
                    num_queries=config.num_queries,
                    hidden_dim=config.hidden_dim,
                    num_levels=config.num_feature_levels,
                    points_per_level=config.bqr_points_per_level,
                    gate_bias=config.bqr_gate_bias,
                )
                detector.transformer.decoder = bridge
                object.__setattr__(self, "bqr_bridge", bridge)

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self._diagnostics_enabled = bool(enabled)

    def forward(self, samples, targets: list[dict] | None = None):
        if self.bqr_bridge is not None:
            self.bqr_bridge.set_context(
                targets if self.training else None,
                diagnostics_enabled=self._diagnostics_enabled,
            )
        try:
            return self.detector(samples, targets)
        finally:
            if self.bqr_bridge is not None:
                self.bqr_bridge.clear_context()

    def training_diagnostics(self) -> dict[str, torch.Tensor]:
        if self.bqr_bridge is None:
            return {}
        return dict(self.bqr_bridge.latest_diagnostics)


def build_experiment(
    config: ExperimentConfig,
    context: DistributedContext,
    *,
    use_common_initialization: bool = True,
):
    initialization = (
        ensure_common_initialization(config, context)
        if use_common_initialization
        else None
    )
    seed_everything(config.seed, context.rank, config.deterministic)
    detector, criterion, postprocessors = build_official_components(
        config, context.device, pretrained_backbone=False
    )
    if initialization is not None:
        detector.load_state_dict(initialization["model_state_dict"], strict=True)
    model = ResearchModel(detector, config).to(context.device)
    model.initialization_fingerprint = (
        initialization.get("initialization_fingerprint")
        if initialization is not None
        else None
    )
    return model, criterion, postprocessors
