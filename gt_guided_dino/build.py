from __future__ import annotations

import os
from pathlib import Path

import torch

from .config import ExperimentConfig, seed_everything
from .criterion import ResearchCriterion
from .model import make_research_model
from .upstream import ensure_upstream_imports, upstream_commit


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
        build_function = MODULE_BUILD_FUNCS.get("dino")
        if build_function is None:
            raise RuntimeError("Official DINO builder was not registered")
        detector, official_criterion, postprocessors = build_function(
            config.official_args(device)
        )
    finally:
        backbone_module.is_main_process = original
    detector.to(device)
    official_criterion.to(device)
    return detector, official_criterion, postprocessors


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def ensure_common_initialization(config: ExperimentConfig, device: torch.device) -> dict:
    path = config.initialization_path
    if not path.is_file():
        seed_everything(config.seed)
        detector, _, _ = build_official_components(
            config, device, pretrained_backbone=True
        )
        _atomic_torch_save(
            {
                "format_version": 1,
                "upstream_commit": upstream_commit(),
                "seed": config.seed,
                "model_recipe": config.model_recipe,
                "model_state_dict": _cpu_state_dict(detector),
            },
            path,
        )
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()
    initialization = torch.load(path, map_location="cpu", weights_only=False)
    if initialization.get("upstream_commit") != upstream_commit():
        raise ValueError("Initialization was made from a different DINO commit")
    if initialization.get("model_recipe") != config.model_recipe:
        raise ValueError("Initialization model recipe does not match this run")
    if int(initialization.get("seed", -1)) != config.seed:
        raise ValueError("Initialization seed does not match this run")
    return initialization


def build_experiment(
    config: ExperimentConfig,
    device: torch.device,
    *,
    use_common_initialization: bool = True,
):
    initialization = None
    if use_common_initialization:
        initialization = ensure_common_initialization(config, device)
    seed_everything(config.seed)
    detector, official_criterion, postprocessors = build_official_components(
        config, device, pretrained_backbone=False
    )
    if initialization is not None:
        detector.load_state_dict(initialization["model_state_dict"], strict=True)
    model = make_research_model(detector, config).to(device)
    criterion = ResearchCriterion(official_criterion, config).to(device)
    return model, criterion, postprocessors
