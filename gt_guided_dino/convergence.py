from __future__ import annotations

import json
from pathlib import Path

import torch

from .build import build_experiment
from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .data import build_loaders
from .evaluation import evaluate_model


@torch.no_grad()
def evaluate_checkpoint_series(
    run_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    val_limit: int = 250,
    precision: str = "fp16",
    force: bool = False,
) -> list[dict]:
    """Evaluate every epoch checkpoint on one deterministic validation subset."""
    directory = Path(run_dir).resolve()
    checkpoint_paths = sorted((directory / "checkpoints").glob("epoch_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No epoch checkpoints found in {directory / 'checkpoints'}")
    cache_path = directory / f"convergence_val{val_limit}.json"
    checkpoint_names = [path.name for path in checkpoint_paths]
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("checkpoints") == checkpoint_names:
            return list(cached["epochs"])

    first = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    overrides = {"val_limit": val_limit}
    if data_root is not None:
        overrides["data_root"] = Path(data_root)
    config = ExperimentConfig.from_dict(first["config"], **overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = build_loaders(config)
    model, _, postprocessors = build_experiment(
        config,
        device,
        use_common_initialization=False,
    )
    rows = []
    try:
        for checkpoint_path in checkpoint_paths:
            start_epoch, _ = load_checkpoint(checkpoint_path, config, model)
            result = evaluate_model(
                model,
                postprocessors,
                val_loader,
                val_loader.dataset,
                device,
                precision=precision,
            )
            rows.append(
                {
                    "epoch": start_epoch,
                    "voc07_map50": result["metrics"]["voc07_map50"],
                    "voc_map50_integral": result["metrics"]["voc_map50_integral"],
                    "seconds": result["metrics"]["seconds"],
                }
            )
    finally:
        model.close()
    cache_path.write_text(
        json.dumps(
            {
                "method": config.method,
                "validation_images": val_limit,
                "checkpoints": checkpoint_names,
                "epochs": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows
