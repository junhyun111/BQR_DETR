from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .distributed import DistributedContext, all_gather_objects, unwrap_model
from .logging_utils import write_history
from .upstream import upstream_commit


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _local_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def _resume_compatible_comparison_recipe(config_dict: dict) -> dict:
    """
    Return only the training settings that must remain identical when
    resuming a run.

    batch_size and accumulation_steps are intentionally excluded.

    This allows execution topology changes such as:

        old:
            2 GPUs x batch 4 x accumulation 2 = global batch 16

        new:
            1 GPU  x batch 4 x accumulation 4 = global batch 16

    while still requiring the actual optimization recipe to remain the same.
    """

    detector_keys = (
        "num_classes",
        "num_queries",
        "hidden_dim",
        "enc_layers",
        "dec_layers",
        "dim_feedforward",
        "nheads",
        "num_feature_levels",
        "enc_n_points",
        "dec_n_points",
        "dn_number",
        "dn_box_noise_scale",
        "dn_label_noise_ratio",
    )

    training_keys = (
        "epochs",
        "target_global_batch_size",
        "precision",
        "lr",
        "backbone_lr",
        "weight_decay",
        "grad_clip",
        "lr_drop_epoch",
        "seed",
    )

    keys = detector_keys + training_keys

    return {
        key: config_dict.get(key)
        for key in keys
    }


def _validate_resume_compatibility(
    checkpoint: dict,
    config: ExperimentConfig,
) -> None:
    """
    Validate whether a checkpoint can safely resume under the current
    execution configuration.

    Exact comparison_fingerprint match is preferred.

    If the fingerprint differs, allow the resume only when the difference
    comes from execution-topology-related settings such as per-GPU batch size
    or gradient accumulation, while the effective global batch and the rest
    of the optimization/model recipe remain unchanged.
    """

    checkpoint_fingerprint = checkpoint.get("comparison_fingerprint")
    current_fingerprint = config.comparison_fingerprint

    if checkpoint_fingerprint == current_fingerprint:
        return

    checkpoint_config = checkpoint.get("config")

    if not isinstance(checkpoint_config, dict):
        raise ValueError(
            "Checkpoint comparison recipe mismatch and checkpoint does not "
            "contain a recoverable config for compatibility validation"
        )

    checkpoint_recipe = _resume_compatible_comparison_recipe(
        checkpoint_config
    )

    current_recipe = _resume_compatible_comparison_recipe(
        config.as_dict()
    )

    mismatches = {}

    for key in checkpoint_recipe:
        old_value = checkpoint_recipe[key]
        new_value = current_recipe[key]

        if old_value != new_value:
            mismatches[key] = {
                "checkpoint": old_value,
                "current": new_value,
            }

    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={values['checkpoint']!r}, "
            f"current={values['current']!r}"
            for key, values in mismatches.items()
        )

        raise ValueError(
            "Checkpoint comparison recipe mismatch. "
            "Resume compatibility check also failed: "
            f"{details}"
        )

    old_batch = checkpoint_config.get("batch_size")
    old_accumulation = checkpoint_config.get("accumulation_steps")
    old_target_global_batch = checkpoint_config.get(
        "target_global_batch_size"
    )

    print(
        "[resume] comparison fingerprint changed, but the checkpoint is "
        "optimization-compatible."
    )

    print(
        "[resume] execution topology change accepted: "
        f"batch_per_gpu {old_batch} -> {config.batch_size}, "
        f"accumulation {old_accumulation} -> {config.accumulation_steps}, "
        f"target_global_batch {old_target_global_batch} -> "
        f"{config.target_global_batch_size}"
    )


def save_training_state(
    config: ExperimentConfig,
    context: DistributedContext,
    model,
    optimizer,
    scheduler,
    scaler,
    *,
    epoch: int,
    global_step: int,
    best_ap: float,
    history: list[dict],
    is_best: bool,
    is_final: bool,
) -> None:
    rng_states = all_gather_objects(_local_rng_state(), context)

    if not context.is_main:
        return

    payload = {
        "format_version": 1,
        "upstream_commit": upstream_commit(),
        "method": config.method,
        "config": config.as_dict(),
        "comparison_fingerprint": config.comparison_fingerprint,
        "detector_fingerprint": config.detector_fingerprint,
        "initialization_fingerprint": getattr(
            unwrap_model(model),
            "initialization_fingerprint",
            None,
        ),
        "epoch": epoch,
        "global_step": global_step,
        "best_ap": best_ap,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_states": rng_states,
        "history": history,
    }

    _atomic_save(payload, config.latest_checkpoint)

    if is_best:
        _atomic_save(payload, config.best_checkpoint)

    if is_final:
        _atomic_save(payload, config.final_checkpoint)

    write_history(config.history_path, history)


def resolve_resume_path(
    config: ExperimentConfig,
    resume: str | Path | None,
) -> Path | None:
    if resume is None:
        return None

    if str(resume).lower() == "auto":
        return (
            config.latest_checkpoint
            if config.latest_checkpoint.is_file()
            else None
        )

    path = Path(resume).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return path


def load_training_state(
    path: Path,
    config: ExperimentConfig,
    context: DistributedContext,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> tuple[int, int, float, list[dict]]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("upstream_commit") != upstream_commit():
        raise ValueError(
            "Checkpoint official DINO commit mismatch"
        )

    if checkpoint.get("method") != config.method:
        raise ValueError(
            "Checkpoint method mismatch"
        )

    _validate_resume_compatibility(
        checkpoint,
        config,
    )

    if checkpoint.get("detector_fingerprint") != config.detector_fingerprint:
        raise ValueError(
            "Checkpoint detector recipe mismatch"
        )

    current_initialization = getattr(
        unwrap_model(model),
        "initialization_fingerprint",
        None,
    )

    if (
        current_initialization is not None
        and checkpoint.get("initialization_fingerprint")
        != current_initialization
    ):
        raise ValueError(
            "Checkpoint common detector initialization mismatch"
        )

    unwrap_model(model).load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if scaler is not None:
        scaler.load_state_dict(
            checkpoint.get("scaler_state_dict", {})
        )

    states = checkpoint.get("rng_states", [])

    if states and context.rank < len(states):
        _restore_rng_state(
            states[context.rank]
        )

    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        float(
            checkpoint.get(
                "best_ap",
                float("-inf"),
            )
        ),
        list(
            checkpoint.get(
                "history",
                [],
            )
        ),
    )