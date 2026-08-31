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
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _snapshot_or_save(source: Path, destination: Path, payload: dict) -> None:
    """Use a hard-link snapshot when supported, otherwise serialize normally."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    except OSError:
        if temporary.exists():
            temporary.unlink()
        _atomic_save(payload, destination)


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
    is_milestone: bool,
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
        "method_fingerprint": config.method_fingerprint,
        "initialization_fingerprint": getattr(
            unwrap_model(model), "initialization_fingerprint", None
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
        _snapshot_or_save(config.latest_checkpoint, config.best_checkpoint, payload)
    if is_milestone:
        _snapshot_or_save(
            config.latest_checkpoint,
            config.checkpoint_dir / f"epoch_{epoch + 1:02d}.pt",
            payload,
        )
    if is_final:
        _snapshot_or_save(config.latest_checkpoint, config.final_checkpoint, payload)
    write_history(config.history_path, history)


def resolve_resume_path(
    config: ExperimentConfig, resume: str | Path | None
) -> Path | None:
    if resume is None:
        return None
    if str(resume).lower() == "auto":
        return config.latest_checkpoint if config.latest_checkpoint.is_file() else None
    path = Path(resume).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
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
    checkpoint = torch.load(path, map_location="cpu")
    checks = {
        "upstream commit": (checkpoint.get("upstream_commit"), upstream_commit()),
        "method": (checkpoint.get("method"), config.method),
        "comparison recipe": (
            checkpoint.get("comparison_fingerprint"),
            config.comparison_fingerprint,
        ),
        "detector recipe": (
            checkpoint.get("detector_fingerprint"),
            config.detector_fingerprint,
        ),
        "method recipe": (
            checkpoint.get("method_fingerprint"),
            config.method_fingerprint,
        ),
    }
    mismatches = {
        name: values for name, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        detail = ", ".join(
            f"{name}: checkpoint={old!r}, current={new!r}"
            for name, (old, new) in mismatches.items()
        )
        raise ValueError(f"Checkpoint recipe mismatch: {detail}")
    current_initialization = getattr(
        unwrap_model(model), "initialization_fingerprint", None
    )
    if (
        current_initialization is not None
        and checkpoint.get("initialization_fingerprint") != current_initialization
    ):
        raise ValueError("Checkpoint common initialization mismatch")
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    states = checkpoint.get("rng_states", [])
    if states and context.rank < len(states):
        _restore_rng_state(states[context.rank])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_ap", float("-inf"))),
        list(checkpoint.get("history", [])),
    )
