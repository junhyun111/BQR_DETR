from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .upstream import upstream_commit


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_checkpoint(
    config: ExperimentConfig,
    model,
    optimizer,
    scheduler,
    scaler,
    *,
    epoch: int,
    history: list[dict],
) -> None:
    state = {
        "format_version": 1,
        "upstream_commit": upstream_commit(),
        "method": config.method,
        "config": config.as_dict(),
        "model_recipe": config.model_recipe,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": _rng_state(),
        "history": history,
    }
    _atomic_save(state, config.latest_checkpoint)
    if (epoch + 1) % config.save_every == 0 or epoch + 1 == config.epochs:
        _atomic_save(state, config.checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")
    write_history(config.history_path, history)


def load_checkpoint(
    path: Path,
    config: ExperimentConfig,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> tuple[int, list[dict]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("upstream_commit") != upstream_commit():
        raise ValueError("Checkpoint DINO commit does not match this checkout")
    if checkpoint.get("method") != config.method:
        raise ValueError("Checkpoint method does not match the requested method")
    if checkpoint.get("model_recipe") != config.model_recipe:
        raise ValueError("Checkpoint model recipe does not match the requested model")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if optimizer is not None and checkpoint.get("rng_state"):
        _restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]) + 1, list(checkpoint.get("history", []))
