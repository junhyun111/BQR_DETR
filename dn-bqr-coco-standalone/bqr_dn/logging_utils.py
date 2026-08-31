from __future__ import annotations

import csv
import json
import math
import os
import platform
from pathlib import Path

import torch

from .config import ExperimentConfig
from .distributed import DistributedContext
from .upstream import upstream_commit, upstream_source_fingerprint


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict) -> None:
    sanitized = {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in row.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitized, sort_keys=True, allow_nan=False) + "\n")


def write_run_metadata(
    config: ExperimentConfig, context: DistributedContext, *, subset_manifest: dict | None
) -> None:
    if not context.is_main:
        return
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        config.run_dir / "config.json",
        json.dumps(config.as_dict(), indent=2, sort_keys=True),
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "world_size": context.world_size,
        "upstream_commit": upstream_commit(),
        "upstream_source_fingerprint": upstream_source_fingerprint(),
        "gpu_names": (
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        ),
    }
    _atomic_text(
        config.run_dir / "environment.json",
        json.dumps(environment, indent=2, sort_keys=True),
    )
    if subset_manifest is not None:
        compact = {
            key: value
            for key, value in subset_manifest.items()
            if key != "image_ids"
        }
        compact["manifest_path"] = str(config.subset_manifest_path)
        compact["image_id_count"] = len(subset_manifest["image_ids"])
        _atomic_text(
            config.run_dir / "subset.json",
            json.dumps(compact, indent=2, sort_keys=True),
        )
