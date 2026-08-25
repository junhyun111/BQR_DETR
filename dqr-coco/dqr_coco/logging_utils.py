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
from .upstream import upstream_commit


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict) -> None:
    sanitized = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in row.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitized, sort_keys=True, allow_nan=False) + "\n")


def write_run_metadata(
    config: ExperimentConfig, context: DistributedContext
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
        "gpu_names": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
    }
    _atomic_text(
        config.run_dir / "environment.json",
        json.dumps(environment, indent=2, sort_keys=True),
    )
