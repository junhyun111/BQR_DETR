from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "dino"


def ensure_upstream_imports() -> Path:
    marker = UPSTREAM_ROOT / "models" / "dino" / "dino.py"
    if not marker.is_file():
        raise RuntimeError(f"Missing vendored official DINO source: {UPSTREAM_ROOT}")
    path = str(UPSTREAM_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return UPSTREAM_ROOT


def upstream_commit() -> str:
    return (UPSTREAM_ROOT / "UPSTREAM_COMMIT").read_text(encoding="ascii").strip()

