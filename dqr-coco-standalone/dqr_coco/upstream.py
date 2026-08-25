from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT
UPSTREAM_ROOT = REPOSITORY_ROOT / "third_party" / "dino"


def ensure_upstream_imports() -> Path:
    marker = UPSTREAM_ROOT / "models" / "dino" / "dino.py"
    if not marker.is_file():
        raise RuntimeError(
            "Vendored official DINO source is missing from this standalone package: "
            f"{UPSTREAM_ROOT}."
        )
    upstream = str(UPSTREAM_ROOT)
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    return UPSTREAM_ROOT


def upstream_commit() -> str:
    return (UPSTREAM_ROOT / "UPSTREAM_COMMIT").read_text(encoding="ascii").strip()
