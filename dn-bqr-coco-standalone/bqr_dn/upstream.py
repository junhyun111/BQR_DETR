from __future__ import annotations

import hashlib
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PACKAGE_ROOT / "third_party" / "dn_detr"
EXPECTED_UPSTREAM_COMMIT = "ff3902a20d521ead052d1243ff249b19bc1ce531"


def ensure_upstream_imports() -> None:
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def upstream_commit() -> str:
    marker = UPSTREAM_ROOT / "UPSTREAM_COMMIT"
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return value
    return EXPECTED_UPSTREAM_COMMIT


def upstream_source_fingerprint() -> str:
    """Fingerprint the small set of upstream files that define this detector."""
    relative_paths = (
        "models/__init__.py",
        "models/dn_dab_deformable_detr/dab_deformable_detr.py",
        "models/dn_dab_deformable_detr/backbone.py",
        "models/dn_dab_deformable_detr/deformable_transformer.py",
        "models/dn_dab_deformable_detr/dn_components.py",
        "models/dn_dab_deformable_detr/ops/modules/ms_deform_attn.py",
        "models/dn_dab_deformable_detr/ops/functions/ms_deform_attn_func.py",
        "models/dn_dab_deformable_detr/ops/setup.py",
        "datasets/coco.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = UPSTREAM_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
