from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "dino"
DQR_V2_ROOT = PROJECT_ROOT / "dqr-v2"
DQR_V2_1_ROOT = PROJECT_ROOT / "dqr-v2.1"
DQR_V2_2_ROOT = PROJECT_ROOT / "dqr-v2_2"
DQR_V2_3_ROOT = PROJECT_ROOT / "dqrv2_3"
DQR_V3_ROOT = PROJECT_ROOT / "dqr-v3"
DQR_V3_1_ROOT = PROJECT_ROOT / "dqr-v3_1"


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


def ensure_dqr_v2_imports() -> Path:
    marker = DQR_V2_ROOT / "dqr_v2" / "model.py"
    if not marker.is_file():
        raise RuntimeError(f"Missing BQR-DN V2 package: {DQR_V2_ROOT}")
    path = str(DQR_V2_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return DQR_V2_ROOT


def ensure_dqr_v2_1_imports() -> Path:
    marker = DQR_V2_1_ROOT / "dqr_v2_1" / "model.py"
    if not marker.is_file():
        raise RuntimeError(f"Missing BQR-DN V2.1 package: {DQR_V2_1_ROOT}")
    path = str(DQR_V2_1_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return DQR_V2_1_ROOT


def ensure_dqr_v2_2_imports() -> Path:
    marker = DQR_V2_2_ROOT / "dqr_v2_2" / "model.py"
    if not marker.is_file():
        raise RuntimeError(f"Missing BQR-DN V2.2 package: {DQR_V2_2_ROOT}")
    path = str(DQR_V2_2_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return DQR_V2_2_ROOT


def ensure_dqr_v2_3_imports() -> Path:
    marker = DQR_V2_3_ROOT / "dqr_v2_3" / "model.py"
    if not marker.is_file():
        raise RuntimeError(f"Missing BQR-DN V2.3 package: {DQR_V2_3_ROOT}")
    path = str(DQR_V2_3_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return DQR_V2_3_ROOT


def ensure_dqr_v3_imports() -> Path:
    marker = DQR_V3_ROOT / "dqr_v3" / "model.py"
    if not marker.is_file():
        raise RuntimeError(f"Missing BQR-DN V3 package: {DQR_V3_ROOT}")
    path = str(DQR_V3_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return DQR_V3_ROOT


def ensure_dqr_v3_1_imports() -> Path:
    # V3.1 intentionally changes only the configured scale-prior weight and
    # therefore shares the V3 implementation.
    ensure_dqr_v3_imports()
    marker = DQR_V3_1_ROOT / "dqr_v3_1" / "model.py"
    if not marker.is_file():
        raise RuntimeError(f"Missing BQR-DN V3.1 package: {DQR_V3_1_ROOT}")
    path = str(DQR_V3_1_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return DQR_V3_1_ROOT
