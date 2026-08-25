from __future__ import annotations

import inspect

from dqr_coco.upstream import ensure_upstream_imports


def test_official_criterion_accepts_accumulation_window_normalizer():
    ensure_upstream_imports()
    from models.dino.dino import SetCriterion

    assert "num_boxes_override" in inspect.signature(SetCriterion.forward).parameters


def test_docker_recipe_requires_compiled_cuda_extension():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    assert "FORCE_CUDA=1" in dockerfile
    assert "TORCH_CUDA_ARCH_LIST=8.9" in dockerfile
    assert "--nproc_per_node=${NPROC_PER_NODE:-2}" in compose
    assert "${BATCH_SIZE:-2}" in compose
    assert "${ACCUMULATION_STEPS:-4}" in compose
