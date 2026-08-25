from __future__ import annotations

import pytest

from dqr_coco.config import ExperimentConfig


def test_default_recipe_is_two_gpu_global_batch_16(tmp_path):
    config = ExperimentConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "artifacts",
        torch_cache=tmp_path / "cache",
    )
    assert config.method == "bqr_dn_v2"
    assert config.precision == "fp16"
    assert config.batch_size == 2
    assert config.accumulation_steps == 4
    config.validate_world_size(2)
    with pytest.raises(ValueError, match="Effective global batch mismatch"):
        config.validate_world_size(1)


def test_baseline_and_v2_share_comparison_fingerprint(tmp_path):
    common = {
        "data_root": tmp_path / "coco",
        "output_root": tmp_path / "artifacts",
        "torch_cache": tmp_path / "cache",
    }
    baseline = ExperimentConfig(method="baseline", **common)
    v2 = ExperimentConfig(method="bqr_dn_v2", **common)
    assert baseline.comparison_fingerprint == v2.comparison_fingerprint
    assert baseline.detector_fingerprint == v2.detector_fingerprint
    assert baseline.run_dir != v2.run_dir
