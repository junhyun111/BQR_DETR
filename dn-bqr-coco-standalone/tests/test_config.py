from __future__ import annotations

from dataclasses import replace

import pytest

from bqr_dn.config import ExperimentConfig


def test_official_dn_deformable_defaults_and_pair_fingerprint(tmp_path):
    baseline = ExperimentConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "out",
        torch_cache=tmp_path / "cache",
        method="baseline",
    )
    bqr = replace(baseline, method="bqr")
    assert baseline.num_queries == 300
    assert baseline.dn_scalar == 5
    assert baseline.dn_label_noise_scale == 0.2
    assert baseline.dn_box_noise_scale == 0.4
    assert baseline.comparison_fingerprint == bqr.comparison_fingerprint
    assert baseline.method_fingerprint != bqr.method_fingerprint
    assert baseline.official_args.__name__ == "official_args"


def test_stop_epoch_does_not_change_comparison_recipe(tmp_path):
    base = ExperimentConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "out",
        torch_cache=tmp_path / "cache",
        stop_after_epoch=10,
    )
    resumed = replace(base, stop_after_epoch=20)
    assert base.comparison_fingerprint == resumed.comparison_fingerprint


def test_eval_epoch_must_be_inside_schedule(tmp_path):
    with pytest.raises(ValueError, match="eval_epochs"):
        ExperimentConfig(
            data_root=tmp_path / "coco",
            output_root=tmp_path / "out",
            torch_cache=tmp_path / "cache",
            epochs=10,
            eval_epochs=(5, 10, 15),
        )
