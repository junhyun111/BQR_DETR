import pytest

from gt_guided_dino.config import ExperimentConfig, smoke_config


def test_default_recipe_matches_official_dino_r50():
    config = ExperimentConfig()
    assert config.model_recipe["backbone"] == "resnet50"
    assert config.num_queries == 900
    assert config.enc_layers == 6
    assert config.dec_layers == 6
    assert config.epochs == 12
    assert config.lr_drop_epoch == 11


def test_smoke_recipe_is_valid():
    config = smoke_config()
    assert config.epochs == 2
    assert config.lr_drop_epoch == 1
    assert config.num_queries == 20
    assert config.hidden_dim == 256


def test_bqr_dn_v2_recipe_is_valid():
    config = smoke_config(method="bqr_dn_v2")
    assert config.method == "bqr_dn_v2"
    assert config.dn_number > 0
    assert config.bqr_points_per_level == 4
    assert config.bqr_dn_weight == 1.0


def test_bqr_dn_v2_1_recipe_is_valid():
    config = smoke_config(method="bqr_dn_v2_1")
    assert config.method == "bqr_dn_v2_1"
    assert config.bqr_content_attention
    assert config.bqr_attention_dim == 64
    assert config.bqr_content_scale_init == 0.05
    assert config.bqr_attention_temperature == 1.0


def test_bqr_dn_v3_recipe_is_four_scale_and_five_point():
    config = smoke_config(method="bqr_dn_v3")
    assert config.method == "bqr_dn_v3"
    assert config.num_feature_levels == 4
    assert config.bqr_points_per_level == 5
    assert config.bqr_scale_aware
    assert config.bqr_target_cells == 4.0
    assert config.bqr_scale_sigma == 0.8
    assert config.bqr_scale_weight == 0.5
    assert config.bqr_scale_logit_floor == -4.0


def test_bqr_dn_v3_1_uses_weaker_scale_prior_in_an_isolated_run():
    config = smoke_config(method="bqr_dn_v3_1")
    assert config.method == "bqr_dn_v3_1"
    assert config.run_dir.name == "seed_42"
    assert config.run_dir.parent.name == "bqr_dn_v3_1"
    assert config.bqr_points_per_level == 5
    assert config.bqr_scale_weight == 0.5


def test_bqr_dn_v3_rejects_non_five_point_configuration():
    with pytest.raises(ValueError, match="requires bqr_points_per_level=5"):
        smoke_config(method="bqr_dn_v3", bqr_points_per_level=4)
