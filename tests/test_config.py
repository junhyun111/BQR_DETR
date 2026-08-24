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
