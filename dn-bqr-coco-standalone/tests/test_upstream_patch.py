from __future__ import annotations

from bqr_dn.upstream import EXPECTED_UPSTREAM_COMMIT, UPSTREAM_ROOT, upstream_commit


def test_upstream_commit_is_pinned():
    assert upstream_commit() == EXPECTED_UPSTREAM_COMMIT


def test_dn_components_are_device_aware():
    source = (
        UPSTREAM_ROOT
        / "models"
        / "dn_dab_deformable_detr"
        / "dn_components.py"
    ).read_text(encoding="utf-8")
    assert ".cuda()" not in source
    assert ".to('cuda')" not in source
    assert "device = tgt_weight.device" in source


def test_msda_custom_operator_is_float32_under_amp():
    source = (
        UPSTREAM_ROOT
        / "models"
        / "dn_dab_deformable_detr"
        / "ops"
        / "modules"
        / "ms_deform_attn.py"
    ).read_text(encoding="utf-8")
    assert "torch.float16, torch.bfloat16" in source
    assert "value.float()" in source
    assert "sampling_locations.float()" in source
    assert "if not HAS_MSDA" in source


def test_msda_can_be_compiled_during_gpu_less_docker_build():
    source = (
        UPSTREAM_ROOT
        / "models"
        / "dn_dab_deformable_detr"
        / "ops"
        / "setup.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ.get("FORCE_CUDA", "0")' in source
