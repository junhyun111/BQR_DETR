from __future__ import annotations

import torch

from gt_guided_dino.config import smoke_config
from gt_guided_dino.model import ResearchModel
from gt_guided_dino.upstream import ensure_dqr_v2_2_imports, ensure_dqr_v2_imports


ensure_dqr_v2_imports()
ensure_dqr_v2_2_imports()
from dqr_v2.query_fusion import RegionGuidedDNFusion as V2Fusion  # noqa: E402
from dqr_v2_2 import METHOD_NAME  # noqa: E402
from dqr_v2_2.query_fusion import RegionGuidedDNFusion as V22Fusion  # noqa: E402


class _Detector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.decoder = torch.nn.Identity()


def _fusion_inputs():
    return (
        torch.randn(2, 4, 8),
        torch.rand(2, 4, 4),
        torch.tensor([[True, True, True, True], [True, False, True, False]]),
        torch.randn(2, 5, 8),
        torch.tensor([[2, 2], [1, 1]]),
        torch.tensor([0, 4]),
        torch.ones(2, 2, 2),
    )


def test_v2_2_routes_to_half_strength_fusion_with_v2_initialization():
    v2 = ResearchModel(_Detector(), smoke_config(method="bqr_dn_v2"))
    v2_2 = ResearchModel(_Detector(), smoke_config(method="bqr_dn_v2_2"))

    assert METHOD_NAME == "bqr_dn_v2_2"
    assert v2_2.bqr_bridge.fusion.fusion_weight == 0.5
    assert v2.bqr_bridge.fusion.state_dict().keys() == v2_2.bqr_bridge.fusion.state_dict().keys()
    for name, value in v2.bqr_bridge.fusion.state_dict().items():
        torch.testing.assert_close(value, v2_2.bqr_bridge.fusion.state_dict()[name])


def test_v2_2_correction_is_exactly_half_of_v2():
    torch.manual_seed(7)
    full = V2Fusion(8, 2, 4, -2.0)
    torch.manual_seed(7)
    half = V22Fusion(8, 2, 4, -2.0, 0.5)
    inputs = _fusion_inputs()

    full_output, full_diagnostics = full(*inputs)
    half_output, half_diagnostics = half(*inputs)
    queries, _, valid_mask, *_ = inputs

    torch.testing.assert_close(
        half_output[valid_mask] - queries[valid_mask],
        0.5 * (full_output[valid_mask] - queries[valid_mask]),
    )
    torch.testing.assert_close(
        half_diagnostics["bqr_gate_mean"],
        full_diagnostics["bqr_gate_mean"],
    )
    torch.testing.assert_close(
        half_diagnostics["bqr_effective_gate_mean"],
        0.5 * half_diagnostics["bqr_gate_mean"],
    )
    assert torch.equal(half_output[~valid_mask], queries[~valid_mask])
