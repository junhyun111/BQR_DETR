from __future__ import annotations

import torch

from gt_guided_dino.config import smoke_config
from gt_guided_dino.model import ResearchModel
from gt_guided_dino.upstream import ensure_dqr_v2_3_imports, ensure_dqr_v2_imports


ensure_dqr_v2_imports()
ensure_dqr_v2_3_imports()
from dqr_v2_3 import METHOD_NAME  # noqa: E402
from dqr_v2_3.query_fusion import (  # noqa: E402
    NormConstrainedRegionGuidedDNFusion,
)


class _Detector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.decoder = torch.nn.Identity()


def _fusion_inputs():
    return (
        torch.randn(2, 4, 8, requires_grad=True),
        torch.rand(2, 4, 4),
        torch.tensor([[True, True, True, True], [True, False, True, False]]),
        torch.randn(2, 5, 8, requires_grad=True),
        torch.tensor([[2, 2], [1, 1]]),
        torch.tensor([0, 4]),
        torch.ones(2, 2, 2),
    )


def test_v2_3_routes_to_norm_cap_with_v2_initialization():
    v2 = ResearchModel(_Detector(), smoke_config(method="bqr_dn_v2"))
    v2_3 = ResearchModel(_Detector(), smoke_config(method="bqr_dn_v2_3"))

    assert METHOD_NAME == "bqr_dn_v2_3"
    assert v2_3.bqr_bridge.fusion.residual_ratio == 0.5
    assert v2.bqr_bridge.fusion.state_dict().keys() == v2_3.bqr_bridge.fusion.state_dict().keys()
    for name, value in v2.bqr_bridge.fusion.state_dict().items():
        torch.testing.assert_close(value, v2_3.bqr_bridge.fusion.state_dict()[name])


def test_v2_3_caps_final_delta_relative_to_query_norm_and_backpropagates():
    torch.manual_seed(17)
    module = NormConstrainedRegionGuidedDNFusion(8, 2, 4, -2.0, 0.5)
    with torch.no_grad():
        module.feature_projection[-1].weight.mul_(100_000.0)
    inputs = _fusion_inputs()

    fused, diagnostics = module(*inputs)
    queries, _, valid_mask, memory, *_ = inputs
    active_queries = queries[valid_mask]
    final_delta = fused[valid_mask] - active_queries
    final_ratio = final_delta.norm(dim=-1) / active_queries.norm(dim=-1).clamp_min(1e-6)

    assert bool((final_ratio <= 0.50001).all())
    assert float(diagnostics["bqr_clip_rate"]) > 0.0
    assert float(diagnostics["bqr_raw_delta_ratio"]) >= float(
        diagnostics["bqr_final_delta_ratio"]
    )
    torch.testing.assert_close(
        diagnostics["bqr_final_delta_ratio"],
        final_ratio.mean(),
    )
    assert torch.equal(fused[~valid_mask], queries[~valid_mask])

    fused.square().mean().backward()
    assert queries.grad is not None and float(queries.grad.abs().sum()) > 0
    assert memory.grad is not None and float(memory.grad.abs().sum()) > 0
