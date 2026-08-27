from __future__ import annotations

import torch

from gt_guided_dino.config import smoke_config
from gt_guided_dino.model import ResearchModel
from gt_guided_dino.upstream import ensure_dqr_v3_1_imports


ensure_dqr_v3_1_imports()
from dqr_v3_1 import METHOD_NAME  # noqa: E402


class _Detector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.decoder = torch.nn.Identity()


def test_v3_1_routes_to_an_isolated_method_with_half_scale_weight():
    config = smoke_config(method="bqr_dn_v3_1")
    model = ResearchModel(_Detector(), config)

    assert METHOD_NAME == "bqr_dn_v3_1"
    assert model.method == "bqr_dn_v3_1"
    assert model.bqr_bridge is model.detector.transformer.decoder
    assert model.bqr_bridge.fusion.scale_weight == 0.5
