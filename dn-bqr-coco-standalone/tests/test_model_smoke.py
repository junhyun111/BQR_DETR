from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from bqr_dn.config import ExperimentConfig
from bqr_dn.evaluation import float_tree
from bqr_dn.model import ResearchModel, build_official_components


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
def test_official_dn_deformable_bqr_fp16_forward_backward(tmp_path):
    device = torch.device("cuda", 0)
    config = ExperimentConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "out",
        torch_cache=tmp_path / "cache",
        method="bqr",
        epochs=1,
        eval_epochs=(1,),
        lr_drop_epoch=1,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=512,
        num_queries=20,
        dn_scalar=2,
        precision="fp16",
    )
    detector, criterion, _ = build_official_components(
        config, device, pretrained_backbone=False
    )
    baseline = ResearchModel(
        copy.deepcopy(detector), replace(config, method="baseline")
    ).to(device).eval()
    model = ResearchModel(detector, config).to(device).train()
    model.set_diagnostics_enabled(True)
    images = [torch.rand(3, 64, 64, device=device) for _ in range(2)]
    targets = [
        {
            "labels": torch.tensor([1, 3], device=device),
            "boxes": torch.tensor(
                [[0.5, 0.5, 0.3, 0.3], [0.25, 0.25, 0.15, 0.15]],
                device=device,
            ),
            "area": torch.tensor([1200.0, 200.0], device=device),
            "size": torch.tensor([64, 64], device=device),
        },
        {
            "labels": torch.tensor([2], device=device),
            "boxes": torch.tensor([[0.6, 0.6, 0.2, 0.2]], device=device),
            "area": torch.tensor([500.0], device=device),
            "size": torch.tensor([64, 64], device=device),
        },
    ]
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        baseline_outputs, _ = baseline(images)
        initial_bqr_outputs, _ = model(images)
    assert torch.equal(
        baseline_outputs["pred_logits"], initial_bqr_outputs["pred_logits"]
    )
    assert torch.equal(
        baseline_outputs["pred_boxes"], initial_bqr_outputs["pred_boxes"]
    )

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs, mask_dict = model(images, targets)
    losses = criterion(float_tree(outputs), targets, float_tree(mask_dict))
    total = sum(
        losses[name] * weight
        for name, weight in criterion.weight_dict.items()
        if name in losses
    )
    total.backward()
    assert torch.isfinite(total)
    assert model.bqr_bridge is not None
    assert model.bqr_bridge.fusion.feature_projection[-1].weight.grad is not None
    assert model.training_diagnostics()["bqr_valid_queries"] > 0

    model.eval()
    model.set_diagnostics_enabled(False)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        eval_outputs, eval_mask = model(images)
    assert eval_mask is None
    assert eval_outputs["pred_logits"].shape[1] == config.num_queries
    assert model.training_diagnostics() == {}
