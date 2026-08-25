from __future__ import annotations

import pytest
import torch

from dqr_coco.config import ExperimentConfig
from dqr_coco.engine import _make_scaler
from dqr_coco.evaluation import _float_tree
from dqr_coco.model import ResearchModel, build_official_components


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke test")
@pytest.mark.parametrize("precision", ("fp32", "fp16"))
def test_official_dino_bqr_v2_forward_backward_cuda(tmp_path, precision):
    device = torch.device("cuda", 0)
    config = ExperimentConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "artifacts",
        torch_cache=tmp_path / "cache",
        method="bqr_dn_v2",
        epochs=2,
        lr_drop_epoch=1,
        num_queries=20,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=512,
        dn_number=4,
        precision=precision,
    )
    detector, criterion, _ = build_official_components(
        config, device, pretrained_backbone=False
    )
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
        },
        {
            "labels": torch.tensor([2], device=device),
            "boxes": torch.tensor([[0.6, 0.6, 0.2, 0.2]], device=device),
            "area": torch.tensor([500.0], device=device),
        },
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = _make_scaler(config)
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=precision == "fp16"
    ):
        outputs = model(images, targets)
    losses = criterion(
        _float_tree(outputs),
        targets,
        num_boxes_override=torch.tensor(1.5, device=device),
    )
    total = sum(
        losses[name] * weight
        for name, weight in criterion.weight_dict.items()
        if name in losses
    )
    scaler.scale(total).backward()
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(total)
    assert model.bqr_bridge is not None
    assert not any(name.startswith("bqr_bridge.") for name in model.state_dict())
    assert model.bqr_bridge.fusion.gate[-1].weight.grad is not None
    assert model.training_diagnostics()["bqr_valid_queries"] > 0
