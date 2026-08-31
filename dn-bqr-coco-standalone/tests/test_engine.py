from __future__ import annotations

import torch

from bqr_dn.config import ExperimentConfig
from bqr_dn.data import DataBundle
from bqr_dn.distributed import DistributedContext
from bqr_dn.engine import _make_scaler, train_one_epoch


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        del enabled

    def bqr_parameters(self):
        return []

    def forward(self, samples, targets=None):
        del targets
        return {"prediction": self.linear(samples)}, None


class TinyCriterion(torch.nn.Module):
    weight_dict = {"loss_value": 1.0}

    def forward(self, outputs, targets, mask_dict):
        del mask_dict
        expected = torch.stack([target["value"] for target in targets])
        return {"loss_value": (outputs["prediction"].flatten() - expected).pow(2).mean()}


def test_accumulation_handles_final_partial_window(tmp_path):
    config = ExperimentConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "out",
        torch_cache=tmp_path / "cache",
        method="baseline",
        epochs=1,
        eval_epochs=(1,),
        lr_drop_epoch=1,
        batch_size=2,
        accumulation_steps=2,
        target_global_batch_size=4,
        precision="fp32",
    )
    batches = []
    for value in (0.0, 1.0, 2.0):
        samples = torch.full((2, 3), value)
        targets = [{"value": torch.tensor(value)} for _ in range(2)]
        batches.append((samples, targets))
    data = DataBundle(
        train_loader=batches,
        val_loader=[],
        train_sampler=None,
        coco_api=None,
        subset_manifest=None,
    )
    model = TinyModel()
    criterion = TinyCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    context = DistributedContext(0, 1, 0, torch.device("cpu"), False)
    metrics, global_step = train_one_epoch(
        config,
        context,
        model,
        criterion,
        optimizer,
        data,
        _make_scaler(config),
        epoch=0,
        global_step=0,
    )
    assert global_step == 2
    assert metrics["optimizer_steps"] == 2
    assert metrics["train_images"] == 6
    assert torch.isfinite(torch.tensor(metrics["train_loss"]))
