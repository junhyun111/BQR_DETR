import torch

from gt_guided_dino.model import EncoderContext, GTGuidedAuxiliary


def test_gt_guided_auxiliary_is_differentiable():
    module = GTGuidedAuxiliary(hidden_dim=8, num_levels=2, points_per_level=4)
    memory = torch.randn(1, 5, 8, requires_grad=True)
    queries = torch.randn(1, 3, 8, requires_grad=True)
    context = EncoderContext(
        memory=memory,
        spatial_shapes=torch.tensor([[2, 2], [1, 1]]),
        level_start_index=torch.tensor([0, 4]),
        valid_ratios=torch.ones(1, 2, 2),
        padding_mask=torch.zeros(1, 5, dtype=torch.bool),
        query_features=queries,
    )
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]
    indices = [(torch.tensor([1]), torch.tensor([0]))]
    losses = module(context, targets, indices, normalizer=1.0)
    total = losses["loss_aux_bbox"] + losses["loss_aux_giou"]
    total.backward()
    assert torch.isfinite(total)
    assert memory.grad is not None
    assert queries.grad is not None
    assert float(losses["aux_gate_mean"]) > 0.0

