import torch

from gt_guided_dino.upstream import ensure_upstream_imports


ensure_upstream_imports()
from models.dino.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


def test_pytorch_msda_fallback_forward_backward():
    value = torch.randn(1, 5, 2, 4, requires_grad=True)
    shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long)
    locations = torch.rand(1, 3, 2, 2, 2, 2, requires_grad=True)
    weights = torch.softmax(torch.randn(1, 3, 2, 2, 2), dim=-1)
    output = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
    assert output.shape == (1, 3, 8)
    output.sum().backward()
    assert value.grad is not None
    assert locations.grad is not None
