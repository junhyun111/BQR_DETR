from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()


def initialize_distributed(
    *,
    allow_cpu: bool = False,
) -> DistributedContext:
    distributed = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    )

    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(
            os.environ["WORLD_SIZE"]
        )
        local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                0,
            )
        )

        if not torch.cuda.is_available():
            if not allow_cpu:
                raise RuntimeError(
                    "NCCL DDP training requires CUDA"
                )

            device = torch.device("cpu")
            backend = "gloo"

        else:
            torch.cuda.set_device(
                local_rank
            )

            device = torch.device(
                "cuda",
                local_rank,
            )

            backend = "nccl"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )

        return DistributedContext(
            rank,
            world_size,
            local_rank,
            device,
            True,
        )

    if torch.cuda.is_available():
        return DistributedContext(
            0,
            1,
            0,
            torch.device(
                "cuda",
                0,
            ),
            False,
        )

    if not allow_cpu:
        raise RuntimeError(
            "CUDA is required; "
            "pass --allow-cpu only for smoke tests"
        )

    return DistributedContext(
        0,
        1,
        0,
        torch.device("cpu"),
        False,
    )


def cleanup_distributed(
    context: DistributedContext,
) -> None:
    if (
        context.distributed
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


def seed_everything(
    seed: int,
    rank: int,
    deterministic: bool,
) -> None:
    rank_seed = seed + rank

    random.seed(
        rank_seed
    )

    np.random.seed(
        rank_seed
    )

    torch.manual_seed(
        rank_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            rank_seed
        )

    torch.backends.cudnn.benchmark = (
        not deterministic
    )

    torch.backends.cudnn.deterministic = (
        deterministic
    )

    if deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )


def all_reduce_sum(
    tensor: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    if context.distributed:
        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

    return tensor


def all_gather_objects(
    value,
    context: DistributedContext,
) -> list:
    if not context.distributed:
        return [value]

    gathered = [
        None
        for _ in range(
            context.world_size
        )
    ]

    dist.all_gather_object(
        gathered,
        value,
    )

    return gathered


def gather_objects_to_main(
    value,
    context: DistributedContext,
) -> list | None:
    """
    Gather an object only to rank 0.

    Unlike all_gather_object, non-main ranks do not receive a
    duplicate copy of every rank's validation predictions.
    """

    if not context.distributed:
        return [value]

    if hasattr(
        dist,
        "gather_object",
    ):
        if context.is_main:
            gathered = [
                None
                for _ in range(
                    context.world_size
                )
            ]
        else:
            gathered = None

        dist.gather_object(
            value,
            object_gather_list=gathered,
            dst=0,
        )

        return gathered

    # Compatibility fallback for older PyTorch.
    gathered = [
        None
        for _ in range(
            context.world_size
        )
    ]

    dist.all_gather_object(
        gathered,
        value,
    )

    if context.is_main:
        return gathered

    return None


def unwrap_model(
    model: torch.nn.Module,
) -> torch.nn.Module:
    return (
        model.module
        if hasattr(
            model,
            "module",
        )
        else model
    )


def require_compiled_msda(
    device: torch.device,
) -> None:
    if device.type != "cuda":
        return

    from .upstream import (
        ensure_upstream_imports,
    )

    ensure_upstream_imports()

    from models.dino.ops.functions import (
        HAS_MSDA,
    )

    if not HAS_MSDA:
        raise RuntimeError(
            "The compiled "
            "MultiScaleDeformableAttention CUDA "
            "extension is required. "
            "Rebuild the dqr-coco Docker image "
            "before training."
        )