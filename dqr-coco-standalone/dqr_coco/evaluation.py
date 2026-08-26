from __future__ import annotations

import copy
import ctypes
import gc
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from .config import ExperimentConfig
from .data import DataBundle
from .distributed import (
    DistributedContext,
    gather_objects_to_main,
)
from .upstream import (
    ensure_upstream_imports,
)


ensure_upstream_imports()

from datasets.coco_eval import (  # noqa: E402
    CocoEvaluator,
)


COCO_STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_S",
    "AP_M",
    "AP_L",
    "AR_1",
    "AR_10",
    "AR",
    "AR_S",
    "AR_M",
    "AR_L",
)


# Do not run pycocotools for every batch.
#
# 5000 COCO val images:
#
# old, eval batch=2:
#     ~2500 CocoEvaluator.update() calls total
#
# new:
#     ~79 calls on rank 0
#
# This dramatically reduces COCO.loadRes / Python
# object-construction overhead.
_EVAL_CHUNK_IMAGES = 64


def _autocast(
    config: ExperimentConfig,
    device: torch.device,
):
    enabled = (
        device.type == "cuda"
        and config.precision
        != "fp32"
    )

    dtype = (
        torch.float16
        if config.precision
        == "fp16"
        else torch.bfloat16
    )

    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def _float_tree(
    value,
):
    """
    Backward compatibility.

    Older engine.py revisions import this function.
    Optimized evaluation does NOT use it anymore.
    """

    if torch.is_tensor(
        value
    ):
        return (
            value.float()
            if value.is_floating_point()
            else value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            key:
                _float_tree(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            _float_tree(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            _float_tree(item)
            for item in value
        )

    return value


def _release_host_memory() -> None:
    """
    Force Python cleanup and, on Linux/glibc, return
    free heap pages to the operating system.

    This targets the apparent epoch-by-epoch RSS growth
    after large pycocotools temporary allocations.
    """

    gc.collect()

    if not sys.platform.startswith(
        "linux"
    ):
        return

    try:
        libc = ctypes.CDLL(
            "libc.so.6"
        )

        malloc_trim = getattr(
            libc,
            "malloc_trim",
            None,
        )

        if malloc_trim is not None:
            malloc_trim(
                0
            )

    except (
        OSError,
        AttributeError,
    ):
        pass


def _allocate_prediction_buffer(
    image_count: int,
    num_select: int,
) -> dict[str, torch.Tensor]:
    """
    Allocate compact CPU storage once.

    We store only:
        image_id
        final boxes
        final scores
        final labels

    Auxiliary decoder outputs and COCO eval_imgs are
    never stored during inference.
    """

    return {
        "image_ids":
            torch.empty(
                image_count,
                dtype=torch.int64,
            ),

        "boxes":
            torch.empty(
                (
                    image_count,
                    num_select,
                    4,
                ),
                dtype=torch.float32,
            ),

        "scores":
            torch.empty(
                (
                    image_count,
                    num_select,
                ),
                dtype=torch.float32,
            ),

        # COCO labels are <= 90 here.
        # int32 is plenty and halves label RAM
        # relative to PyTorch int64.
        "labels":
            torch.empty(
                (
                    image_count,
                    num_select,
                ),
                dtype=torch.int32,
            ),
    }


def _copy_batch_predictions(
    buffer:
        dict[str, torch.Tensor],
    offset: int,
    targets: list[dict],
    results: list[dict],
    *,
    num_select: int,
) -> int:
    batch_size = len(
        results
    )

    end = (
        offset
        + batch_size
    )

    if (
        end
        > buffer[
            "image_ids"
        ].shape[0]
    ):
        raise RuntimeError(
            "Validation prediction "
            "buffer overflow"
        )

    counts = [
        int(
            result[
                "scores"
            ].shape[0]
        )
        for result in results
    ]

    if any(
        count != num_select
        for count in counts
    ):
        raise RuntimeError(
            "Optimized COCO evaluation "
            "expects fixed-size DINO outputs. "
            f"Expected {num_select} "
            f"detections/image, got {counts}."
        )

    image_ids = torch.stack(
        [
            target[
                "image_id"
            ].reshape(())
            for target in targets
        ]
    ).to(
        device="cpu",
        dtype=torch.int64,
    )

    boxes = torch.stack(
        [
            result[
                "boxes"
            ]
            for result in results
        ],
        dim=0,
    ).detach().to(
        device="cpu",
        dtype=torch.float32,
    )

    scores = torch.stack(
        [
            result[
                "scores"
            ]
            for result in results
        ],
        dim=0,
    ).detach().to(
        device="cpu",
        dtype=torch.float32,
    )

    labels = torch.stack(
        [
            result[
                "labels"
            ]
            for result in results
        ],
        dim=0,
    ).detach().to(
        device="cpu",
        dtype=torch.int32,
    )

    buffer[
        "image_ids"
    ][offset:end].copy_(
        image_ids
    )

    buffer[
        "boxes"
    ][offset:end].copy_(
        boxes
    )

    buffer[
        "scores"
    ][offset:end].copy_(
        scores
    )

    buffer[
        "labels"
    ][offset:end].copy_(
        labels
    )

    return end


def _finalize_local_coco_evaluator(
    evaluator: CocoEvaluator,
) -> None:
    """
    Finalize CocoEvaluator without calling its distributed
    synchronization path.

    At this point rank 0 already owns predictions from all ranks,
    so another all_gather would only duplicate huge eval_imgs.
    """

    for iou_type in (
        evaluator.iou_types
    ):
        pieces = (
            evaluator.eval_imgs[
                iou_type
            ]
        )

        if not pieces:
            continue

        eval_imgs = (
            np.concatenate(
                pieces,
                axis=2,
            )
        )

        img_ids = np.asarray(
            evaluator.img_ids,
            dtype=np.int64,
        )

        (
            img_ids,
            first_indices,
        ) = np.unique(
            img_ids,
            return_index=True,
        )

        eval_imgs = (
            eval_imgs[
                ...,
                first_indices,
            ]
        )

        coco_eval = (
            evaluator.coco_eval[
                iou_type
            ]
        )

        coco_eval.evalImgs = list(
            eval_imgs.flatten()
        )

        coco_eval.params.imgIds = (
            list(
                img_ids
            )
        )

        coco_eval._paramsEval = (
            copy.deepcopy(
                coco_eval.params
            )
        )


def _evaluate_on_main(
    coco_api,
    gathered:
        list[
            dict[
                str,
                torch.Tensor,
            ]
        ],
) -> tuple[
    list[float],
    int,
]:
    """
    Rank-0-only COCO evaluation.

    Importantly, we never construct one giant
    1.5-million-detection Python dictionary list.

    Predictions are fed into pycocotools in bounded
    chunks of 64 images.
    """

    evaluator = CocoEvaluator(
        coco_api,
        ["bbox"],
    )

    total_images = 0

    for payload in gathered:
        image_ids = payload[
            "image_ids"
        ]

        boxes = payload[
            "boxes"
        ]

        scores = payload[
            "scores"
        ]

        labels = payload[
            "labels"
        ]

        image_count = int(
            image_ids.shape[0]
        )

        total_images += (
            image_count
        )

        for start in range(
            0,
            image_count,
            _EVAL_CHUNK_IMAGES,
        ):
            end = min(
                start
                + _EVAL_CHUNK_IMAGES,
                image_count,
            )

            chunk_ids = (
                image_ids[
                    start:end
                ]
            )

            chunk_boxes = (
                boxes[
                    start:end
                ]
            )

            chunk_scores = (
                scores[
                    start:end
                ]
            )

            chunk_labels = (
                labels[
                    start:end
                ]
            )

            predictions = {
                int(
                    chunk_ids[
                        index
                    ]
                ): {
                    "boxes":
                        chunk_boxes[
                            index
                        ],

                    "scores":
                        chunk_scores[
                            index
                        ],

                    "labels":
                        chunk_labels[
                            index
                        ].to(
                            dtype=
                                torch.int64
                        ),
                }
                for index in range(
                    end - start
                )
            }

            evaluator.update(
                predictions
            )

            del predictions

    _finalize_local_coco_evaluator(
        evaluator
    )

    evaluator.accumulate()

    evaluator.summarize()

    stats = [
        float(value)
        for value
        in evaluator.coco_eval[
            "bbox"
        ].stats
    ]

    del evaluator

    return (
        stats,
        total_images,
    )


@torch.inference_mode()
def evaluate(
    config: ExperimentConfig,
    context: DistributedContext,
    model,
    postprocessors: dict,
    data: DataBundle,
) -> dict[str, float]:
    model.eval()

    if context.device.type == "cuda":
        torch.cuda.synchronize(
            context.device
        )

    started = (
        time.perf_counter()
    )

    bbox_postprocessor = (
        postprocessors[
            "bbox"
        ]
    )

    num_select = int(
        getattr(
            bbox_postprocessor,
            "num_select",
            300,
        )
    )

    local_capacity = len(
        data.val_loader.sampler
    )

    local_predictions = (
        _allocate_prediction_buffer(
            local_capacity,
            num_select,
        )
    )

    local_images = 0

    for samples, targets in (
        data.val_loader
    ):
        # ---------------------------------------------
        # IMPORTANT:
        #
        # Evaluation does not need:
        #   labels
        #   GT boxes
        #   area
        #   iscrowd
        #   current size
        #
        # on the GPU.
        #
        # Previous code moved the entire target dict.
        # ---------------------------------------------

        samples = samples.to(
            context.device,
            non_blocking=True,
        )

        original_sizes = (
            torch.stack(
                [
                    target[
                        "orig_size"
                    ]
                    for target
                    in targets
                ],
                dim=0,
            )
            .to(
                context.device,
                non_blocking=True,
            )
        )

        with _autocast(
            config,
            context.device,
        ):
            outputs = model(
                samples
            )

        # ---------------------------------------------
        # Previous implementation:
        #
        # outputs = _float_tree(outputs)
        #
        # That converted:
        #   pred_logits
        #   pred_boxes
        #   5 auxiliary decoder outputs
        #   intermediate encoder outputs
        #   etc.
        #
        # all to FP32.
        #
        # COCO postprocess uses only these two.
        # ---------------------------------------------

        final_outputs = {
            "pred_logits":
                outputs[
                    "pred_logits"
                ].float(),

            "pred_boxes":
                outputs[
                    "pred_boxes"
                ].float(),
        }

        results = (
            bbox_postprocessor(
                final_outputs,
                original_sizes,
            )
        )

        local_images = (
            _copy_batch_predictions(
                local_predictions,
                local_images,
                targets,
                results,
                num_select=
                    num_select,
            )
        )

        # Release all GPU/Python references immediately.
        del results
        del final_outputs
        del outputs
        del original_sizes
        del samples

    if (
        local_images
        != local_capacity
    ):
        local_predictions = {
            name:
                tensor[
                    :local_images
                ].clone()
            for name, tensor
            in local_predictions.items()
        }

    if context.device.type == "cuda":
        torch.cuda.synchronize(
            context.device
        )

    # ---------------------------------------------
    # Only compact final predictions are gathered.
    #
    # NOT:
    #   COCOeval objects
    #   eval_imgs
    #   auxiliary model outputs
    #
    # Non-main ranks do not receive rank0/rank1 copies.
    # ---------------------------------------------

    gathered = (
        gather_objects_to_main(
            local_predictions,
            context,
        )
    )

    del local_predictions

    metric_count = (
        len(
            COCO_STAT_NAMES
        )
        + 2
    )

    packed_metrics = (
        torch.zeros(
            metric_count,
            dtype=torch.float64,
            device=context.device,
        )
    )

    if context.is_main:
        if gathered is None:
            raise RuntimeError(
                "Rank 0 did not receive "
                "gathered validation predictions"
            )

        (
            stats,
            total_images,
        ) = _evaluate_on_main(
            data.coco_api,
            gathered,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        packed_metrics[
            :len(
                COCO_STAT_NAMES
            )
        ] = torch.tensor(
            stats[
                :len(
                    COCO_STAT_NAMES
                )
            ],
            dtype=torch.float64,
            device=context.device,
        )

        packed_metrics[
            -2
        ] = elapsed

        packed_metrics[
            -1
        ] = float(
            total_images
        )

        del gathered

    # Only 14 float64 values are broadcast.
    #
    # Previously huge COCO eval structures were
    # synchronized across ranks.
    if context.distributed:
        dist.broadcast(
            packed_metrics,
            src=0,
        )

    values = (
        packed_metrics
        .cpu()
        .tolist()
    )

    metrics = {
        name:
            float(
                values[index]
            )
        for index, name
        in enumerate(
            COCO_STAT_NAMES
        )
    }

    metrics[
        "val_time_seconds"
    ] = float(
        values[-2]
    )

    metrics[
        "val_images"
    ] = int(
        values[-1]
    )

    del packed_metrics

    # COCOeval creates large numbers of Python/NumPy objects.
    # gc.collect() alone often does not make process RSS fall
    # because glibc keeps freed arenas.
    #
    # malloc_trim() explicitly returns those free heap pages.
    _release_host_memory()

    return metrics