#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to a checkpoint path}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" dqr-coco/evaluate.py \
  "${CHECKPOINT}" \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --batch-size "${EVAL_BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --precision "${INFERENCE_PRECISION:-fp16}"
