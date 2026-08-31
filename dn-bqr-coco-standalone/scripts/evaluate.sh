#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT:?Set CHECKPOINT to a container checkpoint path}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" evaluate.py \
  "${CHECKPOINT}" \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --output-root "${OUTPUT_ROOT:-/workspace/artifacts}" \
  --torch-cache "${TORCH_HOME:-/workspace/torch-cache}" \
  --batch-size "${EVAL_BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-8}"
