#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" train.py \
  --method baseline \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --output-root "${OUTPUT_ROOT:-/workspace/artifacts}" \
  --torch-cache "${TORCH_HOME:-/workspace/torch-cache}" \
  --epochs "${EPOCHS:-12}" \
  --batch-size 2 \
  --accumulation-steps 4 \
  --target-global-batch-size 16 \
  --eval-batch-size "${EVAL_BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --precision fp16 \
  --diagnostics-every 0 \
  --seed "${SEED:-42}" \
  --resume "${RESUME:-auto}"
