#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node=2 train.py \
  --method bqr_dn_v2 \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --output-root "${OUTPUT_ROOT:-/workspace/artifacts}" \
  --torch-cache "${TORCH_HOME:-/workspace/torch-cache}" \
  --run-name smoke_2gpu_fp16 \
  --epochs 1 \
  --batch-size 2 \
  --accumulation-steps 4 \
  --target-global-batch-size 16 \
  --eval-batch-size 2 \
  --num-workers "${SMOKE_NUM_WORKERS:-4}" \
  --precision fp16 \
  --diagnostics-every 1 \
  --train-limit 32 \
  --val-limit 16 \
  --seed 42 \
  --resume auto
