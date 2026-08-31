#!/usr/bin/env bash
set -euo pipefail

python prepare.py \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --output-root "${OUTPUT_ROOT:-/workspace/artifacts}" \
  --torch-cache "${TORCH_HOME:-/workspace/torch-cache}" \
  --seed "${SEED:-42}"

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 ACCUMULATION_STEPS=8 bash scripts/train_baseline.sh &
baseline_pid=$!
CUDA_VISIBLE_DEVICES=1 NPROC_PER_NODE=1 ACCUMULATION_STEPS=8 bash scripts/train_bqr.sh &
bqr_pid=$!

wait "${baseline_pid}"
wait "${bqr_pid}"
