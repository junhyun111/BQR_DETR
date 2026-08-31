#!/usr/bin/env bash
set -euo pipefail

run_name="${SMOKE_RUN_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}"
root="${OUTPUT_ROOT:-/workspace/artifacts}"
common=(
  --data-root "${COCO_ROOT:-/workspace/data/coco}"
  --output-root "${root}"
  --torch-cache "${TORCH_HOME:-/workspace/torch-cache}"
  --run-name "${run_name}"
  --epochs 1
  --stop-after-epoch 1
  --eval-epochs 1
  --batch-size 2
  --accumulation-steps 4
  --target-global-batch-size 16
  --eval-batch-size 2
  --num-workers "${SMOKE_NUM_WORKERS:-4}"
  --precision fp16
  --train-size 32
  --val-limit 16
  --subset-seed 42
  --seed 42
  --resume auto
  --enc-layers 1
  --dec-layers 1
  --num-queries 30
  --dn-scalar 2
)

torchrun --standalone --nproc_per_node=2 train.py \
  --method baseline --diagnostics-every 0 "${common[@]}"

torchrun --standalone --nproc_per_node=2 train.py \
  --method bqr --diagnostics-every 1 "${common[@]}"

python compare.py \
  "${root}/baseline/seed_42/${run_name}/history.csv" \
  "${root}/bqr/seed_42/${run_name}/history.csv" \
  --baseline-checkpoint "${root}/baseline/seed_42/${run_name}/checkpoints/final.pt" \
  --bqr-checkpoint "${root}/bqr/seed_42/${run_name}/checkpoints/final.pt" \
  --output-dir "${root}/comparison/${run_name}"
