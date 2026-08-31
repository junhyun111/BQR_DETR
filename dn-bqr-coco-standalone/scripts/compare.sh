#!/usr/bin/env bash
set -euo pipefail

run_name="${RUN_NAME:-poc_10k_20e}"
root="${OUTPUT_ROOT:-/workspace/artifacts}"
seed="${SEED:-42}"
python compare.py \
  "${root}/baseline/seed_${seed}/${run_name}/history.csv" \
  "${root}/bqr/seed_${seed}/${run_name}/history.csv" \
  --baseline-checkpoint "${root}/baseline/seed_${seed}/${run_name}/checkpoints/latest.pt" \
  --bqr-checkpoint "${root}/bqr/seed_${seed}/${run_name}/checkpoints/latest.pt" \
  --output-dir "${root}/comparison/${run_name}"
