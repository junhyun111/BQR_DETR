#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" train.py \
  --method bqr \
  --data-root "${COCO_ROOT:-/workspace/data/coco}" \
  --output-root "${OUTPUT_ROOT:-/workspace/artifacts}" \
  --torch-cache "${TORCH_HOME:-/workspace/torch-cache}" \
  --run-name "${RUN_NAME:-poc_10k_20e}" \
  --epochs "${EPOCHS:-20}" \
  --stop-after-epoch "${STOP_AFTER_EPOCH:-10}" \
  --eval-epochs "${EVAL_EPOCHS:-5,10,15,20}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --accumulation-steps "${ACCUMULATION_STEPS:-4}" \
  --target-global-batch-size "${TARGET_GLOBAL_BATCH_SIZE:-16}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-8}" --persistent-workers \
  --precision "${PRECISION:-fp16}" \
  --train-size "${TRAIN_SIZE:-10000}" --subset-seed "${SUBSET_SEED:-42}" \
  --diagnostics-every "${DIAGNOSTICS_EVERY:-100}" --seed "${SEED:-42}" \
  --resume "${RESUME:-auto}"
