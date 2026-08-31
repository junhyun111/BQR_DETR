#!/usr/bin/env bash
set -euo pipefail

bash scripts/train_baseline.sh
bash scripts/train_bqr.sh
