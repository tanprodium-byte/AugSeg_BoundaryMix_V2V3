#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REGISTRY="${REGISTRY:-configs/experiment_registry_voc662_12_methods.yaml}"
GPU="${GPU:-0}"
MODE="${MODE:-full}"
MIN_FREE_MB="${MIN_FREE_MB:-12000}"

python tools/run_experiment_suite.py \
  --registry "$REGISTRY" \
  --gpu "$GPU" \
  --mode "$MODE" \
  --min-free-mb "$MIN_FREE_MB" \
  "$@"
