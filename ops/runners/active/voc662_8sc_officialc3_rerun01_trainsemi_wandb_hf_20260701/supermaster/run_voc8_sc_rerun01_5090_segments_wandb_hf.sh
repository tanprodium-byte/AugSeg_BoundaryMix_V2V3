#!/usr/bin/env bash
set -euo pipefail

cd "/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3"
source "/home/jupyter-iec2024iot04/.secrets/augseg_scheduler.env"

PY="/home/jupyter-iec2024iot04/.conda/envs/augseg-bm/bin/python"
if [ ! -x "$PY" ]; then PY="/home/jupyter-iec2024iot04/.conda/envs/semseg-ael/bin/python"; fi
if [ ! -x "$PY" ]; then PY="python"; fi

SUITE="voc662_8sc_officialc3_rerun01_trainsemi_wandb_hf_20260701"
REGISTRY="configs/experiment_registry_${SUITE}.yaml"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" tools/run_experiment_suite.py \
  --registry "${REGISTRY}" \
  --mode full \
  --schedule-mode segments \
  --epoch-targets 20,40,60,80 \
  --suite-name "${SUITE}" \
  --log-dir "runs/suite_logs/${SUITE}" \
  --status-dir "runs/suite_status/${SUITE}" \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --init-queue \
  --worker-id "supermaster:gpu0:rerun01" \
  --server-name "supermaster" \
  --gpu 0 \
  --nproc-per-node 1 \
  --master-port 29631 \
  --min-free-mb 9000 \
  --poll-sec 60 \
  --sleep-sec 30 \
  --heartbeat-sec 30 \
  --max-stale-minutes 5 \
  --launcher python-module \
  --loop \
  --resume \
  --retry-failed \
  --max-retries 2
