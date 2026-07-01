#!/usr/bin/env bash
set -euo pipefail

cd /home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3
source /home/jupyter-iec2024iot04/.secrets/augseg_scheduler.env

/home/jupyter-iec2024iot04/.conda/envs/augseg-bm/bin/python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --schedule-mode segments \
  --epoch-targets 20,40,60,80 \
  --suite-name voc662_12_methods_segments_20_40_60_80 \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id supermaster:gpu0 \
  --server-name supermaster \
  --gpu 0 \
  --nproc-per-node 1 \
  --min-free-mb 17000 \
  --poll-sec 60 \
  --launcher python-module \
  --loop \
  --resume \
  --retry-failed \
  --max-retries 1
