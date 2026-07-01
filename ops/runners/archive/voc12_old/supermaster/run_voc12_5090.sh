#!/usr/bin/env bash
set -euo pipefail

cd ~/AugSeg_BoundaryMix_V2V3
source ~/.secrets/augseg_scheduler.env

/home/jupyter-iec2024iot04/.conda/envs/augseg-bm/bin/python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
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
