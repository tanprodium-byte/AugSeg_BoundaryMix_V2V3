#!/usr/bin/env bash
set -euo pipefail

cd "/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3"
source "/home/jupyter-iec2024iot04/.secrets/augseg_scheduler.env"

PY="/home/jupyter-iec2024iot04/.conda/envs/augseg-bm/bin/python"
if [ ! -x "$PY" ]; then PY="/home/jupyter-iec2024iot04/.conda/envs/semseg-ael/bin/python"; fi
if [ ! -x "$PY" ]; then PY="python"; fi

"$PY" tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml \
  --mode full \
  --schedule-mode segments \
  --epoch-targets 20,40,60,80 \
  --suite-name voc662_8_sc_methods_official_c3_20_40_60_80 \
  --log-dir runs/suite_logs/voc662_8_sc_methods_official_c3_20_40_60_80 \
  --status-dir runs/suite_status/voc662_8_sc_methods_official_c3_20_40_60_80 \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id "supermaster:gpu0" \
  --server-name "supermaster" \
  --gpu 0 \
  --nproc-per-node 1 \
  --min-free-mb 17000 \
  --poll-sec 60 \
  --sleep-sec 30 \
  --launcher python-module \
  --loop \
  --resume \
  --retry-failed \
  --max-retries 1
