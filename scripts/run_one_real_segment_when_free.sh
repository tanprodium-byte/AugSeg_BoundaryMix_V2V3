#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

scripts/check_gpu_free.sh
python scheduler/init_voc5_jobs.py --no-reset-if-exists
python scheduler/worker.py --server-name supermaster --gpu-id 0 --coordinator-db scheduler/state.sqlite --once
