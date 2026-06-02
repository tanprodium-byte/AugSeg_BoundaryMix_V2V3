#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="${ROOT}/.codex_smoke/scheduler_lock_test/state.sqlite"

cd "${ROOT}"
mkdir -p "$(dirname "${DB}")"

python scheduler/init_voc5_jobs.py --db "${DB}" --reset

python scheduler/coordinator.py request-job --db "${DB}" --worker-id "fake-a" --server-name "test-a" --gpu-id 0 > "${DB}.a.json" &
pid_a="$!"
python scheduler/coordinator.py request-job --db "${DB}" --worker-id "fake-b" --server-name "test-b" --gpu-id 0 > "${DB}.b.json" &
pid_b="$!"
wait "${pid_a}"
wait "${pid_b}"

python - "${DB}.a.json" "${DB}.b.json" <<'PY'
from pathlib import Path
import json
import sys

jobs = []
for arg in sys.argv[1:]:
    text = Path(arg).read_text()
    obj = json.loads(text)
    if obj:
        jobs.append(obj)

if len(jobs) != 2:
    print(f"FAIL expected 2 jobs, got {len(jobs)}")
    sys.exit(1)

job_ids = {j["job_id"] for j in jobs}
config_ids = {j["config_id"] for j in jobs}
for j in jobs:
    print(f"assigned job_id={j['job_id']} config_id={j['config_id']} worker_id={j['worker_id']}")

if len(job_ids) == 2 and len(config_ids) == 2:
    print("PASS scheduler lock gave different configs")
    sys.exit(0)

print("FAIL overlapping job/config assignment")
sys.exit(1)
PY
