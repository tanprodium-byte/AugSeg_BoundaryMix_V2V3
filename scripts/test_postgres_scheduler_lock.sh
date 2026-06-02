#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${AUGSEG_SCHEDULER_DB_URL:-}" ]]; then
  echo "FAIL AUGSEG_SCHEDULER_DB_URL is required"
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  python - <<'PY' || true
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import init_db, transaction

init_db()
with transaction() as conn:
    conn.execute("DELETE FROM jobs WHERE config_id LIKE 'test_pg_lock_%'")
    conn.execute("DELETE FROM configs WHERE config_id LIKE 'test_pg_lock_%'")
PY
  rmdir "${tmp_dir}" 2>/dev/null || true
}
trap cleanup EXIT

python - <<'PY'
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import init_db, transaction

init_db()
with transaction() as conn:
    conn.execute("DELETE FROM jobs WHERE config_id LIKE 'test_pg_lock_%'")
    conn.execute("DELETE FROM configs WHERE config_id LIKE 'test_pg_lock_%'")
    for idx in range(2):
        config_id = f"test_pg_lock_{idx}"
        conn.execute(
            """
            INSERT INTO configs (
              config_id, config_path, current_epoch, max_epoch, step_epoch,
              status, worker_id, lease_until, attempts, latest_hf_path,
              last_error, updated_at
            )
            VALUES (%s, %s, %s, 0, 1, 'idle', NULL, NULL, 0, NULL, NULL, NOW())
            """,
            (config_id, f"/tmp/{config_id}.yaml", -100 + idx),
        )
PY

python - <<'PY' > "${tmp_dir}/a.json" &
from pathlib import Path
import json
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import claim_next_job

job = claim_next_job("test-worker-a", "test-server-a", 0)
print(json.dumps(job or {}, sort_keys=True))
PY
pid_a="$!"

python - <<'PY' > "${tmp_dir}/b.json" &
from pathlib import Path
import json
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import claim_next_job

job = claim_next_job("test-worker-b", "test-server-b", 1)
print(json.dumps(job or {}, sort_keys=True))
PY
pid_b="$!"

wait "${pid_a}"
wait "${pid_b}"

python - "${tmp_dir}/a.json" "${tmp_dir}/b.json" <<'PY'
from pathlib import Path
import json
import sys

jobs = []
for arg in sys.argv[1:]:
    obj = json.loads(Path(arg).read_text())
    if obj:
        jobs.append(obj)

if len(jobs) != 2:
    print(f"FAIL expected 2 jobs, got {len(jobs)}")
    sys.exit(1)

config_ids = {job["config_id"] for job in jobs}
job_ids = {job["job_id"] for job in jobs}
for job in jobs:
    print(f"assigned job_id={job['job_id']} config_id={job['config_id']} worker_id={job['worker_id']}")

if len(config_ids) == 2 and len(job_ids) == 2 and all(cid.startswith("test_pg_lock_") for cid in config_ids):
    print("PASS Postgres scheduler lock gave different test configs")
    sys.exit(0)

print("FAIL overlapping or non-test assignment")
sys.exit(1)
PY
