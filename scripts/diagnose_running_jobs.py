#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import evaluate_running_job_blocker, transaction


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only diagnosis for Postgres scheduler running jobs.")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    with transaction() as conn:
        rows = conn.execute(
            """
            SELECT
              j.job_id,
              j.config_id,
              j.worker_id,
              j.server_name,
              j.gpu_id,
              j.started_at,
              j.lease_until,
              j.error,
              h.status AS heartbeat_status,
              h.last_seen AS heartbeat_last_seen,
              h.detail AS heartbeat_detail,
              NOW() AS now
            FROM jobs j
            LEFT JOIN worker_heartbeats h ON h.worker_id=j.worker_id
            WHERE j.status='running' AND j.finished_at IS NULL
            ORDER BY j.started_at ASC
            LIMIT %s
            """,
            (args.limit,),
        ).fetchall()

    print(f"running_count={len(rows)}")
    for row in rows:
        decision = evaluate_running_job_blocker(
            row,
            row["worker_id"],
            row["server_name"],
            row["gpu_id"],
            now=row["now"],
        )
        warning = "POSSIBLE_STALE_RUNNING " if decision["possible_stale"] else ""
        print(
            f"{warning}job_id={row['job_id']} config_id={row['config_id']} "
            f"worker_id={row['worker_id']} server_name={row['server_name']} gpu_id={row['gpu_id']} "
            f"started_at={row['started_at']} lease_until={row['lease_until']} "
            f"heartbeat_status={row['heartbeat_status']} heartbeat_last_seen={row['heartbeat_last_seen']} "
            f"reason={decision['reason']} heartbeat_detail={row['heartbeat_detail']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
