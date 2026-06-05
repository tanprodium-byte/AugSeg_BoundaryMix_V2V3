#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import evaluate_running_job_blocker, init_db, transaction


def main() -> int:
    init_db()
    with transaction() as conn:
        rows = conn.execute(
            """
            SELECT
              config_id,
              current_epoch,
              status,
              COALESCE(worker_id, '') AS worker_id,
              attempts,
              COALESCE(last_error, '') AS last_error,
              COALESCE(last_error_class, '') AS last_error_class,
              next_retry_at,
              updated_at
            FROM configs
            ORDER BY config_id
            """
        ).fetchall()
        running_rows = conn.execute(
            """
            SELECT
              j.job_id,
              j.config_id,
              j.worker_id,
              j.server_name,
              j.gpu_id,
              j.started_at,
              j.lease_until,
              h.status AS heartbeat_status,
              h.last_seen AS heartbeat_last_seen,
              h.detail AS heartbeat_detail,
              NOW() AS now
            FROM jobs j
            LEFT JOIN worker_heartbeats h ON h.worker_id=j.worker_id
            WHERE j.status='running' AND j.finished_at IS NULL
            ORDER BY j.started_at ASC
            """
        ).fetchall()

    print("config_id | current_epoch | status | worker_id | attempts | last_error_class | next_retry_at | last_error | updated_at")
    print("-" * 120)
    for row in rows:
        print(
            f"{row['config_id']} | {row['current_epoch']} | {row['status']} | "
            f"{row['worker_id']} | {row['attempts']} | {row['last_error_class']} | "
            f"{row['next_retry_at']} | {row['last_error']} | {row['updated_at']}"
        )
    print()
    print("running jobs")
    print("job_id | config_id | worker_id | server_name | gpu_id | started_at | lease_until | heartbeat_status | heartbeat_last_seen | warning")
    print("-" * 140)
    for row in running_rows:
        decision = evaluate_running_job_blocker(
            row,
            row["worker_id"],
            row["server_name"],
            row["gpu_id"],
            now=row["now"],
        )
        warning = "POSSIBLE_STALE_RUNNING" if decision["possible_stale"] else ""
        print(
            f"{row['job_id']} | {row['config_id']} | {row['worker_id']} | "
            f"{row['server_name']} | {row['gpu_id']} | {row['started_at']} | "
            f"{row['lease_until']} | {row['heartbeat_status']} | "
            f"{row['heartbeat_last_seen']} | {warning}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
