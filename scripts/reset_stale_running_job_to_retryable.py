#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import AUTO_REPAIR_STALE_ERROR, init_db, transaction

RESET_ERROR = AUTO_REPAIR_STALE_ERROR


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset stale running scheduler rows to retryable. Default is dry-run."
    )
    parser.add_argument("--config-id", default=None, help="Limit to a single config_id.")
    parser.add_argument("--worker-id", default=None, help="Limit to worker_id.")
    parser.add_argument("--job-id", default=None, help="Optional job_id filter.")
    parser.add_argument("--dry-run", action="store_true", help="List affected rows without updating. This is the default.")
    parser.add_argument("--apply", action="store_true", help="Apply the reset. Requires --config-id and --worker-id.")
    args = parser.parse_args()

    if args.apply and (not args.config_id or not args.worker_id):
        print("REFUSED apply_requires_config_id_and_worker_id")
        return 2

    init_db()
    where = ["j.status='running'", "j.finished_at IS NULL", "c.status='running'"]
    params: list[str] = []
    if args.config_id:
        where.append("j.config_id=%s")
        params.append(args.config_id)
    if args.worker_id:
        where.append("j.worker_id=%s")
        params.append(args.worker_id)
    if args.job_id:
        where.append("j.job_id=%s")
        params.append(args.job_id)
    where_sql = " AND ".join(where)

    with transaction() as conn:
        rows = conn.execute(
            f"""
            SELECT
              j.job_id,
              j.config_id,
              j.worker_id,
              j.server_name,
              j.gpu_id,
              j.started_at,
              j.lease_until,
              j.error
            FROM jobs j
            JOIN configs c ON c.config_id=j.config_id
            WHERE {where_sql}
            ORDER BY j.started_at ASC
            """,
            params,
        ).fetchall()

        mode = "APPLY" if args.apply else "DRY_RUN"
        print(f"{mode} affected_count={len(rows)}")
        for row in rows:
            print(
                (
                    "job_id={job_id} config_id={config_id} worker_id={worker_id} "
                    "server_name={server_name} gpu_id={gpu_id} started_at={started_at} "
                    "lease_until={lease_until} error={error}"
                ).format(
                    **row
                )
            )

        if not args.apply or not rows:
            return 0
        if len(rows) != 1:
            print("REFUSED apply_requires_exactly_one_running_row")
            return 2

        job_ids = [row["job_id"] for row in rows]
        config_ids = [row["config_id"] for row in rows]
        jobs_result = conn.execute(
            """
            UPDATE jobs
            SET status='failed',
                finished_at=NOW(),
                error=%s
            WHERE status='running'
              AND finished_at IS NULL
              AND job_id = ANY(%s)
              AND worker_id=%s
            """,
            (RESET_ERROR, job_ids, args.worker_id),
        )
        configs_result = conn.execute(
            """
            UPDATE configs
            SET status='failed_retryable',
                worker_id=NULL,
                lease_until=NULL,
                attempts=0,
                last_error=%s,
                last_error_class='interrupted',
                next_retry_at=NOW(),
                updated_at=NOW()
            WHERE status='running'
              AND config_id = ANY(%s)
              AND worker_id=%s
            """,
            (RESET_ERROR, config_ids, args.worker_id),
        )
        print(f"updated jobs={jobs_result.rowcount} configs={configs_result.rowcount}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
