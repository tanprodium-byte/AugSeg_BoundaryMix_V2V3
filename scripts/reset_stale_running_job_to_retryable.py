#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import init_db, transaction

RESET_ERROR = "Interrupted/stale running job reset to retryable"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset stale running scheduler rows to retryable. Default is dry-run."
    )
    parser.add_argument("--config-id", default=None, help="Limit to a single config_id.")
    parser.add_argument("--worker-id", default=None, help="Optional worker_id filter.")
    parser.add_argument("--dry-run", action="store_true", help="List affected rows without updating. This is the default.")
    parser.add_argument("--apply", action="store_true", help="Apply the reset. Requires --config-id.")
    args = parser.parse_args()

    if args.apply and not args.config_id:
        print("REFUSED apply_without_config_id: pass --config-id to avoid resetting all running jobs")
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
    where_sql = " AND ".join(where)

    with transaction() as conn:
        rows = conn.execute(
            f"""
            SELECT
              j.job_id,
              j.config_id,
              j.worker_id,
              j.started_at
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
                "job_id={job_id} config_id={config_id} worker_id={worker_id} started_at={started_at}".format(
                    **row
                )
            )

        if not args.apply or not rows:
            return 0

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
            """,
            (RESET_ERROR, job_ids),
        )
        configs_result = conn.execute(
            """
            UPDATE configs
            SET status='failed_retryable',
                worker_id=NULL,
                attempts=0,
                last_error=%s,
                last_error_class='interrupted',
                next_retry_at=NOW(),
                updated_at=NOW()
            WHERE status='running'
              AND config_id = ANY(%s)
            """,
            (RESET_ERROR, config_ids),
        )
        print(f"updated jobs={jobs_result.rowcount} configs={configs_result.rowcount}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
