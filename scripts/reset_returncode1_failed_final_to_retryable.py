#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import init_db, transaction

RESET_ERROR = "returncode=1 reset after bootstrap fix"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset returncode=1 failed_final configs to retryable after fixing the bootstrap root cause."
    )
    parser.add_argument("--dry-run", action="store_true", help="List affected configs without updating. This is the default.")
    parser.add_argument("--apply", action="store_true", help="Apply the reset. Default is dry-run.")
    args = parser.parse_args()

    if args.apply and args.dry_run:
        print("REFUSED conflicting_flags: use either --dry-run or --apply")
        return 2

    init_db()
    with transaction() as conn:
        rows = conn.execute(
            """
            SELECT config_id, attempts, last_error, updated_at
            FROM configs c
            WHERE c.status='failed_final'
              AND COALESCE(c.last_error, '') ILIKE '%%returncode=1%%'
              AND NOT EXISTS (
                SELECT 1
                FROM jobs j
                WHERE j.config_id=c.config_id
                  AND j.status='running'
                  AND j.finished_at IS NULL
              )
            ORDER BY config_id
            """
        ).fetchall()

        mode = "APPLY" if args.apply else "DRY_RUN"
        print(f"{mode} affected_count={len(rows)}")
        for row in rows:
            print(
                "config_id={config_id} attempts={attempts} updated_at={updated_at} last_error={last_error}".format(
                    **row
                )
            )

        if args.apply and rows:
            config_ids = [row["config_id"] for row in rows]
            result = conn.execute(
                """
                UPDATE configs
                SET status='failed_retryable',
                    worker_id=NULL,
                    attempts=0,
                    last_error=%s,
                    last_error_class='returncode',
                    next_retry_at=NOW(),
                    updated_at=NOW()
                WHERE status='failed_final'
                  AND COALESCE(last_error, '') ILIKE '%%returncode=1%%'
                  AND config_id = ANY(%s)
                """,
                (RESET_ERROR, config_ids),
            )
            print(f"updated configs={result.rowcount}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
