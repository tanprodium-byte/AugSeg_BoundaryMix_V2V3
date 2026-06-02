#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import init_db, transaction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset OOM failed_final configs to retryable after scheduler policy change."
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    args = parser.parse_args()

    init_db()
    with transaction() as conn:
        rows = conn.execute(
            """
            SELECT config_id
            FROM configs
            WHERE status='failed_final'
              AND COALESCE(last_error, '') ILIKE '%%OOM%%'
              AND config_id NOT IN (
                SELECT config_id
                FROM jobs
                WHERE status='running' AND finished_at IS NULL
              )
            ORDER BY config_id
            """
        ).fetchall()
        config_ids = [row["config_id"] for row in rows]

        mode = "APPLY" if args.apply else "DRY_RUN"
        print(f"{mode} affected_count={len(config_ids)}")
        for config_id in config_ids:
            print(config_id)

        if args.apply and config_ids:
            conn.execute(
                """
                UPDATE configs
                SET status='failed_retryable',
                    worker_id=NULL,
                    attempts=0,
                    last_error='OOM - reset to retryable after scheduler policy change',
                    last_error_class='oom',
                    next_retry_at=NOW(),
                    updated_at=NOW()
                WHERE status='failed_final'
                  AND COALESCE(last_error, '') ILIKE '%%OOM%%'
                  AND config_id = ANY(%s)
                """,
                (config_ids,),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
