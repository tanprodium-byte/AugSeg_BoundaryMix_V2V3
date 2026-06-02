#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import init_db, transaction


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

    print("config_id | current_epoch | status | worker_id | attempts | last_error_class | next_retry_at | last_error | updated_at")
    print("-" * 120)
    for row in rows:
        print(
            f"{row['config_id']} | {row['current_epoch']} | {row['status']} | "
            f"{row['worker_id']} | {row['attempts']} | {row['last_error_class']} | "
            f"{row['next_retry_at']} | {row['last_error']} | {row['updated_at']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
