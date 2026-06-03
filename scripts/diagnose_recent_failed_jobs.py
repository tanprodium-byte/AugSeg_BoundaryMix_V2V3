#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import init_db, transaction


def tail_lines(path: Path, limit: int = 80) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-limit:])
    except OSError as exc:
        return f"unable to read log tail: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Print recent failed Postgres scheduler jobs and local log tails.")
    parser.add_argument("--limit", type=int, default=20, help="Number of recent jobs to inspect.")
    args = parser.parse_args()

    init_db()
    with transaction() as conn:
        rows = conn.execute(
            """
            SELECT
              j.job_id,
              j.config_id,
              j.worker_id,
              j.server_name,
              j.started_at,
              j.finished_at,
              COALESCE(j.error, '') AS error
            FROM jobs j
            WHERE j.status='failed'
            ORDER BY j.started_at DESC
            LIMIT %s
            """,
            (args.limit,),
        ).fetchall()

    for row in rows:
        print(
            "job_id={job_id} config_id={config_id} worker_id={worker_id} "
            "server_name={server_name} started_at={started_at} finished_at={finished_at}".format(**row)
        )
        print(f"error={row['error']}")
        log_path = ROOT / ".scheduler_runs" / "logs" / f"{row['job_id']}.log"
        if log_path.is_file():
            print(f"log_path={log_path}")
            print("--- log tail 80 ---")
            print(tail_lines(log_path, 80))
            print("--- end log tail ---")
        else:
            print(f"log_path_missing={log_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
