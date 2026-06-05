#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.config import DB_PATH
from scheduler.db import connect, init_db, expire_leases, immediate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--expire-leases", action="store_true")
    args = parser.parse_args()

    if args.expire_leases:
        conn = connect(args.db)
        init_db(conn)
        with immediate(conn):
            expire_leases(conn)
    else:
        db_path = Path(args.db)
        if not db_path.is_file():
            print(f"DB not found: {db_path}")
            return 1
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT config_id, current_epoch, status, worker_id, attempts, COALESCE(last_error, '') AS last_error
        FROM configs
        ORDER BY config_id
        """
    ).fetchall()
    print("config_id | current_epoch | status | worker_id | attempts | last_error")
    print("-" * 92)
    for r in rows:
        print(
            f"{r['config_id']} | {r['current_epoch']} | {r['status']} | "
            f"{r['worker_id'] or ''} | {r['attempts']} | {r['last_error']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
