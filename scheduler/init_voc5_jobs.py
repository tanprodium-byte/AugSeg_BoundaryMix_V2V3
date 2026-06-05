#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.config import DB_PATH, REAL_CONFIG_ROOT, VOC5_CONFIG_IDS
from scheduler.db import connect, init_db, iso


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-reset-if-exists", action="store_true")
    args = parser.parse_args()

    conn = connect(args.db)
    init_db(conn)
    existing = conn.execute("SELECT COUNT(*) AS n FROM configs").fetchone()["n"]
    if args.reset and not args.no_reset_if_exists:
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM configs")
        existing = 0
    if existing and args.no_reset_if_exists:
        print(f"DB already has {existing} configs; not resetting.")
        return 0

    now = iso()
    for config_id in VOC5_CONFIG_IDS:
        config_path = REAL_CONFIG_ROOT / config_id / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        conn.execute(
            """
            INSERT INTO configs (
              config_id, config_path, current_epoch, max_epoch, step_epoch,
              status, worker_id, lease_until, attempts, latest_hf_path,
              last_error, updated_at
            ) VALUES (?, ?, 0, 80, 20, 'idle', NULL, NULL, 0, NULL, NULL, ?)
            ON CONFLICT(config_id) DO UPDATE SET
              config_path=excluded.config_path,
              max_epoch=excluded.max_epoch,
              step_epoch=excluded.step_epoch,
              updated_at=excluded.updated_at
            """,
            (config_id, str(config_path), now),
        )
    print(f"Initialized {len(VOC5_CONFIG_IDS)} configs in {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
