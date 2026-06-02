#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.config import REAL_CONFIG_ROOT, VOC5_CONFIG_IDS
from scheduler.postgres_db import init_db, transaction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    init_db()
    with transaction() as conn:
        for config_id in VOC5_CONFIG_IDS:
            config_path = REAL_CONFIG_ROOT / config_id / "config.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(config_path)

            if args.reset:
                conn.execute("DELETE FROM jobs WHERE config_id=%s", (config_id,))
                conn.execute(
                    """
                    INSERT INTO configs (
                      config_id, config_path, current_epoch, max_epoch, step_epoch,
                      status, worker_id, lease_until, attempts, latest_hf_path,
                      last_error, updated_at
                    )
                    VALUES (%s, %s, 0, 80, 20, 'idle', NULL, NULL, 0, NULL, NULL, NOW())
                    ON CONFLICT(config_id) DO UPDATE SET
                      config_path=EXCLUDED.config_path,
                      current_epoch=0,
                      max_epoch=80,
                      step_epoch=20,
                      status='idle',
                      worker_id=NULL,
                      lease_until=NULL,
                      attempts=0,
                      latest_hf_path=NULL,
                      last_error=NULL,
                      updated_at=NOW()
                    """,
                    (config_id, str(config_path)),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO configs (
                      config_id, config_path, current_epoch, max_epoch, step_epoch,
                      status, worker_id, lease_until, attempts, latest_hf_path,
                      last_error, updated_at
                    )
                    VALUES (%s, %s, 0, 80, 20, 'idle', NULL, NULL, 0, NULL, NULL, NOW())
                    ON CONFLICT(config_id) DO UPDATE SET
                      config_path=EXCLUDED.config_path,
                      max_epoch=EXCLUDED.max_epoch,
                      step_epoch=EXCLUDED.step_epoch,
                      updated_at=NOW()
                    """,
                    (config_id, str(config_path)),
                )

    action = "reset" if args.reset else "initialized/updated without epoch reset"
    print(f"Postgres VOC5 jobs {action}: {', '.join(VOC5_CONFIG_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
