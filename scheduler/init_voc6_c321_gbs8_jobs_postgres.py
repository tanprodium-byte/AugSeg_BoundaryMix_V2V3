#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import init_db, transaction


PROFILE_ROOT = Path("exps/boundary_mix_v2_v3/voc_semi662_c321_gbs8")
CONFIGS = [
    ("baseline_c321_gbs8", "baseline"),
    ("v2_component_weighting_c321_gbs8", "v2_component_weighting"),
    ("v2_v3_best_template_c321_gbs8", "v2_v3_best_template"),
    ("v3_js_bcr_d1_c321_gbs8", "v3_js_bcr_d1"),
    ("v3_js_bcr_d2_c321_gbs8", "v3_js_bcr_d2"),
    ("v3_js_bcr_d3_c321_gbs8", "v3_js_bcr_d3"),
]


def main() -> int:
    init_db()
    with transaction() as conn:
        for config_id, config_name in CONFIGS:
            config_path = PROFILE_ROOT / config_name / "config.yaml"
            if not (ROOT / config_path).is_file():
                raise FileNotFoundError(config_path)

            conn.execute(
                """
                INSERT INTO configs (
                  config_id, config_path, current_epoch, max_epoch, step_epoch,
                  status, worker_id, lease_until, attempts, latest_hf_path,
                  last_error, last_error_class, next_retry_at, updated_at
                )
                VALUES (%s, %s, 0, 80, 20, 'idle', NULL, NULL, 0, NULL, NULL, NULL, NULL, NOW())
                ON CONFLICT(config_id) DO UPDATE SET
                  config_path=EXCLUDED.config_path,
                  max_epoch=EXCLUDED.max_epoch,
                  step_epoch=EXCLUDED.step_epoch,
                  updated_at=NOW()
                """,
                (config_id, str(config_path)),
            )

    print("Postgres VOC6 C321 GBS8 profile initialized/updated without epoch reset:")
    for config_id, config_name in CONFIGS:
        print(f"{config_id} -> {PROFILE_ROOT / config_name / 'config.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
