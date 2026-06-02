#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import claim_next_job, init_db, report_failed, transaction

PREFIX = "test_pg_policy_"


def cleanup() -> None:
    with transaction() as conn:
        conn.execute("DELETE FROM jobs WHERE config_id LIKE %s", (PREFIX + "%",))
        conn.execute("DELETE FROM configs WHERE config_id LIKE %s", (PREFIX + "%",))


def insert_config(config_id: str, current_epoch: int = -1000, status: str = "idle", next_retry_sql: str | None = None) -> None:
    with transaction() as conn:
        conn.execute(
            f"""
            INSERT INTO configs (
              config_id, config_path, current_epoch, max_epoch, step_epoch,
              status, worker_id, lease_until, attempts, latest_hf_path,
              last_error, last_error_class, next_retry_at, updated_at
            )
            VALUES (%s, %s, %s, 0, 1, %s, NULL, NULL, 0, NULL, NULL, NULL, {next_retry_sql or 'NULL'}, NOW())
            """,
            (config_id, f"/tmp/{config_id}.yaml", current_epoch, status),
        )


def config_row(config_id: str) -> dict:
    with transaction() as conn:
        row = conn.execute("SELECT * FROM configs WHERE config_id=%s", (config_id,)).fetchone()
    assert row is not None, config_id
    return row


def test_oom_never_failed_final() -> None:
    config_id = PREFIX + "oom_retry"
    insert_config(config_id, current_epoch=-5000)
    worker_id = PREFIX + "worker_oom"
    for idx in range(5):
        job = claim_next_job(worker_id, PREFIX + "server_oom", 0)
        assert job and job["config_id"] == config_id, job
        assert report_failed(job["job_id"], worker_id, "torch.cuda.OutOfMemoryError: CUDA out of memory", max_attempts=3)
        row = config_row(config_id)
        assert row["status"] == "failed_retryable", row
        assert row["last_error"] == "OOM", row
        assert row["last_error_class"] == "oom", row
        assert row["next_retry_at"] is not None, row
        with transaction() as conn:
            conn.execute("UPDATE configs SET next_retry_at=NOW() WHERE config_id=%s", (config_id,))
    row = config_row(config_id)
    assert row["status"] != "failed_final", row


def test_non_oom_can_failed_final() -> None:
    config_id = PREFIX + "non_oom_final"
    insert_config(config_id, current_epoch=-4900)
    worker_id = PREFIX + "worker_non_oom"
    for _ in range(3):
        job = claim_next_job(worker_id, PREFIX + "server_non_oom", 0)
        assert job and job["config_id"] == config_id, job
        assert report_failed(job["job_id"], worker_id, "validation crashed", max_attempts=3)
    row = config_row(config_id)
    assert row["status"] == "failed_final", row
    assert row["attempts"] == 3, row


def test_running_job_guard() -> None:
    running_config_id = PREFIX + "running_guard_existing"
    candidate_config_id = PREFIX + "running_guard_candidate"
    insert_config(running_config_id, current_epoch=-4800, status="running")
    insert_config(candidate_config_id, current_epoch=-4700)
    worker_id = PREFIX + "worker_guard"
    server_name = PREFIX + "server_guard"
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              job_id, config_id, from_epoch, to_epoch, status, worker_id,
              server_name, gpu_id, started_at, finished_at, lease_until, error
            )
            VALUES (%s, %s, 0, 1, 'running', %s, %s, 0, NOW(), NULL, NOW() + INTERVAL '48 hours', NULL)
            """,
            (uuid.uuid4().hex, running_config_id, worker_id, server_name),
        )
    assert claim_next_job(worker_id, PREFIX + "other_server", 1) is None
    assert claim_next_job(PREFIX + "other_worker", server_name, 0) is None


def test_next_retry_gate() -> None:
    future_id = PREFIX + "retry_future"
    past_id = PREFIX + "retry_past"
    insert_config(future_id, current_epoch=-10000, status="failed_retryable", next_retry_sql="NOW() + INTERVAL '1 hour'")
    insert_config(past_id, current_epoch=-9999, status="failed_retryable", next_retry_sql="NOW() - INTERVAL '1 second'")
    job = claim_next_job(PREFIX + "worker_retry", PREFIX + "server_retry", 0)
    assert job and job["config_id"] == past_id, job
    future = config_row(future_id)
    assert future["status"] == "failed_retryable", future


def main() -> int:
    init_db()
    cleanup()
    try:
        for test_fn in (
            test_oom_never_failed_final,
            test_non_oom_can_failed_final,
            test_running_job_guard,
            test_next_retry_gate,
        ):
            cleanup()
            test_fn()
    finally:
        cleanup()
    print("PASS Postgres scheduler policy tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
