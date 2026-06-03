#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import uuid

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.postgres_db import claim_next_job, init_db, report_interrupted, transaction
from scheduler.worker import INTERRUPTED_ERROR, WorkerRuntimeState

PREFIX = "test_signal_stale_"


class FakeChild:
    pid = 424242

    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True


def cleanup() -> None:
    for attempt in range(5):
        try:
            with transaction() as conn:
                conn.execute("DELETE FROM jobs WHERE config_id LIKE %s", (PREFIX + "%",))
                conn.execute("DELETE FROM configs WHERE config_id LIKE %s", (PREFIX + "%",))
            return
        except psycopg.errors.DeadlockDetected:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def insert_config(config_id: str, status: str = "idle", worker_id: str | None = None) -> None:
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO configs (
              config_id, config_path, current_epoch, max_epoch, step_epoch,
              status, worker_id, lease_until, attempts, latest_hf_path,
              last_error, last_error_class, next_retry_at, updated_at
            )
            VALUES (%s, %s, -1000, 0, 1, %s, %s, NOW() + INTERVAL '48 hours', 0, NULL, NULL, NULL, NULL, NOW())
            """,
            (config_id, f"/tmp/{config_id}.yaml", status, worker_id),
        )


def insert_running_job(config_id: str, worker_id: str, server_name: str, gpu_id: int) -> str:
    job_id = uuid.uuid4().hex
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              job_id, config_id, from_epoch, to_epoch, status, worker_id,
              server_name, gpu_id, started_at, finished_at, lease_until, error
            )
            VALUES (%s, %s, 0, 1, 'running', %s, %s, %s, NOW(), NULL, NOW() + INTERVAL '48 hours', NULL)
            """,
            (job_id, config_id, worker_id, server_name, gpu_id),
        )
    return job_id


def config_row(config_id: str) -> dict:
    with transaction() as conn:
        row = conn.execute("SELECT * FROM configs WHERE config_id=%s", (config_id,)).fetchone()
    assert row is not None, config_id
    return row


def job_row(job_id: str) -> dict:
    with transaction() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=%s", (job_id,)).fetchone()
    assert row is not None, job_id
    return row


def test_signal_interrupt_report() -> None:
    config_id = PREFIX + "signal"
    worker_id = PREFIX + "worker_signal"
    server_name = PREFIX + "server_signal"
    insert_config(config_id, status="running", worker_id=worker_id)
    job_id = insert_running_job(config_id, worker_id, server_name, 0)

    state = WorkerRuntimeState()
    state.set_current_job(
        {
            "job_id": job_id,
            "config_id": config_id,
            "server_name": server_name,
            "gpu_id": 0,
        },
        worker_id,
    )
    fake_child = FakeChild()
    state.set_child(fake_child)  # type: ignore[arg-type]
    state.request_shutdown(signal.SIGTERM)
    state.terminate_child()

    assert fake_child.terminated
    assert report_interrupted(job_id, worker_id, INTERRUPTED_ERROR)
    row = config_row(config_id)
    assert row["status"] == "failed_retryable", row
    assert row["attempts"] == 0, row
    assert row["last_error_class"] == "interrupted", row
    assert row["next_retry_at"] is not None, row


def run_repair_script(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "scripts/reset_stale_running_job_to_retryable.py", *args]
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)


def test_repair_dry_run_no_update() -> None:
    config_id = PREFIX + "repair_dry"
    worker_id = PREFIX + "worker_repair_dry"
    insert_config(config_id, status="running", worker_id=worker_id)
    job_id = insert_running_job(config_id, worker_id, PREFIX + "server_repair_dry", 0)

    result = run_repair_script("--config-id", config_id, "--worker-id", worker_id, "--dry-run")
    assert "DRY_RUN affected_count=1" in result.stdout, result.stdout
    assert os.environ["AUGSEG_SCHEDULER_DB_URL"] not in result.stdout
    assert config_row(config_id)["status"] == "running"
    assert job_row(job_id)["status"] == "running"


def test_repair_apply_exact_running_job_only() -> None:
    target_config_id = PREFIX + "repair_apply_target"
    other_config_id = PREFIX + "repair_apply_other"
    worker_id = PREFIX + "worker_repair_apply"
    insert_config(target_config_id, status="running", worker_id=worker_id)
    insert_config(other_config_id, status="running", worker_id=worker_id)
    target_job_id = insert_running_job(target_config_id, worker_id, PREFIX + "server_repair_apply", 0)
    other_job_id = insert_running_job(other_config_id, worker_id, PREFIX + "server_repair_apply", 0)

    result = run_repair_script("--config-id", target_config_id, "--worker-id", worker_id, "--apply")
    assert "updated jobs=1 configs=1" in result.stdout, result.stdout
    target_cfg = config_row(target_config_id)
    other_cfg = config_row(other_config_id)
    assert job_row(target_job_id)["status"] == "failed"
    assert target_cfg["status"] == "failed_retryable", target_cfg
    assert target_cfg["worker_id"] is None, target_cfg
    assert target_cfg["attempts"] == 0, target_cfg
    assert target_cfg["last_error_class"] == "interrupted", target_cfg
    assert job_row(other_job_id)["status"] == "running"
    assert other_cfg["status"] == "running", other_cfg


def test_running_job_guard_blocks_claim() -> None:
    running_config_id = PREFIX + "guard_running"
    candidate_config_id = PREFIX + "guard_candidate"
    worker_id = PREFIX + "worker_guard"
    server_name = PREFIX + "server_guard"
    insert_config(running_config_id, status="running", worker_id=worker_id)
    insert_config(candidate_config_id, status="idle")
    insert_running_job(running_config_id, worker_id, server_name, 0)

    assert claim_next_job(worker_id, PREFIX + "other_server", 1) is None
    assert claim_next_job(PREFIX + "other_worker", server_name, 0) is None


def test_existing_oom_policy_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/test_postgres_scheduler_policy.py"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS Postgres scheduler policy tests" in result.stdout, result.stdout


def main() -> int:
    init_db()
    cleanup()
    try:
        for test_fn in (
            test_signal_interrupt_report,
            test_repair_dry_run_no_update,
            test_repair_apply_exact_running_job_only,
            test_running_job_guard_blocks_claim,
            test_existing_oom_policy_script_passes,
        ):
            cleanup()
            test_fn()
    finally:
        cleanup()
    print("PASS scheduler signal and stale policy tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
