from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import os
import uuid

import psycopg
from psycopg.rows import dict_row


DB_ENV = "AUGSEG_SCHEDULER_DB_URL"


def _db_url() -> str:
    value = os.environ.get(DB_ENV)
    if not value:
        raise RuntimeError(f"{DB_ENV} is required for the Postgres scheduler backend")
    return value


def connect() -> psycopg.Connection:
    return psycopg.connect(_db_url(), row_factory=dict_row)


@contextmanager
def transaction():
    conn = connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with transaction() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS configs (
              config_id TEXT PRIMARY KEY,
              config_path TEXT NOT NULL,
              current_epoch INTEGER NOT NULL DEFAULT 0,
              max_epoch INTEGER NOT NULL DEFAULT 80,
              step_epoch INTEGER NOT NULL DEFAULT 20,
              status TEXT NOT NULL,
              worker_id TEXT,
              lease_until TIMESTAMPTZ,
              attempts INTEGER NOT NULL DEFAULT 0,
              latest_hf_path TEXT,
              last_error TEXT,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              config_id TEXT NOT NULL,
              from_epoch INTEGER NOT NULL,
              to_epoch INTEGER NOT NULL,
              status TEXT NOT NULL,
              worker_id TEXT NOT NULL,
              server_name TEXT NOT NULL,
              gpu_id INTEGER NOT NULL,
              started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              finished_at TIMESTAMPTZ,
              lease_until TIMESTAMPTZ,
              error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_heartbeats (
              worker_id TEXT PRIMARY KEY,
              server_name TEXT,
              gpu_id INTEGER,
              status TEXT,
              last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              detail TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_configs_claim ON configs (status, current_epoch, updated_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_config_id ON jobs (config_id)")


def claim_next_job(worker_id: str, server_name: str, gpu_id: int, lease_minutes: int = 30) -> dict | None:
    init_db()
    with transaction() as conn:
        cfg = conn.execute(
            """
            SELECT *
            FROM configs
            WHERE status IN ('idle','failed_retryable')
              AND current_epoch < max_epoch
            ORDER BY current_epoch ASC, updated_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """
        ).fetchone()
        if cfg is None:
            return None

        from_epoch = int(cfg["current_epoch"])
        to_epoch = min(from_epoch + int(cfg["step_epoch"]), int(cfg["max_epoch"]))
        job_id = uuid.uuid4().hex
        lease_interval = timedelta(minutes=lease_minutes)

        conn.execute(
            """
            UPDATE configs
            SET status='running',
                worker_id=%s,
                lease_until=NOW() + %s,
                updated_at=NOW()
            WHERE config_id=%s
            """,
            (worker_id, lease_interval, cfg["config_id"]),
        )
        conn.execute(
            """
            INSERT INTO jobs (
              job_id, config_id, from_epoch, to_epoch, status, worker_id,
              server_name, gpu_id, started_at, finished_at, lease_until, error
            )
            VALUES (%s, %s, %s, %s, 'running', %s, %s, %s, NOW(), NULL, NOW() + %s, NULL)
            """,
            (
                job_id,
                cfg["config_id"],
                from_epoch,
                to_epoch,
                worker_id,
                server_name,
                gpu_id,
                lease_interval,
            ),
        )
        return {
            "job_id": job_id,
            "config_id": cfg["config_id"],
            "config_path": cfg["config_path"],
            "from_epoch": from_epoch,
            "to_epoch": to_epoch,
            "worker_id": worker_id,
            "server_name": server_name,
            "gpu_id": gpu_id,
        }


def _latest_path_from_artifacts(artifact_paths) -> str | None:
    if artifact_paths is None:
        return None
    if isinstance(artifact_paths, str):
        return artifact_paths
    if isinstance(artifact_paths, dict):
        for key in ("latest_hf_path", "job_hf_path", "hf_path", "artifact_dir"):
            if artifact_paths.get(key):
                return str(artifact_paths[key])
        return None
    if isinstance(artifact_paths, (list, tuple)):
        for value in artifact_paths:
            if value:
                return str(value)
    return None


def report_done(job_id: str, worker_id: str, artifact_paths=None) -> bool:
    init_db()
    with transaction() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,)).fetchone()
        if (
            job is None
            or job["status"] != "running"
            or job["worker_id"] != worker_id
            or (job["lease_until"] is not None and job["lease_until"] <= conn.execute("SELECT NOW() AS now").fetchone()["now"])
        ):
            print(f"ignored stale done job_id={job_id} worker_id={worker_id}", flush=True)
            return False

        cfg = conn.execute("SELECT * FROM configs WHERE config_id=%s FOR UPDATE", (job["config_id"],)).fetchone()
        new_epoch = max(int(cfg["current_epoch"]), int(job["to_epoch"]))
        new_status = "done" if new_epoch >= int(cfg["max_epoch"]) else "idle"
        latest_hf_path = _latest_path_from_artifacts(artifact_paths)

        conn.execute(
            "UPDATE jobs SET status='done', finished_at=NOW(), error=NULL WHERE job_id=%s",
            (job_id,),
        )
        conn.execute(
            """
            UPDATE configs
            SET current_epoch=%s,
                status=%s,
                worker_id=NULL,
                lease_until=NULL,
                latest_hf_path=COALESCE(%s, latest_hf_path),
                last_error=NULL,
                updated_at=NOW()
            WHERE config_id=%s
            """,
            (new_epoch, new_status, latest_hf_path, job["config_id"]),
        )
        return True


def report_failed(
    job_id: str,
    worker_id: str,
    error: str,
    retryable: bool = True,
    max_attempts: int = 3,
) -> bool:
    init_db()
    clipped_error = (error or "")[:2000]
    with transaction() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,)).fetchone()
        if job is None or job["worker_id"] != worker_id:
            return False

        cfg = conn.execute("SELECT attempts FROM configs WHERE config_id=%s FOR UPDATE", (job["config_id"],)).fetchone()
        attempts = int(cfg["attempts"]) + 1
        config_status = "failed_final" if attempts >= max_attempts or not retryable else "failed_retryable"

        conn.execute(
            "UPDATE jobs SET status='failed', finished_at=NOW(), error=%s WHERE job_id=%s",
            (clipped_error, job_id),
        )
        conn.execute(
            """
            UPDATE configs
            SET attempts=%s,
                status=%s,
                worker_id=NULL,
                lease_until=NULL,
                last_error=%s,
                updated_at=NOW()
            WHERE config_id=%s
            """,
            (attempts, config_status, clipped_error, job["config_id"]),
        )
        return True


def heartbeat(worker_id: str, server_name: str, gpu_id: int, status: str, detail: str | None = None) -> None:
    init_db()
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO worker_heartbeats (worker_id, server_name, gpu_id, status, last_seen, detail)
            VALUES (%s, %s, %s, %s, NOW(), %s)
            ON CONFLICT(worker_id) DO UPDATE SET
              server_name=EXCLUDED.server_name,
              gpu_id=EXCLUDED.gpu_id,
              status=EXCLUDED.status,
              last_seen=NOW(),
              detail=EXCLUDED.detail
            """,
            (worker_id, server_name, gpu_id, status, detail),
        )


def renew_job_lease(job_id: str, worker_id: str, lease_minutes: int = 30) -> bool:
    init_db()
    with transaction() as conn:
        lease_interval = timedelta(minutes=lease_minutes)
        job = conn.execute(
            """
            UPDATE jobs
            SET lease_until=NOW() + %s
            WHERE job_id=%s AND worker_id=%s AND status='running'
            RETURNING config_id, lease_until
            """,
            (lease_interval, job_id, worker_id),
        ).fetchone()
        if job is None:
            return False
        conn.execute(
            """
            UPDATE configs
            SET lease_until=%s, updated_at=NOW()
            WHERE config_id=%s AND worker_id=%s AND status='running'
            """,
            (job["lease_until"], job["config_id"], worker_id),
        )
        return True
