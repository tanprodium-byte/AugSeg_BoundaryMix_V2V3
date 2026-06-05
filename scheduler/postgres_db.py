from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import os
import re
import time
import uuid

import psycopg
from psycopg.rows import dict_row


DB_ENV = "AUGSEG_SCHEDULER_DB_URL"
OOM_COOLDOWN_ENV = "AUGSEG_OOM_COOLDOWN_MINUTES"
JOB_LEASE_HOURS_ENV = "AUGSEG_JOB_LEASE_HOURS"
DEFAULT_JOB_LEASE_HOURS = 48
OOM_PATTERNS = (
    "OOM",
    "out of memory",
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
)
ERROR_CLASS_PATTERNS = (
    ("oom", ("OOM", "out of memory", "CUDA out of memory", "torch.cuda.OutOfMemoryError")),
    ("interrupted", ("Interrupted", "SIGTERM", "SIGINT", "KeyboardInterrupt")),
    ("import_error", ("ImportError", "ModuleNotFoundError", "No module named")),
    ("file_not_found", ("FileNotFoundError", "No such file or directory", "not found")),
    ("permission", ("PermissionError", "Permission denied")),
    ("cuda_unavailable", ("CUDA unavailable", "cuda is not available", "No CUDA GPUs are available")),
    ("config_error", ("KeyError", "yaml", "YAMLError", "config")),
)
_INIT_DONE = False
DB_RETRY_ATTEMPTS = 5
AUTO_REPAIR_STALE_ERROR = "Auto-repaired stale running job after worker restart/lease expiry"
STALE_HEARTBEAT_MINUTES_ENV = "AUGSEG_STALE_RUNNING_HEARTBEAT_MINUTES"
DEFAULT_STALE_HEARTBEAT_MINUTES = 30
RUNNING_HEARTBEAT_STATUS = "running"


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
    global _INIT_DONE
    if _INIT_DONE:
        return
    with transaction() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext('augseg_scheduler_schema'))")
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
        conn.execute("ALTER TABLE configs ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ")
        conn.execute("ALTER TABLE configs ADD COLUMN IF NOT EXISTS last_error_class TEXT")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_running_worker_gpu
            ON jobs (status, worker_id, server_name, gpu_id)
            """
        )
    _INIT_DONE = True


def job_lease_interval() -> timedelta:
    try:
        hours = float(os.environ.get(JOB_LEASE_HOURS_ENV, str(DEFAULT_JOB_LEASE_HOURS)))
    except ValueError:
        hours = DEFAULT_JOB_LEASE_HOURS
    return timedelta(hours=max(hours, 1.0))


def oom_cooldown_interval() -> timedelta:
    try:
        minutes = float(os.environ.get(OOM_COOLDOWN_ENV, "10"))
    except ValueError:
        minutes = 10.0
    return timedelta(minutes=max(minutes, 0.0))


def stale_heartbeat_interval() -> timedelta:
    try:
        minutes = float(os.environ.get(STALE_HEARTBEAT_MINUTES_ENV, str(DEFAULT_STALE_HEARTBEAT_MINUTES)))
    except ValueError:
        minutes = DEFAULT_STALE_HEARTBEAT_MINUTES
    return timedelta(minutes=max(minutes, 1.0))


def is_oom_error(error: str | None) -> bool:
    text = error or ""
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in OOM_PATTERNS)


def clipped_error(error: str | None, limit: int = 8000) -> str:
    return (error or "")[:limit]


def classify_error(error: str | None) -> str:
    text = error or ""
    lowered = text.lower()
    for error_class, patterns in ERROR_CLASS_PATTERNS:
        if any(pattern.lower() in lowered for pattern in patterns):
            return error_class
    if "returncode=" in lowered:
        return "returncode"
    return "returncode"


def short_oom_error(error: str | None) -> str:
    text = clipped_error(error, 500)
    if not text:
        return "OOM"
    text = re.sub(r"\s+", " ", text).strip()
    return text if text.lower() == "oom" else "OOM"


def _running_job_select_sql() -> str:
    return """
            SELECT
              j.job_id,
              j.config_id,
              j.worker_id,
              j.server_name,
              j.gpu_id,
              j.started_at,
              j.lease_until,
              j.error,
              h.status AS heartbeat_status,
              h.last_seen AS heartbeat_last_seen,
              h.detail AS heartbeat_detail
            FROM jobs j
            LEFT JOIN worker_heartbeats h ON h.worker_id=j.worker_id
            WHERE j.status='running'
              AND j.finished_at IS NULL
              AND (j.worker_id=%s OR (j.server_name=%s AND j.gpu_id=%s))
            ORDER BY j.started_at ASC
            LIMIT 1
            """


def find_running_job_for_worker(worker_id: str, server_name: str, gpu_id: int) -> dict | None:
    init_db()
    with transaction() as conn:
        return conn.execute(
            _running_job_select_sql(),
            (worker_id, server_name, gpu_id),
        ).fetchone()


def running_job_detail(row: dict) -> str:
    return (
        f"job_id={row['job_id']} config_id={row['config_id']} "
        f"worker_id={row['worker_id']} server_name={row['server_name']} gpu_id={row['gpu_id']} "
        f"started_at={row.get('started_at')} lease_until={row.get('lease_until')} "
        f"heartbeat_status={row.get('heartbeat_status')} heartbeat_last_seen={row.get('heartbeat_last_seen')}"
    )


def evaluate_running_job_blocker(
    row: dict,
    worker_id: str,
    server_name: str,
    gpu_id: int,
    now=None,
    current_job_id: str | None = None,
    child_alive: bool = False,
) -> dict:
    same_worker = row["worker_id"] == worker_id
    same_slot = row["server_name"] == server_name and int(row["gpu_id"]) == int(gpu_id)
    owns_exact_slot = same_worker and same_slot
    is_current_child = child_alive and current_job_id == row["job_id"]
    now_value = now
    if now_value is None:
        with transaction() as conn:
            now_value = conn.execute("SELECT NOW() AS now").fetchone()["now"]

    lease_until = row.get("lease_until")
    lease_expired = bool(lease_until is not None and lease_until <= now_value)
    hb_status = row.get("heartbeat_status")
    hb_last_seen = row.get("heartbeat_last_seen")
    hb_stale = bool(hb_last_seen is None or hb_last_seen <= now_value - stale_heartbeat_interval())
    hb_not_running = bool(hb_status and hb_status != RUNNING_HEARTBEAT_STATUS)
    possible_stale = bool(lease_expired or hb_stale or hb_not_running)
    stale = bool(owns_exact_slot and not is_current_child and (lease_expired or hb_stale))

    reasons = []
    if not owns_exact_slot:
        reasons.append("running job is not exact current worker/server/gpu ownership")
    if is_current_child:
        reasons.append("worker still has matching current child process")
    if lease_expired:
        reasons.append("lease expired")
    if hb_stale:
        reasons.append("heartbeat missing or stale")
    if hb_not_running:
        reasons.append(f"heartbeat status is {hb_status}")
    if not possible_stale:
        reasons.append("lease and heartbeat are recent")

    return {
        "same_worker": same_worker,
        "same_slot": same_slot,
        "owns_exact_slot": owns_exact_slot,
        "is_current_child": is_current_child,
        "lease_expired": lease_expired,
        "heartbeat_stale": hb_stale,
        "heartbeat_not_running": hb_not_running,
        "possible_stale": possible_stale,
        "stale": stale,
        "worker_status": "blocked_stale_running" if possible_stale else "blocked_running_job",
        "reason": "; ".join(reasons),
    }


def auto_repair_stale_running_job(job_id: str, worker_id: str, server_name: str, gpu_id: int) -> bool:
    init_db()
    with transaction() as conn:
        row = conn.execute(
            """
            SELECT
              j.job_id,
              j.config_id,
              j.worker_id,
              j.server_name,
              j.gpu_id,
              j.started_at,
              j.lease_until,
              h.status AS heartbeat_status,
              h.last_seen AS heartbeat_last_seen,
              h.detail AS heartbeat_detail,
              c.status AS config_status,
              NOW() AS now
            FROM jobs j
            JOIN configs c ON c.config_id=j.config_id
            LEFT JOIN worker_heartbeats h ON h.worker_id=j.worker_id
            WHERE j.job_id=%s
              AND j.status='running'
              AND j.finished_at IS NULL
              AND c.status='running'
            FOR UPDATE OF j, c
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return False
        decision = evaluate_running_job_blocker(row, worker_id, server_name, gpu_id, now=row["now"])
        if not decision["stale"]:
            return False

        conn.execute(
            """
            UPDATE jobs
            SET status='failed',
                finished_at=NOW(),
                error=%s
            WHERE job_id=%s
              AND status='running'
              AND finished_at IS NULL
              AND worker_id=%s
              AND server_name=%s
              AND gpu_id=%s
            """,
            (AUTO_REPAIR_STALE_ERROR, job_id, worker_id, server_name, gpu_id),
        )
        conn.execute(
            """
            UPDATE configs
            SET status='failed_retryable',
                worker_id=NULL,
                lease_until=NULL,
                attempts=0,
                last_error=%s,
                last_error_class='interrupted',
                next_retry_at=NOW(),
                updated_at=NOW()
            WHERE config_id=%s
              AND status='running'
              AND worker_id=%s
            """,
            (AUTO_REPAIR_STALE_ERROR, row["config_id"], worker_id),
        )
        return True


def report_interrupted(job_id: str, worker_id: str, error: str) -> bool:
    init_db()
    error_text = clipped_error(error)
    with transaction() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,)).fetchone()
        if job is None or job["status"] != "running" or job["worker_id"] != worker_id:
            return False

        conn.execute(
            "UPDATE jobs SET status='failed', finished_at=NOW(), error=%s WHERE job_id=%s",
            (error_text, job_id),
        )
        conn.execute(
            """
            UPDATE configs
            SET status='failed_retryable',
                worker_id=NULL,
                lease_until=NULL,
                last_error=%s,
                last_error_class='interrupted',
                next_retry_at=NOW(),
                updated_at=NOW()
            WHERE config_id=%s
            """,
            (error_text, job["config_id"]),
        )
        return True


def claim_next_job(worker_id: str, server_name: str, gpu_id: int, lease_minutes: int = 30) -> dict | None:
    init_db()
    with transaction() as conn:
        running = conn.execute(
            """
            SELECT job_id
            FROM jobs
            WHERE status='running'
              AND finished_at IS NULL
              AND (worker_id=%s OR (server_name=%s AND gpu_id=%s))
            LIMIT 1
            """,
            (worker_id, server_name, gpu_id),
        ).fetchone()
        if running is not None:
            return None

        cfg = conn.execute(
            """
            SELECT *
            FROM configs
            WHERE status IN ('idle','failed_retryable')
              AND current_epoch < max_epoch
              AND (next_retry_at IS NULL OR next_retry_at <= NOW())
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
        lease_interval = job_lease_interval() if lease_minutes == 30 else timedelta(minutes=lease_minutes)

        conn.execute(
            """
            UPDATE configs
            SET status='running',
                worker_id=%s,
                lease_until=NOW() + %s,
                last_error=NULL,
                last_error_class=NULL,
                next_retry_at=NULL,
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
                last_error_class=NULL,
                next_retry_at=NULL,
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
    error_text = clipped_error(error)
    error_class = classify_error(error_text)
    with transaction() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,)).fetchone()
        if job is None or job["worker_id"] != worker_id:
            return False

        cfg = conn.execute("SELECT attempts FROM configs WHERE config_id=%s FOR UPDATE", (job["config_id"],)).fetchone()
        if is_oom_error(error_text):
            # OOM is transient resource pressure, not final experiment failure.
            conn.execute(
                "UPDATE jobs SET status='failed', finished_at=NOW(), error=%s WHERE job_id=%s",
                (short_oom_error(error_text), job_id),
            )
            conn.execute(
                """
                UPDATE configs
                SET status='failed_retryable',
                    worker_id=NULL,
                    lease_until=NULL,
                    last_error='OOM',
                    last_error_class='oom',
                    next_retry_at=NOW() + %s,
                    updated_at=NOW()
                WHERE config_id=%s
                """,
                (oom_cooldown_interval(), job["config_id"]),
            )
            return True

        attempts = int(cfg["attempts"]) + 1
        config_status = "failed_final" if attempts >= max_attempts or not retryable else "failed_retryable"

        conn.execute(
            "UPDATE jobs SET status='failed', finished_at=NOW(), error=%s WHERE job_id=%s",
            (error_text, job_id),
        )
        conn.execute(
            """
            UPDATE configs
            SET attempts=%s,
                status=%s,
                worker_id=NULL,
                lease_until=NULL,
                last_error=%s,
                last_error_class=%s,
                next_retry_at=NULL,
                updated_at=NOW()
            WHERE config_id=%s
            """,
            (attempts, config_status, error_text, error_class, job["config_id"]),
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
        lease_interval = job_lease_interval() if lease_minutes == 30 else timedelta(minutes=lease_minutes)
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


def _retry_deadlocks(fn):
    def wrapped(*args, **kwargs):
        for attempt in range(DB_RETRY_ATTEMPTS):
            try:
                return fn(*args, **kwargs)
            except psycopg.errors.DeadlockDetected:
                if attempt + 1 >= DB_RETRY_ATTEMPTS:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError("unreachable deadlock retry state")

    return wrapped


init_db = _retry_deadlocks(init_db)
find_running_job_for_worker = _retry_deadlocks(find_running_job_for_worker)
auto_repair_stale_running_job = _retry_deadlocks(auto_repair_stale_running_job)
claim_next_job = _retry_deadlocks(claim_next_job)
report_done = _retry_deadlocks(report_done)
report_failed = _retry_deadlocks(report_failed)
report_interrupted = _retry_deadlocks(report_interrupted)
heartbeat = _retry_deadlocks(heartbeat)
renew_job_lease = _retry_deadlocks(renew_job_lease)
