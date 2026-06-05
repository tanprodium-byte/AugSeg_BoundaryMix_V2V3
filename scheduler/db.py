from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import uuid

from scheduler.config import DB_PATH


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS configs (
          config_id TEXT PRIMARY KEY,
          config_path TEXT,
          current_epoch INTEGER,
          max_epoch INTEGER,
          step_epoch INTEGER,
          status TEXT,
          worker_id TEXT,
          lease_until TEXT,
          attempts INTEGER,
          latest_hf_path TEXT,
          last_error TEXT,
          updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY,
          config_id TEXT,
          from_epoch INTEGER,
          to_epoch INTEGER,
          status TEXT,
          worker_id TEXT,
          server_name TEXT,
          gpu_id INTEGER,
          started_at TEXT,
          finished_at TEXT,
          lease_until TEXT,
          error TEXT
        );
        """
    )


@contextmanager
def immediate(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def expire_leases(conn: sqlite3.Connection) -> None:
    now = iso()
    rows = conn.execute(
        """
        SELECT job_id, config_id FROM jobs
        WHERE status='running' AND lease_until IS NOT NULL AND lease_until < ?
        """,
        (now,),
    ).fetchall()
    for row in rows:
        conn.execute(
            "UPDATE jobs SET status='failed_retryable', finished_at=?, error=? WHERE job_id=?",
            (now, "lease expired", row["job_id"]),
        )
        conn.execute(
            """
            UPDATE configs
            SET status='failed_retryable', worker_id=NULL, lease_until=NULL,
                last_error='lease expired', updated_at=?
            WHERE config_id=?
            """,
            (now, row["config_id"]),
        )


def request_job(
    conn: sqlite3.Connection,
    worker_id: str,
    server_name: str,
    gpu_id: int,
    lease_minutes: int = 180,
) -> dict | None:
    with immediate(conn):
        expire_leases(conn)
        cfg = conn.execute(
            """
            SELECT * FROM configs
            WHERE status IN ('idle','failed_retryable') AND current_epoch < max_epoch
            ORDER BY current_epoch ASC, config_id ASC
            LIMIT 1
            """
        ).fetchone()
        if cfg is None:
            return None

        from_epoch = int(cfg["current_epoch"])
        to_epoch = min(int(cfg["max_epoch"]), from_epoch + int(cfg["step_epoch"]))
        lease_until = iso(utcnow() + timedelta(minutes=lease_minutes))
        now = iso()
        job_id = uuid.uuid4().hex

        cur = conn.execute(
            """
            UPDATE configs
            SET status='running', worker_id=?, lease_until=?, attempts=attempts+1,
                last_error=NULL, updated_at=?
            WHERE config_id=? AND status IN ('idle','failed_retryable') AND current_epoch=?
            """,
            (worker_id, lease_until, now, cfg["config_id"], from_epoch),
        )
        if cur.rowcount != 1:
            return None

        conn.execute(
            """
            INSERT INTO jobs (
              job_id, config_id, from_epoch, to_epoch, status, worker_id,
              server_name, gpu_id, started_at, finished_at, lease_until, error
            ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, NULL, ?, NULL)
            """,
            (
                job_id,
                cfg["config_id"],
                from_epoch,
                to_epoch,
                worker_id,
                server_name,
                gpu_id,
                now,
                lease_until,
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
            "lease_until": lease_until,
        }


def mark_done(
    conn: sqlite3.Connection,
    job_id: str,
    worker_id: str,
    latest_hf_path: str | None = None,
) -> bool:
    with immediate(conn):
        job = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if job is None:
            return False
        now_dt = utcnow()
        lease = parse_iso(job["lease_until"])
        if job["status"] != "running" or job["worker_id"] != worker_id or (lease and lease < now_dt):
            return False

        now = iso(now_dt)
        conn.execute(
            "UPDATE jobs SET status='done', finished_at=?, error=NULL WHERE job_id=?",
            (now, job_id),
        )
        cfg = conn.execute("SELECT * FROM configs WHERE config_id=?", (job["config_id"],)).fetchone()
        new_epoch = max(int(cfg["current_epoch"]), int(job["to_epoch"]))
        new_status = "done" if new_epoch >= int(cfg["max_epoch"]) else "idle"
        conn.execute(
            """
            UPDATE configs
            SET current_epoch=?, status=?, worker_id=NULL, lease_until=NULL,
                latest_hf_path=COALESCE(?, latest_hf_path), last_error=NULL, updated_at=?
            WHERE config_id=?
            """,
            (new_epoch, new_status, latest_hf_path, now, job["config_id"]),
        )
        return True


def mark_failed(
    conn: sqlite3.Connection,
    job_id: str,
    worker_id: str,
    error: str,
    retryable: bool = True,
) -> bool:
    with immediate(conn):
        job = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if job is None or job["worker_id"] != worker_id:
            return False
        status = "failed_retryable" if retryable else "failed"
        now = iso()
        conn.execute(
            "UPDATE jobs SET status=?, finished_at=?, error=? WHERE job_id=?",
            (status, now, error[:2000], job_id),
        )
        conn.execute(
            """
            UPDATE configs
            SET status=?, worker_id=NULL, lease_until=NULL, last_error=?, updated_at=?
            WHERE config_id=?
            """,
            (status, error[:2000], now, job["config_id"]),
        )
        return True


def heartbeat(
    conn: sqlite3.Connection,
    job_id: str,
    worker_id: str,
    lease_minutes: int = 180,
) -> str | None:
    with immediate(conn):
        job = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if job is None or job["status"] != "running" or job["worker_id"] != worker_id:
            return None
        lease_until = iso(utcnow() + timedelta(minutes=lease_minutes))
        now = iso()
        conn.execute(
            "UPDATE jobs SET lease_until=? WHERE job_id=?",
            (lease_until, job_id),
        )
        conn.execute(
            """
            UPDATE configs
            SET lease_until=?, updated_at=?
            WHERE config_id=? AND status='running' AND worker_id=?
            """,
            (lease_until, now, job["config_id"], worker_id),
        )
        return lease_until


def set_latest_hf_path(conn: sqlite3.Connection, config_id: str, latest_hf_path: str) -> None:
    conn.execute(
        "UPDATE configs SET latest_hf_path=?, updated_at=? WHERE config_id=?",
        (latest_hf_path, iso(), config_id),
    )
