#!/usr/bin/env python
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.config import DB_PATH
from scheduler.db import connect, expire_leases, heartbeat, immediate, init_db, mark_done, mark_failed, request_job


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class CoordinatorHandler(BaseHTTPRequestHandler):
    server_version = "AugSegCoordinator/1.0"

    @property
    def db_path(self) -> Path:
        return self.server.db_path  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def db(self):
        conn = connect(self.db_path)
        init_db(conn)
        return conn

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            return json_response(self, 200, {"ok": True})
        if path == "/status":
            return self.handle_status()
        return json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
        except Exception as exc:
            return json_response(self, 400, {"ok": False, "error": f"invalid json: {exc}"})

        try:
            if path == "/request_job":
                return self.handle_request_job(payload)
            if path == "/heartbeat":
                return self.handle_heartbeat(payload)
            if path == "/report_done":
                return self.handle_report_done(payload)
            if path == "/report_failed":
                return self.handle_report_failed(payload)
        except Exception as exc:
            return json_response(self, 500, {"ok": False, "error": str(exc)})
        return json_response(self, 404, {"ok": False, "error": "not found"})

    def handle_status(self) -> None:
        conn = self.db()
        with immediate(conn):
            expire_leases(conn)
        rows = conn.execute(
            """
            SELECT config_id, config_path, current_epoch, max_epoch, step_epoch,
                   status, worker_id, lease_until, attempts, latest_hf_path,
                   last_error, updated_at
            FROM configs
            ORDER BY config_id
            """
        ).fetchall()
        jobs = conn.execute(
            """
            SELECT job_id, config_id, from_epoch, to_epoch, status, worker_id,
                   server_name, gpu_id, started_at, finished_at, lease_until, error
            FROM jobs
            ORDER BY started_at DESC
            LIMIT 20
            """
        ).fetchall()
        return json_response(
            self,
            200,
            {
                "ok": True,
                "configs": [dict(r) for r in rows],
                "recent_jobs": [dict(r) for r in jobs],
            },
        )

    def handle_request_job(self, payload: dict) -> None:
        worker_id = str(payload.get("worker_id") or "")
        server_name = str(payload.get("server_name") or "")
        gpu_id = int(payload.get("gpu_id", 0))
        if not worker_id or not server_name:
            return json_response(self, 400, {"ok": False, "error": "worker_id and server_name are required"})
        conn = self.db()
        job = request_job(conn, worker_id, server_name, gpu_id)
        return json_response(self, 200, {"ok": True, "job": job})

    def handle_heartbeat(self, payload: dict) -> None:
        job_id = str(payload.get("job_id") or "")
        worker_id = str(payload.get("worker_id") or "")
        if not job_id or not worker_id:
            return json_response(self, 400, {"ok": False, "error": "job_id and worker_id are required"})
        conn = self.db()
        lease_until = heartbeat(conn, job_id, worker_id)
        if lease_until is None:
            return json_response(self, 409, {"ok": False, "error": "heartbeat rejected"})
        return json_response(self, 200, {"ok": True, "lease_until": lease_until})

    def handle_report_done(self, payload: dict) -> None:
        job_id = str(payload.get("job_id") or "")
        worker_id = str(payload.get("worker_id") or "")
        latest_hf_path = payload.get("latest_hf_path")
        if not job_id or not worker_id:
            return json_response(self, 400, {"ok": False, "error": "job_id and worker_id are required"})
        conn = self.db()
        ok = mark_done(conn, job_id, worker_id, latest_hf_path)
        return json_response(self, 200 if ok else 409, {"ok": ok})

    def handle_report_failed(self, payload: dict) -> None:
        job_id = str(payload.get("job_id") or "")
        worker_id = str(payload.get("worker_id") or "")
        error = str(payload.get("error") or "worker failed")
        retryable = bool(payload.get("retryable", True))
        if not job_id or not worker_id:
            return json_response(self, 400, {"ok": False, "error": "job_id and worker_id are required"})
        conn = self.db()
        ok = mark_failed(conn, job_id, worker_id, error, retryable=retryable)
        return json_response(self, 200 if ok else 409, {"ok": ok})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = connect(db_path)
    init_db(conn)
    conn.close()

    server = ThreadingHTTPServer((args.host, args.port), CoordinatorHandler)
    server.db_path = db_path  # type: ignore[attr-defined]
    print(f"coordinator_api listening on http://{args.host}:{args.port} db={db_path}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
