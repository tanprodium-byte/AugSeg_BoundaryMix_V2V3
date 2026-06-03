#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.config import DB_PATH, ROOT, default_worker_id, load_hf_settings
from scheduler.db import connect, init_db, mark_done, mark_failed, request_job, set_latest_hf_path

INTERRUPTED_ERROR = "Interrupted by service stop/restart SIGTERM"
DEFAULT_CHILD_TERMINATE_TIMEOUT_SEC = 120.0
ERROR_TAIL_LINES = 200
ERROR_TAIL_CHARS = 8000


class WorkerRuntimeState:
    def __init__(self) -> None:
        self.shutdown_requested = False
        self.shutdown_signal: int | None = None
        self.current_job_id: str | None = None
        self.current_config_id: str | None = None
        self.current_worker_id: str | None = None
        self.current_server_name: str | None = None
        self.current_gpu_id: int | None = None
        self.current_child: subprocess.Popen | None = None
        self.interrupted_reported = False

    def request_shutdown(self, signum: int) -> None:
        self.shutdown_requested = True
        self.shutdown_signal = signum

    def set_current_job(self, job: dict, worker_id: str) -> None:
        self.current_job_id = job["job_id"]
        self.current_config_id = job["config_id"]
        self.current_worker_id = worker_id
        self.current_server_name = job.get("server_name")
        self.current_gpu_id = job.get("gpu_id")
        self.current_child = None
        self.interrupted_reported = False

    def set_child(self, proc: subprocess.Popen | None) -> None:
        self.current_child = proc

    def clear_current_job(self) -> None:
        self.current_job_id = None
        self.current_config_id = None
        self.current_worker_id = None
        self.current_server_name = None
        self.current_gpu_id = None
        self.current_child = None
        self.interrupted_reported = False

    def terminate_child(self) -> None:
        proc = self.current_child
        if proc is not None and proc.poll() is None:
            print(f"SHUTDOWN_TERMINATE_CHILD pid={proc.pid}", flush=True)
            proc.terminate()


RUNTIME_STATE = WorkerRuntimeState()


def install_signal_handlers(state: WorkerRuntimeState) -> None:
    def _handler(signum, _frame) -> None:
        name = signal.Signals(signum).name
        print(f"SHUTDOWN_SIGNAL_RECEIVED signal={name}", flush=True)
        state.request_shutdown(signum)
        state.terminate_child()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def child_terminate_timeout_sec() -> float:
    raw = os.environ.get("AUGSEG_CHILD_TERMINATE_TIMEOUT_SEC")
    if not raw:
        return DEFAULT_CHILD_TERMINATE_TIMEOUT_SEC
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return DEFAULT_CHILD_TERMINATE_TIMEOUT_SEC


def auto_repair_stale_running_enabled() -> bool:
    return os.environ.get("AUGSEG_AUTO_REPAIR_STALE_RUNNING", "0") == "1"


def interruptible_sleep(seconds: int | float, state: WorkerRuntimeState) -> bool:
    deadline = time.monotonic() + max(float(seconds), 0.0)
    while not state.shutdown_requested and time.monotonic() < deadline:
        time.sleep(min(1.0, deadline - time.monotonic()))
    return state.shutdown_requested


def gpu_memory_mib(gpu_id: int) -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    parts = [part.strip() for part in out.strip().splitlines()[0].split(",")]
    return {
        "name": parts[0],
        "total_mib": int(parts[1]),
        "used_mib": int(parts[2]),
        "free_mib": int(parts[3]),
    }


def required_free_vram_mib(gpu_name: str) -> int:
    raw = os.environ.get("AUGSEG_REQUIRED_FREE_VRAM_MB")
    if raw:
        return int(raw)
    name = gpu_name.lower()
    if "5090" in name:
        return 28000
    if "a6000" in name:
        return 30000
    return 28000


def is_oom(log_path: str | None) -> bool:
    if not log_path:
        return False
    try:
        text = open(log_path, "r", errors="replace").read()
    except OSError:
        return False
    lowered = text.lower()
    return "outofmemoryerror" in lowered or "cuda out of memory" in lowered or "out of memory" in lowered


def text_tail(text: str, max_lines: int = ERROR_TAIL_LINES, max_chars: int = ERROR_TAIL_CHARS) -> str:
    lines = text.splitlines()[-max_lines:]
    return "\n".join(lines)[-max_chars:]


def parse_runner_output(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {}


def promote_latest(config_id: str, artifact_dir: str) -> str | None:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub unavailable for latest promote: {exc}") from exc

    settings = load_hf_settings()
    if not settings.get("upload_after_segment", True):
        return None
    repo_id = settings["repo_id"]
    repo_type = settings.get("repo_type", "model")
    base_path = settings.get("base_path", "voc5_single_gpu_gbs8").strip("/")
    latest_prefix = f"{base_path}/{config_id}/latest"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for path in sorted(Path(artifact_dir).glob("*")):
        if path.is_file():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{latest_prefix}/{path.name}",
                repo_id=repo_id,
                repo_type=repo_type,
            )
    return latest_prefix


def post_json(coordinator_url: str, endpoint: str, payload: dict, timeout: int = 30) -> dict:
    url = coordinator_url.rstrip("/") + endpoint
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"ok": False, "error": body}
        parsed.setdefault("http_status", exc.code)
        return parsed
    except URLError as exc:
        return {"ok": False, "error": str(exc)}


def coordinator_request_job(coordinator_url: str, worker_id: str, server_name: str, gpu_id: int) -> dict | None:
    resp = post_json(
        coordinator_url,
        "/request_job",
        {"worker_id": worker_id, "server_name": server_name, "gpu_id": gpu_id},
    )
    if not resp.get("ok"):
        raise RuntimeError(f"request_job failed: {resp}")
    return resp.get("job")


def coordinator_report_done(coordinator_url: str, job_id: str, worker_id: str, latest_hf_path: str | None) -> bool:
    resp = post_json(
        coordinator_url,
        "/report_done",
        {"job_id": job_id, "worker_id": worker_id, "latest_hf_path": latest_hf_path},
    )
    return bool(resp.get("ok"))


def coordinator_report_failed(
    coordinator_url: str,
    job_id: str,
    worker_id: str,
    error: str,
    retryable: bool = True,
) -> bool:
    resp = post_json(
        coordinator_url,
        "/report_failed",
        {"job_id": job_id, "worker_id": worker_id, "error": error, "retryable": retryable},
    )
    return bool(resp.get("ok"))


def coordinator_heartbeat(coordinator_url: str, job_id: str, worker_id: str) -> bool:
    resp = post_json(coordinator_url, "/heartbeat", {"job_id": job_id, "worker_id": worker_id}, timeout=15)
    if not resp.get("ok"):
        print(f"HEARTBEAT_REJECTED {resp}", flush=True)
        return False
    print(f"HEARTBEAT_OK lease_until={resp.get('lease_until')}", flush=True)
    return True


def run_job(
    job: dict,
    coordinator_url: str | None = None,
    worker_id: str | None = None,
    heartbeat_sec: int = 300,
    heartbeat_fn=None,
    runtime_state: WorkerRuntimeState | None = None,
) -> tuple[bool, dict, str]:
    cmd = [
        sys.executable,
        "scheduler/run_train_job.py",
        "--job-id",
        job["job_id"],
        "--config-id",
        job["config_id"],
        "--config-path",
        job["config_path"],
        "--from-epoch",
        str(job["from_epoch"]),
        "--to-epoch",
        str(job["to_epoch"]),
        "--gpu-id",
        str(job["gpu_id"]),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    if runtime_state is not None:
        runtime_state.set_child(proc)
        if runtime_state.shutdown_requested:
            runtime_state.terminate_child()
    lines: list[str] = []
    last_heartbeat = time.monotonic()
    terminate_started: float | None = None
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    while True:
        if runtime_state is not None and runtime_state.shutdown_requested and proc.poll() is None:
            runtime_state.terminate_child()
            if terminate_started is None:
                terminate_started = time.monotonic()
            elif time.monotonic() - terminate_started >= child_terminate_timeout_sec():
                print("SHUTDOWN_CHILD_TIMEOUT reporting interrupted before child exit", flush=True)
                break
        events = selector.select(timeout=1)
        for key, _ in events:
            line = key.fileobj.readline()
            if line:
                lines.append(line)
                print(line, end="", flush=True)
        if proc.poll() is not None:
            rest = proc.stdout.read()
            if rest:
                lines.append(rest)
                print(rest, end="", flush=True)
            break
        if time.monotonic() - last_heartbeat >= heartbeat_sec:
            if heartbeat_fn:
                heartbeat_fn()
            elif coordinator_url and worker_id:
                coordinator_heartbeat(coordinator_url, job["job_id"], worker_id)
            last_heartbeat = time.monotonic()

    stdout = "".join(lines)
    parsed = parse_runner_output(stdout)
    if runtime_state is not None:
        runtime_state.set_child(None)
    if runtime_state is not None and runtime_state.shutdown_requested:
        return False, parsed, INTERRUPTED_ERROR
    if proc.returncode == 0 and parsed.get("status") == "success":
        return True, parsed, stdout
    log = parsed.get("log")
    if is_oom(log):
        reason = "OOM"
    else:
        parts = [f"returncode={proc.returncode}"]
        if log:
            parts.append(f"log_path={log}")
        error_tail = parsed.get("error_tail") or text_tail(stdout)
        if error_tail:
            parts.append("error_tail:\n" + text_tail(str(error_tail)))
        reason = "\n".join(parts)
    return False, parsed, reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--server-name", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--coordinator-db", default=str(DB_PATH))
    parser.add_argument("--coordinator-url", default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--sleep-sec", type=int, default=None)
    parser.add_argument("--heartbeat-sec", type=int, default=300)
    parser.add_argument("--gpu-free-threshold-mib", type=int, default=None)
    args = parser.parse_args()
    install_signal_handlers(RUNTIME_STATE)

    if args.sleep_sec is not None:
        args.sleep_seconds = args.sleep_sec
    if args.loop:
        args.once = False

    worker_id = default_worker_id(args.server_name, args.gpu_id)
    conn = None
    pg_db = None
    if args.backend == "postgres":
        from scheduler import postgres_db as pg_db

        pg_db.init_db()
    elif not args.coordinator_url:
        conn = connect(args.coordinator_db)
        init_db(conn)

    while True:
        if RUNTIME_STATE.shutdown_requested:
            if pg_db:
                pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "stopping", None)
            print("SHUTDOWN_IDLE_EXIT", flush=True)
            return 0

        memory = gpu_memory_mib(args.gpu_id)
        required_free = (
            args.gpu_free_threshold_mib
            if args.gpu_free_threshold_mib is not None
            else required_free_vram_mib(memory["name"])
        )
        memory_detail = (
            f"gpu={args.gpu_id} name={memory['name']} "
            f"memory.total={memory['total_mib']} MiB "
            f"memory.used={memory['used_mib']} MiB "
            f"memory.free={memory['free_mib']} MiB "
            f"required_free={required_free} MiB"
        )
        if memory["free_mib"] < required_free:
            print(f"BUSY {memory_detail}")
            if pg_db:
                pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "busy", memory_detail)
            if args.once:
                return 1
            if interruptible_sleep(args.sleep_seconds, RUNTIME_STATE):
                continue
            continue

        if pg_db:
            running = pg_db.find_running_job_for_worker(worker_id, args.server_name, args.gpu_id)
            if running:
                detail = (
                    f"existing running job job_id={running['job_id']} "
                    f"config_id={running['config_id']} worker_id={running['worker_id']} "
                    f"server_name={running['server_name']} gpu_id={running['gpu_id']}"
                )
                same_slot = (
                    running["worker_id"] == worker_id
                    and running["server_name"] == args.server_name
                    and int(running["gpu_id"]) == int(args.gpu_id)
                )
                if same_slot and auto_repair_stale_running_enabled():
                    print(f"AUTO_REPAIR_STALE_RUNNING {detail}", flush=True)
                    pg_db.report_interrupted(running["job_id"], worker_id, "Interrupted/stale running job reset to retryable")
                    pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "auto_repaired_stale_running", detail)
                    continue
                print(f"BLOCKED_STALE_RUNNING {detail}")
                pg_db.heartbeat(
                    worker_id,
                    args.server_name,
                    args.gpu_id,
                    "blocked_stale_running",
                    "running job exists; manual repair required",
                )
                if args.once:
                    return 1
                if interruptible_sleep(args.sleep_seconds, RUNTIME_STATE):
                    continue
                continue
            pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "free_claiming", None)
            job = pg_db.claim_next_job(worker_id, args.server_name, args.gpu_id)
        elif args.coordinator_url:
            job = coordinator_request_job(args.coordinator_url, worker_id, args.server_name, args.gpu_id)
        else:
            assert conn is not None
            job = request_job(conn, worker_id, args.server_name, args.gpu_id)
        if not job:
            print("NO_JOB")
            if pg_db:
                pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "idle_no_job", None)
            if args.once:
                return 0
            if interruptible_sleep(args.sleep_seconds, RUNTIME_STATE):
                continue
            continue

        print(f"JOB_ASSIGNED {json.dumps(job, sort_keys=True)}")
        RUNTIME_STATE.set_current_job(job, worker_id)
        if pg_db:
            pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "running", job["job_id"])
        ok, parsed, detail = run_job(
            job,
            coordinator_url=None if pg_db else args.coordinator_url,
            worker_id=worker_id,
            heartbeat_sec=args.heartbeat_sec,
            runtime_state=RUNTIME_STATE,
            heartbeat_fn=(
                lambda: (
                    pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "running", job["job_id"]),
                    pg_db.renew_job_lease(job["job_id"], worker_id),
                )
            )
            if pg_db
            else None,
        )
        if RUNTIME_STATE.shutdown_requested:
            if pg_db and not RUNTIME_STATE.interrupted_reported:
                accepted = pg_db.report_interrupted(job["job_id"], worker_id, INTERRUPTED_ERROR)
                RUNTIME_STATE.interrupted_reported = True
                pg_db.heartbeat(
                    worker_id,
                    args.server_name,
                    args.gpu_id,
                    "interrupted" if accepted else "interrupted_rejected",
                    INTERRUPTED_ERROR,
                )
                print("JOB_INTERRUPTED_RETRYABLE" if accepted else "JOB_INTERRUPTED_REJECTED", flush=True)
            elif args.coordinator_url:
                coordinator_report_failed(args.coordinator_url, job["job_id"], worker_id, INTERRUPTED_ERROR, retryable=True)
            else:
                assert conn is not None
                mark_failed(conn, job["job_id"], worker_id, INTERRUPTED_ERROR, retryable=True)
            RUNTIME_STATE.clear_current_job()
            return 0
        if ok:
            try:
                latest_hf_path = promote_latest(job["config_id"], parsed["artifact_dir"])
            except Exception as exc:
                detail = f"latest artifact upload failed: {exc}"
                if pg_db:
                    pg_db.report_failed(job["job_id"], worker_id, detail, retryable=True)
                    pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "failed_retryable", detail)
                elif args.coordinator_url:
                    coordinator_report_failed(args.coordinator_url, job["job_id"], worker_id, detail, retryable=True)
                else:
                    assert conn is not None
                    mark_failed(conn, job["job_id"], worker_id, detail, retryable=True)
                print(f"JOB_FAILED_RETRYABLE {detail}")
                RUNTIME_STATE.clear_current_job()
                if args.once:
                    return 1
                if interruptible_sleep(args.sleep_seconds, RUNTIME_STATE):
                    continue
                continue
            if pg_db:
                accepted = pg_db.report_done(
                    job["job_id"],
                    worker_id,
                    {"job_hf_path": parsed.get("job_hf_path"), "latest_hf_path": latest_hf_path},
                )
                pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "done" if accepted else "done_rejected", job["job_id"])
            elif args.coordinator_url:
                accepted = coordinator_report_done(args.coordinator_url, job["job_id"], worker_id, latest_hf_path)
            else:
                assert conn is not None
                accepted = mark_done(conn, job["job_id"], worker_id, parsed.get("job_hf_path"))
                if latest_hf_path:
                    set_latest_hf_path(conn, job["config_id"], latest_hf_path)
            if not accepted:
                print("JOB_DONE_REJECTED")
                RUNTIME_STATE.clear_current_job()
                return 1
            print("JOB_DONE_ACCEPTED")
            RUNTIME_STATE.clear_current_job()
            if args.once:
                return 0
            continue

        if pg_db:
            pg_db.report_failed(job["job_id"], worker_id, detail, retryable=True)
            pg_db.heartbeat(worker_id, args.server_name, args.gpu_id, "failed_retryable", detail)
        elif args.coordinator_url:
            coordinator_report_failed(args.coordinator_url, job["job_id"], worker_id, detail, retryable=True)
        else:
            assert conn is not None
            mark_failed(conn, job["job_id"], worker_id, detail, retryable=True)
        print(f"JOB_FAILED_RETRYABLE {detail}")
        RUNTIME_STATE.clear_current_job()
        if args.once:
            return 1
        if interruptible_sleep(args.sleep_seconds, RUNTIME_STATE):
            continue


if __name__ == "__main__":
    raise SystemExit(main())
