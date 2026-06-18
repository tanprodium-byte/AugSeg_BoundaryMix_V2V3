#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs/experiment_registry_voc662_12_methods.yaml"
LOCK_ROOT = ROOT / "runs/locks"
SMOKE_CONFIG_ROOT = ROOT / "tmp/suite_smoke_configs"
SMOKE_RUN_ROOT = ROOT / "tmp/suite_smoke_runs"
SEGMENT_CONFIG_ROOT = ROOT / "runs/suite_temp/segment_configs"
ERROR_PATTERNS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "Traceback",
    "FloatingPointError",
    "non-finite",
    "SignalException",
    "RuntimeError",
    "KeyboardInterrupt",
)
SMOKE_PROGRESS_PATTERNS = (
    "Epoch",
    "epoch",
    "Iter",
    "iter",
    "Train",
    "train",
    "loss",
)
FINAL_STATUSES = {
    "success",
    "failed",
    "failed_oom",
    "failed_traceback",
    "timeout",
    "timeout_smoke_ok",
    "skipped",
    "skipped_success",
    "gpu_busy",
    "failed_stale",
}
RETRYABLE_STATUSES = {
    "failed",
    "failed_oom",
    "failed_traceback",
    "timeout",
    "gpu_busy",
    "failed_stale",
}
EXPECTED_METHOD_COUNT = 12
DEFAULT_SEGMENT_SUITE_NAME = "voc662_12_methods_segments_20_40_60_80"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def load_registry(path: Path) -> dict[str, Any]:
    registry = load_yaml(path)
    methods = registry.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("registry.methods must be a non-empty list")
    seen: set[str] = set()
    for method in methods:
        if not isinstance(method, dict):
            raise ValueError("Each registry method must be a mapping")
        for key in ("name", "group", "config"):
            if not method.get(key):
                raise ValueError(f"Registry method missing {key}: {method}")
        if method["name"] in seen:
            raise ValueError(f"Duplicate method name in registry: {method['name']}")
        seen.add(method["name"])
        config_path = resolve_path(method["config"])
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found for {method['name']}: {method['config']}")
    return registry


def effective_registry(registry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    registry = copy.deepcopy(registry)
    if args.suite_name:
        registry["suite_name"] = args.suite_name
    elif args.schedule_mode == "segments":
        registry["suite_name"] = DEFAULT_SEGMENT_SUITE_NAME
    return registry


def parse_epoch_targets(value: str) -> list[int]:
    targets = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not targets:
        raise ValueError("--epoch-targets must contain at least one epoch")
    if targets != sorted(set(targets)):
        raise ValueError("--epoch-targets must be strictly increasing unique integers")
    if any(t <= 0 for t in targets):
        raise ValueError("--epoch-targets must be positive")
    if len(targets) != 4:
        raise ValueError("segment scheduler currently expects exactly four epoch targets, e.g. 20,40,60,80")
    return targets


def next_epoch_target(current_epoch: int, targets: list[int]) -> int | None:
    for target in targets:
        if int(current_epoch) < int(target):
            return int(target)
    return None


def max_config_epochs(cfg: dict[str, Any]) -> int:
    try:
        return int(cfg["trainer"]["epochs"])
    except Exception as exc:
        raise ValueError("config must define trainer.epochs") from exc


def crop_size(cfg: dict[str, Any]) -> list[int] | None:
    size = cfg.get("dataset", {}).get("train", {}).get("crop", {}).get("size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return [int(size[0]), int(size[1])]
    return None


def train_batch_size(cfg: dict[str, Any]) -> int | None:
    value = cfg.get("dataset", {}).get("train", {}).get("batch_size")
    if value is None:
        return None
    return int(value)


def validate_full_config(
    method: dict[str, Any],
    registry: dict[str, Any],
    nproc_per_node: int,
) -> tuple[list[int], int]:
    config_path = resolve_path(method["config"])
    cfg = load_yaml(config_path)
    expected_crop = [int(v) for v in registry.get("crop_size", [321, 321])]
    expected_gbs = int(registry.get("global_batch_size", 8))
    actual_crop = crop_size(cfg)
    if actual_crop != expected_crop:
        raise ValueError(
            f"{method['name']} has crop={actual_crop}; expected {expected_crop} for full mode"
        )
    batch = train_batch_size(cfg)
    if batch is None:
        raise ValueError(f"{method['name']} missing dataset.train.batch_size")
    global_batch = batch * int(nproc_per_node)
    if global_batch != expected_gbs:
        raise ValueError(
            f"{method['name']} has batch_size={batch}, nproc_per_node={nproc_per_node}, "
            f"global_batch_size={global_batch}; expected {expected_gbs}"
        )
    return actual_crop, global_batch


def select_methods(
    registry: dict[str, Any],
    only: str | None = None,
    skip: str | None = None,
    group: str | None = None,
) -> list[dict[str, Any]]:
    methods = list(registry["methods"])
    known_names = {m["name"] for m in methods}
    if only:
        wanted = [x.strip() for x in only.split(",") if x.strip()]
        missing = sorted(set(wanted) - known_names)
        if missing:
            raise ValueError(f"Unknown --only method(s): {', '.join(missing)}")
        methods = [m for m in methods if m["name"] in set(wanted)]
    if skip:
        skipped = {x.strip() for x in skip.split(",") if x.strip()}
        missing = sorted(skipped - known_names)
        if missing:
            raise ValueError(f"Unknown --skip method(s): {', '.join(missing)}")
        methods = [m for m in methods if m["name"] not in skipped]
    if group:
        groups = {x.strip() for x in group.split(",") if x.strip()}
        known_groups = {m["group"] for m in registry["methods"]}
        missing_groups = sorted(groups - known_groups)
        if missing_groups:
            raise ValueError(f"Unknown --group value(s): {', '.join(missing_groups)}")
        methods = [m for m in methods if m["group"] in groups]
    return methods


def torchrun_matches_python_env(torchrun_path: str) -> bool:
    python_dir = Path(sys.executable).resolve().parent
    try:
        return Path(torchrun_path).resolve().parent == python_dir
    except OSError:
        return False


def resolve_launcher(launcher: str) -> tuple[str, list[str], list[str]]:
    """Return resolved launcher name, command prefix, and warnings."""
    warnings: list[str] = []
    torchrun_path = shutil.which("torchrun")
    if launcher == "python-module":
        return "python-module", [sys.executable, "-m", "torch.distributed.run"], warnings
    if launcher == "torchrun":
        if not torchrun_path:
            raise RuntimeError("launcher=torchrun requested but torchrun was not found on PATH")
        if not torchrun_matches_python_env(torchrun_path):
            warnings.append(
                f"torchrun is not from sys.executable env: torchrun={torchrun_path} "
                f"sys.executable={sys.executable}"
            )
        return "torchrun", [torchrun_path], warnings
    if launcher != "auto":
        raise ValueError(f"Unsupported launcher: {launcher}")
    if torchrun_path and torchrun_matches_python_env(torchrun_path):
        return "torchrun", [torchrun_path], warnings
    if torchrun_path:
        warnings.append(
            f"auto launcher ignored PATH torchrun from different env: torchrun={torchrun_path} "
            f"sys.executable={sys.executable}"
        )
    return "python-module", [sys.executable, "-m", "torch.distributed.run"], warnings


def build_command(
    config_path: Path,
    nproc_per_node: int,
    master_port: int,
    seed: int,
    launcher: str,
) -> list[str]:
    resolved_launcher, prefix, warnings = resolve_launcher(launcher)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"launcher resolved: requested={launcher} actual={resolved_launcher} prefix={command_for_display(prefix)}")
    return [
        *prefix,
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "train_semi.py",
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--port",
        str(master_port),
    ]


def command_for_display(cmd: list[str]) -> str:
    return " ".join(cmd)


def launcher_from_command(cmd: list[str]) -> str:
    if len(cmd) >= 3 and cmd[0] == sys.executable and cmd[1:3] == ["-m", "torch.distributed.run"]:
        return "python-module"
    return "torchrun"


def make_smoke_config(src: Path, method_name: str, timestamp: str, lowmem_batch_size: int | None) -> Path:
    cfg = load_yaml(src)
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg.setdefault("trainer", {})["epochs"] = 1
    if lowmem_batch_size is not None:
        smoke_cfg.setdefault("dataset", {}).setdefault("train", {})["batch_size"] = int(lowmem_batch_size)
    smoke_cfg.setdefault("hf", {})["enabled"] = False
    smoke_cfg.setdefault("hf", {})["auto_upload"] = False
    smoke_cfg.setdefault("hf", {})["upload_every_epoch"] = False
    smoke_cfg.setdefault("wandb", {})["enable"] = False
    smoke_cfg.setdefault("saver", {})["snapshot_dir"] = str(SMOKE_RUN_ROOT / timestamp / method_name)
    out_dir = SMOKE_CONFIG_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method_name}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(smoke_cfg, f, sort_keys=False)
    return out_path


def make_segment_config(src: Path, method_name: str, timestamp: str, target_epoch: int) -> Path:
    cfg = load_yaml(src)
    segment_cfg = copy.deepcopy(cfg)
    original_epochs = max_config_epochs(segment_cfg)
    if int(target_epoch) > original_epochs:
        raise ValueError(
            f"segment target_epoch={target_epoch} exceeds original trainer.epochs={original_epochs} for {method_name}"
        )
    segment_cfg.setdefault("trainer", {})["epochs"] = int(target_epoch)
    out_dir = SEGMENT_CONFIG_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method_name}_to_epoch_{int(target_epoch):03d}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(segment_cfg, f, sort_keys=False)
    return out_path


def latest_status_by_method(status_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not status_path.is_file():
        return latest
    with status_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: ignoring malformed status line {line_no}: {exc}", file=sys.stderr)
                continue
            method = record.get("method")
            if method:
                latest[str(method)] = record
    return latest


def append_status(status_path: Path, record: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def create_lock(suite_name: str, gpu: int, force_lock: bool) -> Path:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_ROOT / f"{suite_name}_gpu{gpu}.lock"
    if force_lock and lock_path.exists():
        lock_path.unlink()
    payload = {
        "suite_name": suite_name,
        "gpu": gpu,
        "pid": os.getpid(),
        "created_at": utc_now(),
    }
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
    return lock_path


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def run_nvidia_smi(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["nvidia-smi", *args],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found; cannot check GPU availability")
    query = run_nvidia_smi(
        [
            f"--id={gpu}",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed: {query.stderr.strip()}")
    first = query.stdout.strip().splitlines()[0]
    parts = [int(x.strip()) for x in first.split(",")]
    used_mb, total_mb, util = parts
    apps = run_nvidia_smi(
        [
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[dict[str, Any]] = []
    if apps.returncode == 0:
        for line in apps.stdout.strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 3 and cols[0] != "[Not Supported]":
                processes.append({"pid": cols[0], "process_name": cols[1], "used_memory_mb": cols[2]})
    return {
        "used_mb": used_mb,
        "total_mb": total_mb,
        "free_mb": total_mb - used_mb,
        "utilization_gpu": util,
        "processes": processes,
    }


def wait_for_gpu(gpu: int, min_free_mb: int, poll_sec: int, no_wait: bool, log_file: Path) -> bool:
    while True:
        snap = gpu_snapshot(gpu)
        if snap["free_mb"] >= min_free_mb:
            return True
        message = (
            f"GPU {gpu} busy: free_mb={snap['free_mb']} min_free_mb={min_free_mb} "
            f"used_mb={snap['used_mb']} total_mb={snap['total_mb']} processes={snap['processes']}"
        )
        print(message)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
        if no_wait:
            return False
        time.sleep(max(1, int(poll_sec)))


def read_log_text(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


FATAL_NAN_PATTERNS = (
    re.compile(r"\b(Sup|Uns|Pseudo)\s*:\s*(?:nan|inf)\b", re.IGNORECASE),
    re.compile(r"\b(?:total_)?loss\s*[:=]\s*(?:nan|inf)\b", re.IGNORECASE),
    re.compile(r"\bL_BCR\s*=\s*(?:nan|inf)\b", re.IGNORECASE),
    re.compile(r"\bbcr/loss_bcr\s*=\s*(?:nan|inf)\b", re.IGNORECASE),
    re.compile(r"\b(?:nan|inf)\s+(?:loss|gradient|grad)\b", re.IGNORECASE),
    re.compile(r"\b(?:loss|gradient|grad)\s+(?:is|became|produced)\s+(?:nan|inf|non-finite)\b", re.IGNORECASE),
)


def has_fatal_nan_or_inf(text: str) -> bool:
    """Return True only for NaN/Inf that indicates a failed training signal.

    Some debug statistics intentionally log NaN for empty sets, for example
    missing same/diff BCR pairs or absent saliency components. Those lines should
    not turn a zero-return-code full run into a failed suite method.
    """
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in FATAL_NAN_PATTERNS):
            return True
    return False


def classify_result(return_code: int | None, timed_out: bool, mode: str, log_path: Path) -> tuple[str, str]:
    text = read_log_text(log_path)
    lower = text.lower()
    matched = [p for p in ERROR_PATTERNS if p in text or p.lower() in lower]
    last_error = "; ".join(matched)
    if timed_out:
        has_progress = any(p in text for p in SMOKE_PROGRESS_PATTERNS)
        timeout_signal_only = (
            "Received 15 death signal" in text
            and "SignalException" in text
            and "CUDA out of memory" not in text
            and "torch.OutOfMemoryError" not in text
            and "RuntimeError" not in text
            and "KeyboardInterrupt" not in text
        )
        has_nan = has_fatal_nan_or_inf(text)
        if mode == "smoke" and has_progress and (not matched or (timeout_signal_only and not has_nan)):
            return "timeout_smoke_ok", last_error
        return "timeout", last_error
    if return_code == 0:
        if "cuda out of memory" in lower or "torch.outofmemoryerror" in lower:
            return "failed_oom", last_error
        if "traceback" in lower:
            return "failed_traceback", last_error
        if has_fatal_nan_or_inf(text):
            return "failed", last_error or "fatal NaN/Inf detected in loss log"
        return "success", ""
    if "cuda out of memory" in lower or "torch.outofmemoryerror" in lower:
        return "failed_oom", last_error
    if "traceback" in lower:
        return "failed_traceback", last_error
    return "failed", last_error


def run_process(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout_sec: int | None,
    heartbeat: Any | None = None,
) -> tuple[int | None, bool]:
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        if heartbeat is not None:
            heartbeat(proc.pid)
        try:
            return proc.wait(timeout=timeout_sec), False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return proc.returncode, True
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise


def base_status_record(
    registry: dict[str, Any],
    method: dict[str, Any],
    args: argparse.Namespace,
    crop: list[int],
    global_batch: int,
    cmd: list[str],
    log_path: Path,
    mode: str,
) -> dict[str, Any]:
    return {
        "suite_name": registry["suite_name"],
        "method": method["name"],
        "group": method["group"],
        "config": method["config"],
        "mode": mode,
        "gpu": int(args.gpu),
        "nproc_per_node": int(args.nproc_per_node),
        "global_batch_size": int(global_batch),
        "crop_size": crop,
        "command": command_for_display(cmd),
        "launcher": args.launcher,
        "resolved_launcher": launcher_from_command(cmd),
        "start_time": "",
        "end_time": "",
        "duration_sec": 0.0,
        "return_code": None,
        "status": "pending",
        "log_path": str(log_path),
        "last_error": "",
    }


def db_import():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception as exc:
        raise RuntimeError(
            "Postgres queue backend requires psycopg. Install psycopg or use --queue-backend local."
        ) from exc
    return psycopg, dict_row


class PostgresQueue:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.psycopg, self.dict_row = db_import()

    def connect(self):
        return self.psycopg.connect(self.db_url, row_factory=self.dict_row)

    def init_schema(self) -> None:
        with self.connect() as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(hashtext('augseg_experiment_suite_queue_schema'))")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiment_suite_queue (
                      suite_name TEXT NOT NULL,
                      method TEXT NOT NULL,
                      group_name TEXT NOT NULL,
                      config_path TEXT NOT NULL,
                      mode TEXT NOT NULL,
                      status TEXT NOT NULL DEFAULT 'pending',
                      worker_id TEXT,
                      server_name TEXT,
                      gpu_id INTEGER,
                      nproc_per_node INTEGER,
                      crop_size TEXT,
                      global_batch_size INTEGER,
                      command TEXT,
                      log_path TEXT,
                      pid INTEGER,
                      return_code INTEGER,
                      retries INTEGER NOT NULL DEFAULT 0,
                      max_retries INTEGER NOT NULL DEFAULT 1,
                      started_at TIMESTAMPTZ,
                      ended_at TIMESTAMPTZ,
                      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                      heartbeat_at TIMESTAMPTZ,
                      last_error TEXT NOT NULL DEFAULT '',
                      PRIMARY KEY (suite_name, method, mode)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_experiment_suite_queue_claim
                    ON experiment_suite_queue (suite_name, mode, status, updated_at)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_experiment_suite_queue_heartbeat
                    ON experiment_suite_queue (suite_name, mode, status, heartbeat_at)
                    """
                )
                for statement in (
                    "ALTER TABLE experiment_suite_queue ADD COLUMN IF NOT EXISTS schedule_mode TEXT NOT NULL DEFAULT 'full_method'",
                    "ALTER TABLE experiment_suite_queue ADD COLUMN IF NOT EXISTS current_epoch INTEGER NOT NULL DEFAULT 0",
                    "ALTER TABLE experiment_suite_queue ADD COLUMN IF NOT EXISTS target_epoch INTEGER",
                    "ALTER TABLE experiment_suite_queue ADD COLUMN IF NOT EXISTS epoch_targets TEXT",
                ):
                    conn.execute(statement)
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_experiment_suite_queue_segment_claim
                    ON experiment_suite_queue (suite_name, mode, schedule_mode, status, current_epoch, method)
                    """
                )

    def seed(
        self,
        registry: dict[str, Any],
        methods: list[dict[str, Any]],
        mode: str,
        nproc_per_node: int,
        force: bool,
        max_retries: int,
        schedule_mode: str = "full_method",
        epoch_targets: list[int] | None = None,
    ) -> None:
        self.init_schema()
        target_epoch = int(epoch_targets[0]) if schedule_mode == "segments" and epoch_targets else None
        epoch_targets_text = ",".join(str(x) for x in epoch_targets) if epoch_targets else ""
        with self.connect() as conn:
            with conn.transaction():
                for method in methods:
                    crop, global_batch = validate_full_config(method, registry, nproc_per_node)
                    row = conn.execute(
                        """
                        SELECT status FROM experiment_suite_queue
                        WHERE suite_name=%s AND method=%s AND mode=%s
                        FOR UPDATE
                        """,
                        (registry["suite_name"], method["name"], mode),
                    ).fetchone()
                    if row is None:
                        conn.execute(
                            """
                            INSERT INTO experiment_suite_queue (
                              suite_name, method, group_name, config_path, mode, status,
                              nproc_per_node, crop_size, global_batch_size, retries,
                              max_retries, schedule_mode, current_epoch, target_epoch,
                              epoch_targets, updated_at
                            )
                            VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s, %s, 0, %s,
                                    %s, 0, %s, %s, NOW())
                            """,
                            (
                                registry["suite_name"],
                                method["name"],
                                method["group"],
                                method["config"],
                                mode,
                                int(nproc_per_node),
                                json.dumps(crop),
                                int(global_batch),
                                int(max_retries),
                                schedule_mode,
                                target_epoch,
                                epoch_targets_text,
                            ),
                        )
                    else:
                        reset_sql = ", status='pending', worker_id=NULL, server_name=NULL, gpu_id=NULL, pid=NULL, return_code=NULL, started_at=NULL, ended_at=NULL, heartbeat_at=NULL, last_error='', current_epoch=0, target_epoch=%s"
                        reset_params = [target_epoch] if force else []
                        conn.execute(
                            f"""
                            UPDATE experiment_suite_queue
                            SET group_name=%s, config_path=%s, nproc_per_node=%s,
                                crop_size=%s, global_batch_size=%s, max_retries=%s,
                                schedule_mode=%s, epoch_targets=%s,
                                updated_at=NOW()
                                {reset_sql if force else ""}
                            WHERE suite_name=%s AND method=%s AND mode=%s
                            """,
                            (
                                method["group"],
                                method["config"],
                                int(nproc_per_node),
                                json.dumps(crop),
                                int(global_batch),
                                int(max_retries),
                                schedule_mode,
                                epoch_targets_text,
                                *reset_params,
                                registry["suite_name"],
                                method["name"],
                                mode,
                            ),
                        )

    def mark_stale(self, suite_name: str, mode: str, max_stale_minutes: int) -> int:
        self.init_schema()
        with self.connect() as conn:
            with conn.transaction():
                result = conn.execute(
                    """
                    UPDATE experiment_suite_queue
                    SET status='failed_stale',
                        ended_at=NOW(),
                        updated_at=NOW(),
                        last_error='running heartbeat stale; process was not killed by queue runner'
                    WHERE suite_name=%s
                      AND mode=%s
                      AND status='running'
                      AND COALESCE(heartbeat_at, started_at, updated_at) < NOW() - (%s::text || ' minutes')::interval
                    """,
                    (suite_name, mode, int(max_stale_minutes)),
                )
                return int(result.rowcount or 0)

    def claim_one(
        self,
        registry: dict[str, Any],
        mode: str,
        worker_id: str,
        server_name: str,
        gpu: int,
        retry_failed: bool,
        max_retries: int,
        schedule_mode: str = "full_method",
        epoch_targets: list[int] | None = None,
    ) -> dict[str, Any] | None:
        self.init_schema()
        retry_statuses = tuple(sorted(RETRYABLE_STATUSES))
        max_target = int(epoch_targets[-1]) if epoch_targets else None
        with self.connect() as conn:
            with conn.transaction():
                active = conn.execute(
                    """
                    SELECT method, worker_id, server_name, gpu_id, pid, heartbeat_at, log_path
                    FROM experiment_suite_queue
                    WHERE suite_name=%s
                      AND mode=%s
                      AND schedule_mode=%s
                      AND status='running'
                      AND (
                        worker_id=%s
                        OR (server_name=%s AND gpu_id=%s)
                      )
                    ORDER BY started_at ASC NULLS LAST, updated_at ASC, method ASC
                    FOR UPDATE
                    LIMIT 1
                    """,
                    (registry["suite_name"], mode, schedule_mode, worker_id, server_name, int(gpu)),
                ).fetchone()
                if active is not None:
                    print(
                        "worker already has running job "
                        f"method={active.get('method')} "
                        f"worker_id={active.get('worker_id')} "
                        f"server_name={active.get('server_name')} "
                        f"gpu_id={active.get('gpu_id')} "
                        f"pid={active.get('pid')} "
                        f"heartbeat_at={active.get('heartbeat_at')} "
                        f"log_path={active.get('log_path') or ''}"
                    )
                    return None

                if schedule_mode == "segments":
                    if not epoch_targets or max_target is None:
                        raise ValueError("segments schedule requires epoch_targets")
                    row = conn.execute(
                        """
                        WITH candidate AS (
                          SELECT suite_name, method, mode,
                                 current_epoch,
                                 CASE
                                   WHEN current_epoch < %s THEN %s
                                   WHEN current_epoch < %s THEN %s
                                   WHEN current_epoch < %s THEN %s
                                   WHEN current_epoch < %s THEN %s
                                   ELSE NULL
                                 END AS next_target_epoch
                          FROM experiment_suite_queue
                          WHERE suite_name=%s
                            AND mode=%s
                            AND schedule_mode='segments'
                            AND current_epoch < %s
                            AND (
                              status='pending'
                              OR (
                                %s
                                AND status = ANY(%s)
                                AND retries < LEAST(max_retries, %s)
                              )
                            )
                          ORDER BY current_epoch ASC, method ASC
                          FOR UPDATE SKIP LOCKED
                          LIMIT 1
                        )
                        UPDATE experiment_suite_queue q
                        SET status='running',
                            worker_id=%s,
                            server_name=%s,
                            gpu_id=%s,
                            pid=NULL,
                            return_code=NULL,
                            target_epoch=candidate.next_target_epoch,
                            started_at=NOW(),
                            ended_at=NULL,
                            updated_at=NOW(),
                            heartbeat_at=NOW(),
                            last_error='',
                            retries=q.retries + CASE WHEN q.status='pending' THEN 0 ELSE 1 END
                        FROM candidate
                        WHERE q.suite_name=candidate.suite_name
                          AND q.method=candidate.method
                          AND q.mode=candidate.mode
                        RETURNING q.*
                        """,
                        (
                            epoch_targets[0], epoch_targets[0],
                            epoch_targets[1], epoch_targets[1],
                            epoch_targets[2], epoch_targets[2],
                            epoch_targets[3], epoch_targets[3],
                            registry["suite_name"],
                            mode,
                            max_target,
                            bool(retry_failed),
                            list(retry_statuses),
                            int(max_retries),
                            worker_id,
                            server_name,
                            int(gpu),
                        ),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        WITH candidate AS (
                          SELECT suite_name, method, mode
                          FROM experiment_suite_queue
                          WHERE suite_name=%s
                            AND mode=%s
                            AND schedule_mode='full_method'
                            AND (
                              status='pending'
                              OR (
                                %s
                                AND status = ANY(%s)
                                AND retries < LEAST(max_retries, %s)
                              )
                            )
                          ORDER BY updated_at ASC, method ASC
                          FOR UPDATE SKIP LOCKED
                          LIMIT 1
                        )
                        UPDATE experiment_suite_queue q
                        SET status='running',
                            worker_id=%s,
                            server_name=%s,
                            gpu_id=%s,
                            pid=NULL,
                            return_code=NULL,
                            started_at=NOW(),
                            ended_at=NULL,
                            updated_at=NOW(),
                            heartbeat_at=NOW(),
                            last_error='',
                            retries=q.retries + CASE WHEN q.status='pending' THEN 0 ELSE 1 END
                        FROM candidate
                        WHERE q.suite_name=candidate.suite_name
                          AND q.method=candidate.method
                          AND q.mode=candidate.mode
                        RETURNING q.*
                        """,
                        (
                            registry["suite_name"],
                            mode,
                            bool(retry_failed),
                            list(retry_statuses),
                            int(max_retries),
                            worker_id,
                            server_name,
                            int(gpu),
                        ),
                    ).fetchone()
                return dict(row) if row else None

    def update_running(
        self,
        suite_name: str,
        method: str,
        mode: str,
        **fields: Any,
    ) -> None:
        if not fields:
            return
        allowed = {
            "command",
            "log_path",
            "pid",
            "nproc_per_node",
            "crop_size",
            "global_batch_size",
            "status",
            "last_error",
            "target_epoch",
        }
        assignments = []
        values = []
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"unsupported queue field: {key}")
            assignments.append(f"{key}=%s")
            values.append(value)
        assignments.append("updated_at=NOW()")
        with self.connect() as conn:
            with conn.transaction():
                conn.execute(
                    f"""
                    UPDATE experiment_suite_queue
                    SET {", ".join(assignments)}
                    WHERE suite_name=%s AND method=%s AND mode=%s
                    """,
                    (*values, suite_name, method, mode),
                )

    def heartbeat(self, suite_name: str, method: str, mode: str, pid: int | None = None) -> None:
        extra = ", pid=%s" if pid is not None else ""
        params: tuple[Any, ...]
        if pid is not None:
            params = (int(pid), suite_name, method, mode)
        else:
            params = (suite_name, method, mode)
        with self.connect() as conn:
            with conn.transaction():
                conn.execute(
                    f"""
                    UPDATE experiment_suite_queue
                    SET heartbeat_at=NOW(), updated_at=NOW(){extra}
                    WHERE suite_name=%s AND method=%s AND mode=%s AND status='running'
                    """,
                    params,
                )

    def finish(
        self,
        suite_name: str,
        method: str,
        mode: str,
        status: str,
        return_code: int | None,
        last_error: str,
    ) -> None:
        with self.connect() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE experiment_suite_queue
                    SET status=%s,
                        return_code=%s,
                        ended_at=NOW(),
                        updated_at=NOW(),
                        heartbeat_at=NOW(),
                        last_error=%s
                    WHERE suite_name=%s AND method=%s AND mode=%s
                    """,
                    (status, return_code, last_error, suite_name, method, mode),
                )

    def finish_segment(
        self,
        suite_name: str,
        method: str,
        mode: str,
        status: str,
        return_code: int | None,
        last_error: str,
        target_epoch: int,
        max_target_epoch: int,
        next_target_epoch: int | None,
    ) -> None:
        with self.connect() as conn:
            with conn.transaction():
                if status == "success":
                    final_status = "success" if int(target_epoch) >= int(max_target_epoch) else "pending"
                    conn.execute(
                        """
                        UPDATE experiment_suite_queue
                        SET status=%s,
                            current_epoch=%s,
                            target_epoch=%s,
                            worker_id=NULL,
                            server_name=NULL,
                            gpu_id=NULL,
                            pid=NULL,
                            return_code=%s,
                            ended_at=NOW(),
                            updated_at=NOW(),
                            heartbeat_at=NOW(),
                            last_error=%s
                        WHERE suite_name=%s AND method=%s AND mode=%s AND status='running'
                        """,
                        (
                            final_status,
                            int(target_epoch),
                            next_target_epoch,
                            return_code,
                            last_error,
                            suite_name,
                            method,
                            mode,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE experiment_suite_queue
                        SET status=%s,
                            return_code=%s,
                            ended_at=NOW(),
                            updated_at=NOW(),
                            heartbeat_at=NOW(),
                            last_error=%s
                        WHERE suite_name=%s AND method=%s AND mode=%s AND status='running'
                        """,
                        (status, return_code, last_error, suite_name, method, mode),
                    )

    def rows(
        self,
        suite_name: str,
        mode: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self.init_schema()
        clauses = ["suite_name=%s"]
        params: list[Any] = [suite_name]
        if mode:
            clauses.append("mode=%s")
            params.append(mode)
        if status:
            clauses.append("status=%s")
            params.append(status)
        where_sql = " AND ".join(clauses)
        if mode:
            sql = """
                SELECT * FROM experiment_suite_queue
                WHERE {where_sql}
                ORDER BY current_epoch, method
            """.format(where_sql=where_sql)
        else:
            sql = """
                SELECT * FROM experiment_suite_queue
                WHERE {where_sql}
                ORDER BY mode, current_epoch, method
            """.format(where_sql=where_sql)
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]

    def release_claim_test(self, suite_name: str, method: str, mode: str, last_error: str) -> None:
        with self.connect() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE experiment_suite_queue
                    SET status='pending',
                        worker_id=NULL,
                        server_name=NULL,
                        gpu_id=NULL,
                        pid=NULL,
                        return_code=NULL,
                        started_at=NULL,
                        ended_at=NULL,
                        heartbeat_at=NULL,
                        updated_at=NOW(),
                        last_error=%s
                    WHERE suite_name=%s AND method=%s AND mode=%s AND status='running'
                    """,
                    (last_error, suite_name, method, mode),
                )


def db_url_from_args(args: argparse.Namespace) -> str:
    value = os.environ.get(args.db_url_env)
    if not value:
        raise RuntimeError(f"{args.db_url_env} is required for --queue-backend postgres")
    return value


def start_heartbeat_thread(
    queue: PostgresQueue,
    suite_name: str,
    method: str,
    mode: str,
    heartbeat_sec: int,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def loop() -> None:
        while not stop_event.wait(max(1, int(heartbeat_sec))):
            try:
                queue.heartbeat(suite_name, method, mode)
            except Exception as exc:
                print(f"warning: heartbeat failed for {method}: {exc}", file=sys.stderr)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop_event, thread


def print_postgres_status(rows: list[dict[str, Any]]) -> None:
    headers = [
        "method",
        "group",
        "status",
        "schedule_mode",
        "current_epoch",
        "target_epoch",
        "next_target_epoch",
        "worker_id",
        "server",
        "gpu",
        "pid",
        "started_at",
        "ended_at",
        "updated_at",
        "heartbeat_at",
        "retries",
        "log_path",
        "last_error",
    ]
    print("\t".join(headers))
    for row in rows:
        epoch_targets = parse_epoch_targets(row.get("epoch_targets") or "20,40,60,80") if row.get("schedule_mode") == "segments" else []
        next_target = next_epoch_target(int(row.get("current_epoch") or 0), epoch_targets) if epoch_targets else None
        values = [
            row.get("method", ""),
            row.get("group_name", ""),
            row.get("status", ""),
            row.get("schedule_mode", ""),
            str(row.get("current_epoch") or 0),
            "" if row.get("target_epoch") is None else str(row.get("target_epoch")),
            "" if next_target is None else str(next_target),
            row.get("worker_id") or "",
            row.get("server_name") or "",
            "" if row.get("gpu_id") is None else str(row.get("gpu_id")),
            "" if row.get("pid") is None else str(row.get("pid")),
            str(row.get("started_at") or ""),
            str(row.get("ended_at") or ""),
            str(row.get("updated_at") or ""),
            str(row.get("heartbeat_at") or ""),
            str(row.get("retries") or 0),
            row.get("log_path") or "",
            (row.get("last_error") or "").replace("\n", " ")[:160],
        ]
        print("\t".join(values))
    running_by_gpu: dict[tuple[str, str], int] = {}
    for row in rows:
        if row.get("status") != "running":
            continue
        gpu_id = row.get("gpu_id")
        key = (str(row.get("server_name") or ""), "" if gpu_id is None else str(gpu_id))
        running_by_gpu[key] = running_by_gpu.get(key, 0) + 1
    if running_by_gpu:
        summary = ", ".join(
            f"{server}:gpu{gpu}={count}" for (server, gpu), count in sorted(running_by_gpu.items())
        )
        print(f"running_by_gpu\t{summary}")


def check_nvidia_smi(args: argparse.Namespace) -> tuple[bool, str]:
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi not found"
    try:
        snap = gpu_snapshot(args.gpu)
    except Exception as exc:
        return False, str(exc)
    if snap["free_mb"] < int(args.min_free_mb):
        return (
            False,
            f"GPU {args.gpu} free_mb={snap['free_mb']} below min_free_mb={args.min_free_mb}; "
            f"processes={snap['processes']}",
        )
    return True, (
        f"GPU {args.gpu} free_mb={snap['free_mb']} used_mb={snap['used_mb']} "
        f"total_mb={snap['total_mb']} util={snap['utilization_gpu']}%"
    )


def check_torch() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:
        return False, f"import torch failed: {exc}"
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        return False, f"torch cuda query failed: {exc}"
    if not cuda_available or device_count < 1:
        return False, (
            f"torch={torch.__version__} cuda_available={cuda_available} device_count={device_count}"
        )
    return True, f"torch={torch.__version__} cuda_available={cuda_available} device_count={device_count}"


def preflight(args: argparse.Namespace) -> int:
    failures: list[str] = []

    def report(name: str, ok: bool, detail: str) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failures.append(f"{name}: {detail}")

    registry_path = resolve_path(args.registry)
    try:
        registry = effective_registry(load_registry(registry_path), args)
        epoch_targets = parse_epoch_targets(args.epoch_targets)
        methods = select_methods(registry, args.only, args.skip, args.group)
        report("registry", True, f"{registry_path} selected_methods={len(methods)}")
        report(
            "method-count",
            len(registry["methods"]) == EXPECTED_METHOD_COUNT,
            f"registry methods={len(registry['methods'])} expected={EXPECTED_METHOD_COUNT}",
        )
        if args.schedule_mode == "segments":
            report(
                "segments",
                True,
                f"suite={registry['suite_name']} epoch_targets={epoch_targets}",
            )
    except Exception as exc:
        report("registry", False, str(exc))
        return 1

    for method in methods:
        config_path = resolve_path(method["config"])
        report(f"config:{method['name']}", config_path.is_file(), str(config_path))
        if args.mode == "full":
            try:
                crop, global_batch = validate_full_config(method, registry, args.nproc_per_node)
                report(
                    f"full-config:{method['name']}",
                    True,
                    f"crop={crop} global_batch={global_batch}",
                )
            except Exception as exc:
                report(f"full-config:{method['name']}", False, str(exc))

    report("sys.executable", True, sys.executable)
    report("python", True, platform.python_version())
    torch_ok, torch_detail = check_torch()
    report("torch", torch_ok, torch_detail)
    smi_ok, smi_detail = check_nvidia_smi(args)
    report("nvidia-smi", smi_ok, smi_detail)

    try:
        resolved, prefix, warnings = resolve_launcher(args.launcher)
        command = [
            *prefix,
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}",
            "train_semi.py",
            "--config",
            str(resolve_path(methods[0]["config"])),
            "--seed",
            str(args.seed),
            "--port",
            str(args.master_port),
        ]
        detail = f"requested={args.launcher} actual={resolved} command={command_for_display(command)}"
        if warnings:
            detail += f" warnings={' | '.join(warnings)}"
        report("launcher", True, detail)
    except Exception as exc:
        report("launcher", False, str(exc))

    if args.queue_backend == "postgres":
        try:
            queue = PostgresQueue(db_url_from_args(args))
            rows = queue.rows(registry["suite_name"], args.mode)
            counts: dict[str, int] = {}
            for row in rows:
                counts[str(row.get("status", ""))] = counts.get(str(row.get("status", "")), 0) + 1
            report("postgres", True, f"connected rows={len(rows)} status_counts={counts}")
        except Exception as exc:
            report("postgres", False, str(exc))

    if failures:
        print("PRECHECK FAIL")
        return 1
    print("PRECHECK PASS")
    return 0


def run_claim_test(queue: PostgresQueue, registry: dict[str, Any], args: argparse.Namespace) -> int:
    if args.mode == "full":
        raise ValueError("--claim-test is only allowed with --mode smoke or --mode dry-run")
    if not args.worker_id:
        raise ValueError("--worker-id is required for --claim-test")
    if not args.server_name:
        raise ValueError("--server-name is required for --claim-test")
    row = queue.claim_one(
        registry,
        args.mode,
        args.worker_id,
        args.server_name,
        args.gpu,
        args.retry_failed,
        args.max_retries,
        args.schedule_mode,
        parse_epoch_targets(args.epoch_targets) if args.schedule_mode == "segments" else None,
    )
    if row is None:
        print(f"claim-test: no pending queue row for suite={registry['suite_name']} mode={args.mode}")
        return 1
    method = str(row["method"])
    queue.heartbeat(registry["suite_name"], method, args.mode, pid=os.getpid())
    queue.release_claim_test(
        registry["suite_name"],
        method,
        args.mode,
        f"claim-test released by {args.worker_id} at {utc_now()}",
    )
    print(f"claim-test: claimed and released method={method} mode={args.mode} worker_id={args.worker_id}")
    return 0


def dry_run_commands(registry: dict[str, Any], args: argparse.Namespace) -> list[list[str]]:
    methods = select_methods(registry, args.only, args.skip, args.group)
    epoch_targets = parse_epoch_targets(args.epoch_targets)
    timestamp = "dry_run_segments" if args.schedule_mode == "segments" else "dry_run"
    commands: list[list[str]] = []
    for index, method in enumerate(methods):
        validate_full_config(method, registry, args.nproc_per_node)
        if args.schedule_mode == "segments":
            cfg = load_yaml(resolve_path(method["config"]))
            if epoch_targets[-1] > max_config_epochs(cfg):
                raise ValueError(f"{method['name']} trainer.epochs is below final segment target")
            config_path = make_segment_config(resolve_path(method["config"]), method["name"], timestamp, epoch_targets[0])
        else:
            config_path = resolve_path(method["config"])
        port = int(args.master_port) + index
        commands.append(build_command(config_path, args.nproc_per_node, port, args.seed, args.launcher))
    return commands


def method_by_name(registry: dict[str, Any], method_name: str) -> dict[str, Any]:
    for method in registry["methods"]:
        if method["name"] == method_name:
            return method
    raise KeyError(f"Method from queue is not in registry: {method_name}")


def prepare_run_config(
    method: dict[str, Any],
    registry: dict[str, Any],
    args: argparse.Namespace,
    timestamp: str,
) -> tuple[Path, list[int], int]:
    original_config = resolve_path(method["config"])
    if args.mode == "full" and args.schedule_mode == "full_method":
        crop, global_batch = validate_full_config(method, registry, args.nproc_per_node)
        return original_config, crop, global_batch
    if args.mode == "full" and args.schedule_mode == "segments":
        crop, global_batch = validate_full_config(method, registry, args.nproc_per_node)
        target_epoch = int(args.segment_target_epoch)
        return make_segment_config(original_config, method["name"], timestamp, target_epoch), crop, global_batch

    full_crop, _ = validate_full_config(method, registry, 1)
    run_config = make_smoke_config(original_config, method["name"], timestamp, args.lowmem_batch_size)
    smoke_cfg = load_yaml(run_config)
    batch = train_batch_size(smoke_cfg)
    if batch is None:
        raise ValueError(f"smoke config missing batch size for {method['name']}")
    return run_config, full_crop, batch * int(args.nproc_per_node)


def run_postgres_claimed_method(
    queue: PostgresQueue,
    registry: dict[str, Any],
    row: dict[str, Any],
    args: argparse.Namespace,
    timestamp: str,
) -> int:
    method = method_by_name(registry, row["method"])
    if args.schedule_mode == "segments":
        args.segment_target_epoch = int(row["target_epoch"])
    run_config, crop, global_batch = prepare_run_config(method, registry, args, timestamp)
    method_index = [m["name"] for m in registry["methods"]].index(method["name"])
    port = int(args.master_port) + method_index
    cmd = build_command(run_config, args.nproc_per_node, port, args.seed, args.launcher)
    log_dir = resolve_path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.schedule_mode == "segments":
        log_path = log_dir / (
            f"{method['name']}_e{int(row['current_epoch']):03d}_to_e{int(row['target_epoch']):03d}.log"
        )
    else:
        log_path = log_dir / f"{method['name']}.log"
    queue.update_running(
        registry["suite_name"],
        method["name"],
        args.mode,
        command=command_for_display(cmd),
        log_path=str(log_path),
        nproc_per_node=int(args.nproc_per_node),
        crop_size=json.dumps(crop),
        global_batch_size=int(global_batch),
        target_epoch=int(row["target_epoch"]) if args.schedule_mode == "segments" else None,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    stop_event, thread = start_heartbeat_thread(
        queue,
        registry["suite_name"],
        method["name"],
        args.mode,
        args.heartbeat_sec,
    )

    def first_heartbeat(pid: int) -> None:
        queue.heartbeat(registry["suite_name"], method["name"], args.mode, pid=pid)

    if args.schedule_mode == "segments":
        print(
            f"postgres running segment {method['name']} "
            f"{int(row['current_epoch'])}->{int(row['target_epoch'])} -> {log_path}"
        )
    else:
        print(f"postgres running {method['name']} -> {log_path}")
    try:
        try:
            return_code, timed_out = run_process(cmd, env, log_path, args.timeout_sec, heartbeat=first_heartbeat)
        except Exception as exc:
            queue.finish(registry["suite_name"], method["name"], args.mode, "failed", None, str(exc))
            raise
    finally:
        stop_event.set()
        thread.join(timeout=5)
    status, last_error = classify_result(return_code, timed_out, args.mode, log_path)
    if args.schedule_mode == "segments":
        epoch_targets = parse_epoch_targets(args.epoch_targets)
        target_epoch = int(row["target_epoch"])
        queue.finish_segment(
            registry["suite_name"],
            method["name"],
            args.mode,
            status,
            return_code,
            last_error,
            target_epoch,
            epoch_targets[-1],
            next_epoch_target(target_epoch, epoch_targets),
        )
        print(
            f"{method['name']} segment {int(row['current_epoch'])}->{target_epoch}: "
            f"{status} return_code={return_code} log={log_path}"
        )
    else:
        queue.finish(registry["suite_name"], method["name"], args.mode, status, return_code, last_error)
        print(f"{method['name']}: {status} return_code={return_code} log={log_path}")
    return 0 if status in ("success", "timeout_smoke_ok") else 1


def postgres_worker_step(
    queue: PostgresQueue,
    registry: dict[str, Any],
    args: argparse.Namespace,
    timestamp: str,
) -> tuple[bool, int]:
    gpu_log_dir = resolve_path(args.log_dir) / timestamp
    gpu_log_dir.mkdir(parents=True, exist_ok=True)
    worker_label = args.worker_id.replace("/", "_").replace(":", "_")
    gpu_log = gpu_log_dir / f"{worker_label}_gpu_guard.log"
    if not wait_for_gpu(args.gpu, args.min_free_mb, args.poll_sec, args.no_wait, gpu_log):
        print(f"GPU {args.gpu} is busy; no method claimed")
        return False, 1

    stale_count = queue.mark_stale(registry["suite_name"], args.mode, args.max_stale_minutes)
    if stale_count:
        print(f"marked {stale_count} stale running queue item(s)")
    row = queue.claim_one(
        registry,
        args.mode,
        args.worker_id,
        args.server_name,
        args.gpu,
        args.retry_failed,
        args.max_retries,
        args.schedule_mode,
        parse_epoch_targets(args.epoch_targets) if args.schedule_mode == "segments" else None,
    )
    if row is None:
        print("No pending queue item available")
        return False, 0
    return True, run_postgres_claimed_method(queue, registry, row, args, timestamp)


def run_postgres_suite(args: argparse.Namespace) -> int:
    registry_path = resolve_path(args.registry)
    registry = effective_registry(load_registry(registry_path), args)
    epoch_targets = parse_epoch_targets(args.epoch_targets)
    methods = select_methods(registry, args.only, args.skip, args.group)
    if not methods:
        raise ValueError("No methods selected")

    if args.mode == "dry-run":
        commands = dry_run_commands(registry, args)
        for method, cmd in zip(methods, commands):
            print(f"{method['name']}: {command_for_display(cmd)}")
        print(f"dry-run commands: {len(commands)}")
        return 0

    queue = PostgresQueue(db_url_from_args(args))

    if args.init_queue:
        queue.seed(
            registry,
            methods,
            args.mode,
            args.nproc_per_node,
            args.force,
            args.max_retries,
            args.schedule_mode,
            epoch_targets if args.schedule_mode == "segments" else None,
        )
        print(
            f"seeded queue rows for {len(methods)} method(s) in suite={registry['suite_name']} "
            f"mode={args.mode} schedule_mode={args.schedule_mode}"
        )

    if args.claim_test:
        return run_claim_test(queue, registry, args)

    if args.status or args.status_running:
        status_filter = "running" if args.status_running else None
        print_postgres_status(queue.rows(registry["suite_name"], args.mode, status_filter))
        if not args.once and not args.loop:
            return 0

    if not args.once and not args.loop:
        if args.init_queue:
            return 0
        raise ValueError("Postgres backend requires --init-queue, --status, --once, or --loop")
    if not args.worker_id:
        raise ValueError("--worker-id is required for Postgres workers")
    if not args.server_name:
        raise ValueError("--server-name is required for Postgres workers")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.once:
        _, code = postgres_worker_step(queue, registry, args, timestamp)
        return code

    exit_code = 0
    while True:
        claimed, code = postgres_worker_step(queue, registry, args, timestamp)
        exit_code = max(exit_code, code)
        if claimed:
            continue
        print(f"Sleeping {args.sleep_sec}s before checking queue again")
        time.sleep(max(1, int(args.sleep_sec)))
    return exit_code


def run_entry(args: argparse.Namespace) -> int:
    if args.preflight:
        return preflight(args)
    if args.queue_backend == "postgres":
        return run_postgres_suite(args)
    if args.claim_test:
        raise ValueError("--claim-test requires --queue-backend postgres")
    return run_suite(args)


def run_suite(args: argparse.Namespace) -> int:
    registry_path = resolve_path(args.registry)
    registry = effective_registry(load_registry(registry_path), args)
    suite_name = registry["suite_name"]
    methods = select_methods(registry, args.only, args.skip, args.group)
    if not methods:
        raise ValueError("No methods selected")

    if args.mode == "dry-run":
        commands = dry_run_commands(registry, args)
        for method, cmd in zip(methods, commands):
            print(f"{method['name']}: {command_for_display(cmd)}")
        print(f"dry-run commands: {len(commands)}")
        return 0

    status_dir = resolve_path(args.status_dir)
    log_root = resolve_path(args.log_dir)
    status_path = status_dir / "status.jsonl"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_root / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    latest = latest_status_by_method(status_path)
    lock_path = create_lock(suite_name, args.gpu, args.force_lock)
    exit_code = 0
    try:
        for index, method in enumerate(methods):
            if args.resume and not args.force and latest.get(method["name"], {}).get("status") == "success":
                log_path = log_dir / f"{method['name']}.log"
                crop, global_batch = validate_full_config(method, registry, args.nproc_per_node)
                cmd = build_command(
                    resolve_path(method["config"]),
                    args.nproc_per_node,
                    args.master_port + index,
                    args.seed,
                    args.launcher,
                )
                record = base_status_record(registry, method, args, crop, global_batch, cmd, log_path, args.mode)
                record.update({"status": "skipped_success", "start_time": utc_now(), "end_time": utc_now()})
                append_status(status_path, record)
                print(f"skip success: {method['name']}")
                continue

            original_config = resolve_path(method["config"])
            if args.mode == "full":
                crop, global_batch = validate_full_config(method, registry, args.nproc_per_node)
                run_config = original_config
            else:
                full_crop, _ = validate_full_config(method, registry, 1)
                crop = full_crop
                run_config = make_smoke_config(original_config, method["name"], timestamp, args.lowmem_batch_size)
                smoke_cfg = load_yaml(run_config)
                batch = train_batch_size(smoke_cfg)
                if batch is None:
                    raise ValueError(f"smoke config missing batch size for {method['name']}")
                global_batch = batch * int(args.nproc_per_node)

            port = int(args.master_port) + index
            cmd = build_command(run_config, args.nproc_per_node, port, args.seed, args.launcher)
            log_path = log_dir / f"{method['name']}.log"
            record = base_status_record(registry, method, args, crop, global_batch, cmd, log_path, args.mode)

            if not wait_for_gpu(args.gpu, args.min_free_mb, args.poll_sec, args.no_wait, log_path):
                now = utc_now()
                record.update({"status": "gpu_busy", "start_time": now, "end_time": now})
                append_status(status_path, record)
                exit_code = 1
                continue

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            started = time.time()
            record["start_time"] = utc_now()
            record["status"] = "running"
            append_status(status_path, record)
            print(f"running {method['name']} -> {log_path}")
            return_code, timed_out = run_process(cmd, env, log_path, args.timeout_sec)
            status, last_error = classify_result(return_code, timed_out, args.mode, log_path)
            ended = time.time()
            record.update(
                {
                    "end_time": utc_now(),
                    "duration_sec": round(ended - started, 3),
                    "return_code": return_code,
                    "status": status,
                    "last_error": last_error,
                }
            )
            append_status(status_path, record)
            print(f"{method['name']}: {status} return_code={return_code} log={log_path}")
            if status not in ("success", "timeout_smoke_ok"):
                exit_code = 1
    finally:
        release_lock(lock_path)
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VOC662 12-method experiment suite.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29531)
    parser.add_argument("--mode", choices=("full", "smoke", "dry-run"), default="dry-run")
    parser.add_argument("--schedule-mode", choices=("full_method", "segments"), default="full_method")
    parser.add_argument("--epoch-targets", default="20,40,60,80")
    parser.add_argument("--suite-name")
    parser.add_argument("--launcher", choices=("auto", "torchrun", "python-module"), default="auto")
    parser.add_argument("--only")
    parser.add_argument("--skip")
    parser.add_argument("--group")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-free-mb", type=int, default=12000)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int)
    parser.add_argument("--status-dir", default="runs/suite_status/voc662_12_methods")
    parser.add_argument("--log-dir", default="runs/suite_logs/voc662_12_methods")
    parser.add_argument("--lowmem-batch-size", type=int, default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--force-lock", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--queue-backend", choices=("local", "postgres"), default="local")
    parser.add_argument("--db-url-env", default="AUGSEG_SCHEDULER_DB_URL")
    parser.add_argument("--worker-id")
    parser.add_argument("--server-name")
    parser.add_argument("--init-queue", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--status-running", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep-sec", type=int, default=30)
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    parser.add_argument("--max-stale-minutes", type=int, default=60)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--claim-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_entry(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
