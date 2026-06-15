#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs/experiment_registry_voc662_12_methods.yaml"
LOCK_ROOT = ROOT / "runs/locks"
SMOKE_CONFIG_ROOT = ROOT / "tmp/suite_smoke_configs"
SMOKE_RUN_ROOT = ROOT / "tmp/suite_smoke_runs"
ERROR_PATTERNS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "Traceback",
    "NaN",
    " nan",
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


def launcher_prefix() -> list[str]:
    if shutil.which("torchrun"):
        return ["torchrun"]
    return [sys.executable, "-m", "torch.distributed.run"]


def build_command(config_path: Path, nproc_per_node: int, master_port: int, seed: int) -> list[str]:
    return [
        *launcher_prefix(),
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
        has_nan = "nan" in lower
        if mode == "smoke" and has_progress and (not matched or (timeout_signal_only and not has_nan)):
            return "timeout_smoke_ok", last_error
        return "timeout", last_error
    if return_code == 0:
        if "cuda out of memory" in lower or "torch.outofmemoryerror" in lower:
            return "failed_oom", last_error
        if "traceback" in lower:
            return "failed_traceback", last_error
        if "nan" in lower:
            return "failed", last_error or "NaN/nan detected in log"
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
        "start_time": "",
        "end_time": "",
        "duration_sec": 0.0,
        "return_code": None,
        "status": "pending",
        "log_path": str(log_path),
        "last_error": "",
    }


def dry_run_commands(registry: dict[str, Any], args: argparse.Namespace) -> list[list[str]]:
    methods = select_methods(registry, args.only, args.skip, args.group)
    commands: list[list[str]] = []
    for index, method in enumerate(methods):
        validate_full_config(method, registry, args.nproc_per_node)
        config_path = resolve_path(method["config"])
        port = int(args.master_port) + index
        commands.append(build_command(config_path, args.nproc_per_node, port, args.seed))
    return commands


def run_suite(args: argparse.Namespace) -> int:
    registry_path = resolve_path(args.registry)
    registry = load_registry(registry_path)
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
                cmd = build_command(resolve_path(method["config"]), args.nproc_per_node, args.master_port + index, args.seed)
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
            cmd = build_command(run_config, args.nproc_per_node, port, args.seed)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_suite(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
