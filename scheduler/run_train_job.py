#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tarfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.config import ROOT, RUN_ROOT, load_hf_settings

CURRENT_TRAIN_PROCESS: subprocess.Popen | None = None
ERROR_TAIL_LINES = 200
ERROR_TAIL_CHARS = 8000


def install_train_signal_handlers() -> None:
    def _handler(signum, _frame) -> None:
        name = signal.Signals(signum).name
        print(f"run_train_job received {name}; terminating child torchrun", flush=True)
        proc = CURRENT_TRAIN_PROCESS
        if proc is not None and proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def choose_port(job_id: str) -> int:
    return 54000 + (int(job_id[:6], 16) % 1000)


def text_tail(text: str, max_lines: int = ERROR_TAIL_LINES, max_chars: int = ERROR_TAIL_CHARS) -> str:
    lines = text.splitlines()[-max_lines:]
    return "\n".join(lines)[-max_chars:]


def file_tail(path: Path, max_lines: int = ERROR_TAIL_LINES, max_chars: int = ERROR_TAIL_CHARS) -> str:
    try:
        return text_tail(path.read_text(errors="replace"), max_lines=max_lines, max_chars=max_chars)
    except OSError as exc:
        return f"unable to read log tail: {exc}"


def resolve_config_path(config_path: str | Path, repo_root: Path = ROOT) -> Path:
    original = str(config_path)
    src = Path(original)
    if src.is_file():
        return src

    resolved: Path | None = None
    if not src.is_absolute():
        candidate = repo_root / src
        if candidate.is_file():
            return candidate
        resolved = candidate

    normalized = original.replace("\\", "/")
    for marker in ("exps/", ".codex_smoke/"):
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            suffix = normalized[marker_index:]
            candidate = repo_root / Path(suffix)
            resolved = candidate
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "Unable to resolve scheduler config path: "
        f"original_config_path={original} "
        f"resolved_config_path={resolved if resolved is not None else src} "
        f"repo_root={repo_root}"
    )


def write_job_config(args: argparse.Namespace) -> Path:
    src = resolve_config_path(args.config_path, ROOT)
    cfg = yaml.safe_load(src.read_text())
    cfg["trainer"]["epochs"] = int(args.to_epoch)
    cfg.setdefault("hf", {})["enabled"] = False
    cfg.setdefault("wandb", {})["enable"] = False
    cfg.setdefault("saver", {})["snapshot_dir"] = f"./scheduler_runs/{args.config_id}/checkpoints"

    out = RUN_ROOT / "configs" / f"{args.job_id}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def collect_artifacts(job_id: str, config_id: str, job_config: Path) -> tuple[Path, list[Path]]:
    save_dir = ROOT / ".scheduler_runs/artifacts" / job_id
    save_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    checkpoint_dir = ROOT / "scheduler_runs" / config_id / "checkpoints"
    candidates = [
        checkpoint_dir / "ckpt.pth",
        checkpoint_dir / "latest.pth",
        checkpoint_dir / "ckpt_best.pth",
        checkpoint_dir / "epoch_metrics.csv",
        checkpoint_dir / "manifest.json",
        checkpoint_dir / "run_id.txt",
    ]
    for path in candidates:
        if path.is_file():
            dst = save_dir / path.name
            shutil.copy2(path, dst)
            copied.append(dst)

    for log in sorted((job_config.parent / "log").glob("*")) if (job_config.parent / "log").exists() else []:
        if log.is_file():
            dst = save_dir / f"log_{log.name}"
            shutil.copy2(log, dst)
            copied.append(dst)

    cfg_dst = save_dir / "config.yaml"
    shutil.copy2(job_config, cfg_dst)
    copied.append(cfg_dst)
    return save_dir, copied


def make_tarball(artifact_dir: Path, tar_path: Path) -> Path:
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(artifact_dir.glob("*")):
            if path.is_file() and path.resolve() != tar_path.resolve():
                tar.add(path, arcname=path.name)
    return tar_path


def config_hf_latest_path(job_config: Path, config_id: str) -> str:
    cfg = yaml.safe_load(job_config.read_text())
    configured = cfg.get("hf", {}).get("path_in_repo")
    if configured:
        return str(configured).strip("/")

    settings = load_hf_settings()
    base_path = settings.get("base_path", "voc5_single_gpu_gbs8").strip("/")
    return f"{base_path}/{config_id}/latest.tar.gz"


def upload_job_artifacts(
    job_id: str,
    config_id: str,
    job_config: Path,
    artifact_dir: Path,
    progress: dict,
) -> str | None:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub unavailable: {exc}") from exc

    settings = load_hf_settings()
    if not settings.get("upload_after_segment", True):
        return None
    repo_id = settings["repo_id"]
    repo_type = settings.get("repo_type", "model")
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    progress_path = artifact_dir / "progress.json"
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")

    latest_path = config_hf_latest_path(job_config, config_id)
    if latest_path.endswith("/latest.tar.gz"):
        job_path = latest_path[: -len("/latest.tar.gz")] + f"/jobs/{job_id}.tar.gz"
    else:
        job_path = f"{latest_path.rstrip('/')}/jobs/{job_id}.tar.gz"
    bundle_path = make_tarball(artifact_dir, artifact_dir / f"{job_id}.tar.gz")
    api.upload_file(path_or_fileobj=str(bundle_path), path_in_repo=job_path, repo_id=repo_id, repo_type=repo_type)
    return job_path


def main() -> int:
    global CURRENT_TRAIN_PROCESS
    install_train_signal_handlers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--from-epoch", type=int, required=True)
    parser.add_argument("--to-epoch", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    job_config = write_job_config(args)
    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.job_id}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        f"--master_port={choose_port(args.job_id)}",
        "train_semi.py",
        f"--config={job_config}",
        "--seed",
        str(args.seed),
    ]
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        CURRENT_TRAIN_PROCESS = proc
        returncode = proc.wait()
        CURRENT_TRAIN_PROCESS = None
    if returncode != 0:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "returncode": returncode,
                    "log": str(log_path),
                    "error_tail": file_tail(log_path),
                }
            )
        )
        return returncode

    artifact_dir, files = collect_artifacts(args.job_id, args.config_id, job_config)
    progress = {
        "job_id": args.job_id,
        "config_id": args.config_id,
        "from_epoch": args.from_epoch,
        "to_epoch": args.to_epoch,
        "seconds": round(time.time() - started, 3),
        "log": str(log_path),
    }
    job_hf_path = upload_job_artifacts(args.job_id, args.config_id, job_config, artifact_dir, progress)
    print(json.dumps({"status": "success", "job_hf_path": job_hf_path, "artifact_dir": str(artifact_dir), "log": str(log_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
