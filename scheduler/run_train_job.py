#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.config import ROOT, RUN_ROOT, load_hf_settings


def choose_port(job_id: str) -> int:
    return 54000 + (int(job_id[:6], 16) % 1000)


def write_job_config(args: argparse.Namespace) -> Path:
    src = Path(args.config_path)
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


def upload_job_artifacts(job_id: str, config_id: str, artifact_dir: Path, files: list[Path], progress: dict) -> str | None:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub unavailable: {exc}") from exc

    settings = load_hf_settings()
    if not settings.get("upload_after_segment", True):
        return None
    repo_id = settings["repo_id"]
    repo_type = settings.get("repo_type", "model")
    base_path = settings.get("base_path", "voc5_single_gpu_gbs8").strip("/")
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    progress_path = artifact_dir / "progress.json"
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    files = list(files) + [progress_path]

    job_prefix = f"{base_path}/{config_id}/jobs/{job_id}"
    for path in files:
        rel_name = path.name
        api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{job_prefix}/{rel_name}", repo_id=repo_id, repo_type=repo_type)
    return job_prefix


def main() -> int:
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
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(json.dumps({"status": "failed", "returncode": proc.returncode, "log": str(log_path)}))
        return proc.returncode

    artifact_dir, files = collect_artifacts(args.job_id, args.config_id, job_config)
    progress = {
        "job_id": args.job_id,
        "config_id": args.config_id,
        "from_epoch": args.from_epoch,
        "to_epoch": args.to_epoch,
        "seconds": round(time.time() - started, 3),
        "log": str(log_path),
    }
    job_hf_path = upload_job_artifacts(args.job_id, args.config_id, artifact_dir, files, progress)
    print(json.dumps({"status": "success", "job_hf_path": job_hf_path, "artifact_dir": str(artifact_dir), "log": str(log_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
