from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from huggingface_hub import HfApi

from util.hf_auto import maybe_upload_hf_bundle


SUITE = "voc662_8sc_officialc3_rerun01_trainsemi_wandb_hf_20260701"
REPO_ID = "tanprodium/augseg-boundarymix-v2v3-runs"
REPO_TYPE = "model"

SEGMENT_END_EPOCH = 20

CONFIG_ROOT = Path("exps/boundary_mix_v2_v3/voc_semi662_reruns") / SUITE
SAVE_ROOT = Path("exp_boundary_mix_v2_v3/reruns") / SUITE
LOG_ROOT = Path("runs/suite_logs") / SUITE
SEGMENT_CONFIG_ROOT = Path("runs/suite_temp/segment_configs")


@dataclass
class MethodStatus:
    method: str
    cfg_path: Optional[Path]
    save_path: Path
    ckpt_path: Path
    ckpt_exists: bool
    raw_epoch: Optional[int]
    completed_epochs_guess: Optional[int]
    trainer_epochs: int
    complete: bool
    hf_path: str
    hf_latest_exists: Optional[bool]
    latest_log: Optional[Path]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    if not isinstance(data, dict):
        raise ValueError(f"YAML is not a dict: {path}")
    return data


def get_method_from_cfg(cfg: dict[str, Any]) -> str:
    run = cfg.get("run") or {}
    name = run.get("name")
    if name:
        return str(name)

    hf = cfg.get("hf") or {}
    path_in_repo = hf.get("path_in_repo")
    if path_in_repo:
        return str(path_in_repo).rstrip("/").split("/")[-1]

    save_path = cfg.get("save_path") or ""
    if save_path:
        base = Path(str(save_path)).name
        return base.replace("_r101_c321_bs8x1_gbs8", "")

    return ""


def find_config_path(method: str) -> Optional[Path]:
    direct = CONFIG_ROOT / method / "config.yaml"
    if direct.exists():
        return direct

    candidates: list[Path] = []

    if SEGMENT_CONFIG_ROOT.exists():
        candidates.extend(SEGMENT_CONFIG_ROOT.glob(f"**/{method}/config.yaml"))
        candidates.extend(SEGMENT_CONFIG_ROOT.glob(f"**/{method}*.yaml"))
        candidates.extend(SEGMENT_CONFIG_ROOT.glob(f"**/{method}*.yml"))

    candidates = sorted(
        [p for p in candidates if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for path in candidates:
        try:
            cfg = load_yaml(path)
            if get_method_from_cfg(cfg) == method:
                return path
        except Exception:
            continue

    return None


def discover_methods() -> list[str]:
    methods: set[str] = set()

    if CONFIG_ROOT.exists():
        for cfg_path in CONFIG_ROOT.glob("*/config.yaml"):
            methods.add(cfg_path.parent.name)

    if SAVE_ROOT.exists():
        for save_dir in SAVE_ROOT.glob("*_r101_c321_bs8x1_gbs8"):
            if save_dir.is_dir():
                method = save_dir.name.replace("_r101_c321_bs8x1_gbs8", "")
                methods.add(method)

    return sorted(methods)


def read_ckpt_epoch(ckpt_path: Path) -> Optional[int]:
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    epoch = ckpt.get("epoch")
    if epoch is None:
        return None

    return int(epoch)


def latest_log_for_method(method: str) -> Optional[Path]:
    if not LOG_ROOT.exists():
        return None

    logs = sorted(
        LOG_ROOT.glob(f"**/{method}_e*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def get_hf_files(api: HfApi) -> Optional[set[str]]:
    try:
        return set(api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))
    except Exception as exc:
        print(f"[WARN] cannot list HF repo files: {repr(exc)}")
        return None


def build_status(method: str, hf_files: Optional[set[str]]) -> MethodStatus:
    cfg_path = find_config_path(method)
    save_path = SAVE_ROOT / f"{method}_r101_c321_bs8x1_gbs8"
    ckpt_path = save_path / "ckpt.pth"

    trainer_epochs = SEGMENT_END_EPOCH
    hf_path = f"{SUITE}/{method}/latest.tar.gz"

    if cfg_path is not None:
        try:
            cfg = load_yaml(cfg_path)
            # This script checks segment 0->20 completion, not full 80-epoch ladder completion.
            trainer_epochs = SEGMENT_END_EPOCH
            hf_cfg = cfg.get("hf") or {}
            hf_path = str(hf_cfg.get("path_in_repo") or f"{SUITE}/{method}").rstrip("/") + "/latest.tar.gz"
        except Exception as exc:
            print(f"[WARN] cannot parse config for {method}: {repr(exc)}")

    raw_epoch = None
    completed_epochs_guess = None
    complete = False

    if ckpt_path.exists():
        try:
            raw_epoch = read_ckpt_epoch(ckpt_path)
            if raw_epoch is not None:
                # In this run, ckpt["epoch"]=20 means the 0->20 segment is done.
                completed_epochs_guess = raw_epoch
                complete = raw_epoch >= SEGMENT_END_EPOCH
        except Exception as exc:
            print(f"[WARN] cannot read checkpoint for {method}: {repr(exc)}")

    hf_latest_exists = None
    if hf_files is not None:
        hf_latest_exists = hf_path in hf_files

    return MethodStatus(
        method=method,
        cfg_path=cfg_path,
        save_path=save_path,
        ckpt_path=ckpt_path,
        ckpt_exists=ckpt_path.exists(),
        raw_epoch=raw_epoch,
        completed_epochs_guess=completed_epochs_guess,
        trainer_epochs=trainer_epochs,
        complete=complete,
        hf_path=hf_path,
        hf_latest_exists=hf_latest_exists,
        latest_log=latest_log_for_method(method),
    )


def print_status_table(statuses: list[MethodStatus]) -> None:
    print()
    print("METHOD STATUS")
    print("-" * 190)
    print(
        f"{'method':60s} "
        f"{'ckpt':5s} "
        f"{'raw_ep':6s} "
        f"{'done_ep':7s} "
        f"{'target':6s} "
        f"{'complete':8s} "
        f"{'hf':5s} "
        f"{'latest_log'}"
    )
    print("-" * 190)

    for s in statuses:
        hf_text = "?"
        if s.hf_latest_exists is True:
            hf_text = "YES"
        elif s.hf_latest_exists is False:
            hf_text = "NO"

        print(
            f"{s.method:60s} "
            f"{str(s.ckpt_exists):5s} "
            f"{str(s.raw_epoch):6s} "
            f"{str(s.completed_epochs_guess):7s} "
            f"{str(s.trainer_epochs):6s} "
            f"{str(s.complete):8s} "
            f"{hf_text:5s} "
            f"{str(s.latest_log) if s.latest_log else '-'}"
        )

    print("-" * 190)
    print()


def ensure_hf_defaults(cfg: dict[str, Any], method: str) -> dict[str, Any]:
    hf = cfg.setdefault("hf", {})
    hf.setdefault("enabled", True)
    hf.setdefault("repo_id", REPO_ID)
    hf.setdefault("repo_type", REPO_TYPE)
    hf.setdefault("auto_upload", True)
    hf.setdefault("upload_every_epoch", True)
    hf.setdefault("keep_only_latest", True)
    hf.setdefault("bundle_name", "latest.tar.gz")
    hf.setdefault("path_in_repo", f"{SUITE}/{method}")
    hf.setdefault("squash_after_upload", True)
    return cfg


def upload_missing_completed(statuses: list[MethodStatus], yes: bool) -> None:
    if not yes:
        raise SystemExit("Refusing to upload without --yes")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing. Source augseg_scheduler.env first.")

    uploaded = 0
    skipped = 0
    failed = 0

    print()
    print("BACKFILL UPLOAD")
    print("=" * 120)

    for s in statuses:
        if not s.complete:
            print(f"[SKIP] {s.method}: not complete, done_ep={s.completed_epochs_guess}, target={s.trainer_epochs}")
            skipped += 1
            continue

        if s.hf_latest_exists is True:
            print(f"[SKIP] {s.method}: already on HF -> {s.hf_path}")
            skipped += 1
            continue

        if s.cfg_path is None:
            print(f"[SKIP] {s.method}: config not found")
            skipped += 1
            continue

        if not s.ckpt_exists:
            print(f"[SKIP] {s.method}: checkpoint not found")
            skipped += 1
            continue

        try:
            cfg = load_yaml(s.cfg_path)
            cfg = ensure_hf_defaults(cfg, s.method)

            manifest = {
                "suite_id": SUITE,
                "method": s.method,
                "raw_epoch": s.raw_epoch,
                "completed_epochs_guess": s.completed_epochs_guess,
                "trainer_epochs": s.trainer_epochs,
                "save_path": str(s.save_path),
                "config_path": str(s.cfg_path),
                "manual_backfill_hf_upload": True,
            }

            print(f"[UPLOAD] {s.method}: done_ep={s.completed_epochs_guess}, hf_path={s.hf_path}")

            maybe_upload_hf_bundle(
                cfg=cfg,
                save_path=str(s.save_path),
                config_path=str(s.cfg_path),
                manifest=manifest,
            )

            uploaded += 1

        except Exception as exc:
            print(f"[FAIL] {s.method}: {repr(exc)}")
            failed += 1

    print("=" * 120)
    print(f"DONE uploaded={uploaded} skipped={skipped} failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Print local checkpoint and HF status.")
    parser.add_argument("--upload-missing-completed", action="store_true", help="Upload completed methods missing HF latest.tar.gz.")
    parser.add_argument("--yes", action="store_true", help="Required for upload mode.")
    args = parser.parse_args()

    if not args.status and not args.upload_missing_completed:
        args.status = True

    token = os.environ.get("HF_TOKEN")
    print("HF_TOKEN_PRESENT =", bool(token))

    api = HfApi(token=token) if token else HfApi()
    hf_files = get_hf_files(api)

    methods = discover_methods()
    statuses = [build_status(method, hf_files) for method in methods]

    print_status_table(statuses)

    if args.upload_missing_completed:
        upload_missing_completed(statuses, yes=args.yes)


if __name__ == "__main__":
    main()
