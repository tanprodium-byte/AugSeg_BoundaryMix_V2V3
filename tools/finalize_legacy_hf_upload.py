#!/usr/bin/env python3
"""Strict finalization for the 13 reviewed Legacy RTX5090 configs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path("exps/boundary_mix_v2_v3/voc_semi662_legacy_c321_rtx5090")
OUTPUT = "exp_boundary_mix_v2_v3_legacy_c321_rtx5090"
PARENT = "bbe2224c50db2c277e5d76dbb64f6527489d5b5c"
SUFFIX = "legacy_c321_rerun01_rtx5090_r101_c321_bs8x1_gbs8_seed2_attempt01"
REPO = "tanprodium/augseg-legacy-c321-rerun01-rtx5090"
PROJECT = "AugsegLegacyC321Rerun01-RTX5090"
ENTITY = "tanprodium-uit"
METHODS = (
    "baseline_augseg_fair80_rerun01", "v3_js_bcr_d2", "v3_js_bcr_d1",
    "v3_js_bcr_d3", "v3_d2_affinity_bce", "v3_d2_teacher_feature_gate",
    "v3_d2_teacher_relation_consistency", "s2_saliency_component_box_cutmix",
    "s3_saliency_component_box_plus_v3_d2", "s2_saliency_component_mask_direct_cutmix",
    "s3_saliency_component_mask_direct_plus_v3_d2",
    "c3_csl_guided_cutmix_plus_v3_d2", "c3_csl_official_guided_cutmix_plus_v3_d2",
)
REQUIRED = ("config.yaml", "run_id.txt", "ckpt.pth", "ckpt_best.pth",
            "epoch_metrics.csv", "iter_metrics.csv", "manifest.json")

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def sha_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()

def sha_file(path):
    with Path(path).open("rb") as stream:
        return sha_stream(stream)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def reviewed_config(value):
    raw = str(value)
    if any(ch in raw for ch in "*?[]"):
        raise ValueError("globs forbidden")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    allowed = {ROOT / BASE / m / "config.yaml": m for m in METHODS}
    if path not in allowed or path.is_symlink():
        raise ValueError("config outside reviewed cohort")
    method = allowed[path]
    cfg = yaml.safe_load(path.read_text())
    identity = f"{method}_{SUFFIX}"
    hf_path = f"boundary_mix_v2_v3/voc_semi662/legacy_c321_rerun01/rtx5090/{method}/seed2/attempt01/r101_c321_bs8x1_gbs8/latest.tar.gz"
    require(cfg["name"] == cfg["run"]["name"] == cfg["wandb"]["name"] == identity, "run identity mismatch")
    require(not cfg["run"].get("suite_id"), "historical suite ID present")
    require(cfg["wandb"]["entity"] == ENTITY and cfg["wandb"]["project"] == PROJECT, "W&B destination mismatch")
    require(cfg["wandb"]["enable"] is True and cfg["wandb"]["resume"] == "never", "W&B run control mismatch")
    require(cfg["saver"]["snapshot_dir"] == f"../../../../{OUTPUT}/{identity}", "snapshot identity mismatch")
    require(cfg["saver"]["auto_profile_dir"] is False and cfg["saver"]["auto_resume"] is False, "saver fresh controls mismatch")
    require(cfg["checkpoint"]["auto_resume"] is False, "checkpoint auto-resume enabled")
    hf = cfg["hf"]
    require(hf["repo_id"] == REPO and hf["repo_type"] == "model", "HF destination mismatch")
    require(hf["path_in_repo"] == hf_path and hf["bundle_name"] == "latest.tar.gz", "HF path mismatch")
    require(hf["auto_profile_path"] is False and hf["auto_download"] is False, "HF fresh controls mismatch")
    require(hf["enabled"] and hf["auto_upload"] and not hf["upload_every_epoch"], "HF upload policy mismatch")
    require(hf["keep_only_latest"] and hf["squash_after_upload"], "HF retention policy mismatch")
    save = (path.parent / cfg["saver"]["snapshot_dir"]).resolve()
    expected_save = (ROOT / OUTPUT / identity).absolute()
    if save != expected_save or save.is_symlink():
        raise ValueError("noncanonical output path")
    if git("branch", "--show-current") != "experiment/legacy-c321-rerun01":
        raise ValueError("wrong branch")
    head = git("rev-parse", "HEAD")
    if git("rev-list", "--count", f"{PARENT}..{head}") != "1" or git("rev-parse", f"{head}^") != PARENT:
        raise ValueError("Legacy parent or commit count drift")
    subprocess.check_call(["git", "ls-files", "--error-unmatch", "--", str(path.relative_to(ROOT))], cwd=ROOT, stdout=subprocess.DEVNULL)
    if subprocess.call(["git", "diff", "--quiet", "--"], cwd=ROOT) or subprocess.call(["git", "diff", "--cached", "--quiet", "--"], cwd=ROOT):
        raise ValueError("dirty tracked tree")
    return method, path, cfg, save, head

def validate_local(config_path):
    method, path, cfg, save, head = reviewed_config(config_path)
    if not save.is_dir():
        raise ValueError("output directory absent")
    for name in REQUIRED:
        item = save / name
        if item.is_symlink() or not item.is_file() or item.stat().st_size == 0:
            raise ValueError(f"missing or invalid {name}")
    if (save / "config.yaml").read_bytes() != path.read_bytes():
        raise ValueError("saved config differs")
    with (save / "epoch_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 80 or [int(row["epoch"]) for row in rows] != list(range(80)):
        raise ValueError("epoch history is not exactly 0..79")
    manifest = json.loads((save / "manifest.json").read_text())
    run_id = (save / "run_id.txt").read_text().strip()
    if not re.fullmatch(r"legacy5090_[0-9a-f]{24}", run_id) or manifest.get("epoch") != 79 or manifest.get("run_id") != run_id:
        raise ValueError("manifest epoch or run ID mismatch")
    if manifest.get("run_name") != cfg["run"]["name"] or manifest.get("world_size") != 1:
        raise ValueError("manifest identity mismatch")
    if manifest.get("git_commit") not in (head, head[:7]):
        raise ValueError("manifest Git commit mismatch")
    if Path(manifest.get("save_path", "")).resolve() != save:
        raise ValueError("manifest save path mismatch")
    import torch
    for name in ("ckpt.pth", "ckpt_best.pth"):
        checkpoint = torch.load(save / name, map_location="cpu", weights_only=False)
        epoch = checkpoint.get("epoch")
        if name == "ckpt.pth" and epoch != 80:
            raise ValueError("latest checkpoint epoch is not 80")
        if name == "ckpt_best.pth" and not (1 <= int(epoch) <= 80):
            raise ValueError("best checkpoint epoch invalid")
        if checkpoint.get("run_id") != run_id:
            raise ValueError("checkpoint run ID mismatch")
        saved = checkpoint.get("cfg", {})
        if saved.get("name") != cfg["name"]:
            raise ValueError("checkpoint config name mismatch")
        for group, fields in {
            "run": ("name",), "wandb": ("name", "project", "entity"),
            "hf": ("repo_id", "path_in_repo"),
        }.items():
            for field in fields:
                if saved.get(group, {}).get(field) != cfg[group][field]:
                    raise ValueError("checkpoint config identity mismatch")
        args = checkpoint.get("args", {})
        if int(args.get("seed", -1)) != 2 or Path(args.get("config", "")).resolve() != path:
            raise ValueError("checkpoint launch identity mismatch")
    return method, path, cfg, save, head, manifest, run_id

def verify_members(bundle, save, path):
    with tarfile.open(bundle, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(n.startswith("/") or ".." in Path(n).parts for n in names):
            raise ValueError("unsafe or duplicate archive member")
        for name in REQUIRED:
            if name not in names or not tar.getmember(name).isfile():
                raise ValueError(f"archive missing {name}")
            source = path if name == "config.yaml" else save / name
            with tar.extractfile(name) as member_stream:
                member_hash = sha_stream(member_stream)
            if member_hash != sha_file(source):
                raise ValueError(f"archive hash mismatch: {name}")

def verify_remote(cfg, save, path, api, downloader):
    from util.hf_auto import _default_path_in_repo
    remote_path = _default_path_in_repo(cfg["hf"], save)
    if remote_path != cfg["hf"]["path_in_repo"]:
        raise ValueError("HF raw/effective path mismatch")
    local = save / "_hf_bundle" / cfg["hf"]["bundle_name"]
    verify_members(local, save, path)
    items = api.get_paths_info(repo_id=REPO, paths=[remote_path], repo_type="model", expand=True)
    if len(items) != 1 or items[0].path != remote_path:
        raise ValueError("remote artifact absent")
    local_size = local.stat().st_size
    local_oid = sha_file(local)
    lfs = getattr(items[0], "lfs", None) or {}
    if items[0].size != local_size or lfs.get("sha256") != local_oid:
        raise ValueError("remote size or LFS OID mismatch")
    remote = Path(downloader(repo_id=REPO, filename=remote_path, repo_type="model", force_download=True))
    if remote.stat().st_size != local_size or sha_file(remote) != local_oid:
        raise ValueError("remote content mismatch")
    verify_members(remote, save, path)
    return {"path": remote_path, "size": local_size, "lfs_oid": local_oid}

def finalize(config_path):
    _, path, cfg, save, _, manifest, _ = validate_local(config_path)
    from util import hf_auto
    result = hf_auto.maybe_upload_hf_bundle(cfg=cfg, save_path=save, config_path=path, manifest=manifest)
    if not result.get("uploaded"):
        raise RuntimeError("HF upload did not succeed")
    from huggingface_hub import HfApi, hf_hub_download
    return verify_remote(cfg, save, path, HfApi(), hf_hub_download)

def check_wandb(config_path):
    _, _, cfg, _, _, _, run_id = validate_local(config_path)
    import wandb
    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{run_id}")
    if run.state != "finished" or run.name != cfg["wandb"]["name"]:
        raise ValueError("W&B run is not finished with expected name")
    return run_id

def check_fresh(config_path):
    _, path, cfg, save, _ = reviewed_config(config_path)
    if save.exists() or (path.parent / "log").exists():
        raise ValueError("output or W&B log directory already exists")
    from util.hf_auto import _default_path_in_repo
    remote_path = _default_path_in_repo(cfg["hf"], save)
    if remote_path != cfg["hf"]["path_in_repo"]:
        raise ValueError("HF raw/effective path mismatch")
    from huggingface_hub import HfApi
    try:
        found = HfApi().get_paths_info(repo_id=REPO, paths=[remote_path], repo_type="model", expand=True)
    except Exception as exc:
        raise RuntimeError("cannot verify remote HF path freshness") from exc
    if found:
        raise ValueError("remote HF artifact already exists")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="one literal reviewed cohort config")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-local", action="store_true")
    modes.add_argument("--check-wandb", action="store_true")
    modes.add_argument("--check-fresh", action="store_true")
    args = parser.parse_args()
    try:
        if args.check_local:
            validate_local(args.config)
        elif args.check_wandb:
            check_wandb(args.config)
        elif args.check_fresh:
            check_fresh(args.config)
        else:
            print(json.dumps(finalize(args.config), sort_keys=True))
    except Exception as exc:
        print(f"Legacy finalization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
