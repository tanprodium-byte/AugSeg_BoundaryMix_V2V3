#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import copy
import os
import yaml


ROOT = Path.cwd()

SUITE_ID = "voc662_8sc_officialc3_rerun01_trainsemi_wandb_hf_20260701"

SRC_REGISTRY = Path(
    "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml"
)

NEW_REGISTRY = Path(
    f"configs/experiment_registry_{SUITE_ID}.yaml"
)

EXPECTED_METHODS = [
    "s1_saliency_box_direct_cutmix",
    "s1_saliency_box_relocated_cutmix",
    "s1_saliency_box_adaptive_relocated_cutmix",
    "s2_saliency_component_mask_direct_cutmix",
    "s3_saliency_component_mask_direct_plus_v3_d2",
    "c1_csl_official_reliability_replace_confidence",
    "c2_csl_official_reliable_mask_perturbation",
    "c3_csl_official_guided_cutmix_plus_v3_d2",
]

ALLOWED_TOP_LEVEL_DIFFS = {
    "saver",
    "run",
    "wandb",
    "checkpoint",
    "hf",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")

    return data


def strip_allowed_blocks(cfg: dict) -> dict:
    copied = copy.deepcopy(cfg)

    for key in ALLOWED_TOP_LEVEL_DIFFS:
        copied.pop(key, None)

    return copied


def safe_token(value: str) -> str:
    import re

    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return token.strip("_") or "unknown"


def runtime_profile_from_cfg(cfg: dict, world_size: int = 1) -> str:
    encoder_type = cfg.get("net", {}).get("encoder", {}).get("type", "")

    if encoder_type == "augseg.models.resnet.resnet101":
        backbone_token = "r101"
    elif encoder_type == "augseg.models.resnet.resnet50":
        backbone_token = "r50"
    else:
        backbone_token = safe_token(str(encoder_type).split(".")[-1])

    crop_size = cfg.get("dataset", {}).get("train", {}).get("crop", {}).get("size", [])

    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        h, w = int(crop_size[0]), int(crop_size[1])
        if h == w:
            crop_token = f"c{h}"
        else:
            crop_token = f"c{h}x{w}"
    else:
        crop_token = "cunknown"

    per_gpu_batch = int(
        cfg.get("dataset", {}).get("train", {}).get("batch_size", 1)
    )
    world_size = int(world_size)
    global_batch = per_gpu_batch * world_size

    return (
        f"{backbone_token}_{crop_token}_"
        f"bs{per_gpu_batch}x{world_size}_gbs{global_batch}"
    )


def expected_save_path(method_name: str, profile: str) -> Path:
    return (
        ROOT
        / "exp_boundary_mix_v2_v3"
        / "reruns"
        / SUITE_ID
        / f"{method_name}_{profile}"
    ).resolve()


def actual_save_path_from_config(config_path: Path, cfg: dict, profile: str) -> Path:
    snapshot_dir = cfg.get("saver", {}).get("snapshot_dir")

    if not snapshot_dir:
        raise ValueError(f"missing saver.snapshot_dir in {config_path}")

    auto_profile_dir = bool(cfg.get("saver", {}).get("auto_profile_dir", False))

    if auto_profile_dir:
        snapshot_dir = f"{snapshot_dir}_{profile}"

    return (config_path.parent / snapshot_dir).resolve()


def audit_hf(method_name: str, cfg: dict) -> bool:
    hf = cfg.get("hf", {})
    expected_path = f"{SUITE_ID}/{method_name}"

    return (
        hf.get("enabled") is True
        and hf.get("repo_id") == "tanprodium/augseg-boundarymix-v2v3-runs"
        and hf.get("repo_type") == "model"
        and hf.get("auto_download") is True
        and hf.get("auto_upload") is True
        and hf.get("upload_every_epoch") is True
        and hf.get("keep_only_latest") is True
        and hf.get("bundle_name") == "latest.tar.gz"
        and hf.get("path_in_repo") == expected_path
        and hf.get("auto_profile_path") is False
        and hf.get("squash_after_upload") is True
    )


def audit_wandb(cfg: dict) -> bool:
    wandb = cfg.get("wandb", {})

    forbidden_keys = {"id", "group", "name", "log_every"}

    return (
        wandb.get("enable") is True
        and wandb.get("project") == "augseg-voc662"
        and wandb.get("entity") == "tanprodium-uit"
        and not any(key in wandb for key in forbidden_keys)
    )


def audit_run(method_name: str, cfg: dict) -> bool:
    run = cfg.get("run", {})

    return (
        run.get("suite_id") == SUITE_ID
        and run.get("name") == method_name
        and int(run.get("log_every", -1)) == 50
        and "id" not in run
    )


def main() -> int:
    print(f"ROOT: {ROOT}")
    print(f"SRC_REGISTRY: {SRC_REGISTRY}")
    print(f"NEW_REGISTRY: {NEW_REGISTRY}")
    print()

    if not SRC_REGISTRY.exists():
        raise FileNotFoundError(f"missing source registry: {SRC_REGISTRY}")

    if not NEW_REGISTRY.exists():
        raise FileNotFoundError(f"missing new registry: {NEW_REGISTRY}")

    src_registry = load_yaml(SRC_REGISTRY)
    new_registry = load_yaml(NEW_REGISTRY)

    src_methods = src_registry.get("methods", [])
    new_methods = new_registry.get("methods", [])

    new_names = [m.get("name") for m in new_methods]

    print(f"new suite_name: {new_registry.get('suite_name')}")
    print(f"method_count: {len(new_methods)}")
    print(f"names_match_expected: {new_names == EXPECTED_METHODS}")
    print()

    if new_names != EXPECTED_METHODS:
        print("EXPECTED:")
        for name in EXPECTED_METHODS:
            print(f"  - {name}")

        print("ACTUAL:")
        for name in new_names:
            print(f"  - {name}")

        raise SystemExit(1)

    src_by_name = {m["name"]: m for m in src_methods}
    new_by_name = {m["name"]: m for m in new_methods}

    all_ok = True

    for method_name in EXPECTED_METHODS:
        print("=" * 110)
        print(f"method: {method_name}")

        src_config_path = Path(src_by_name[method_name]["config"])
        new_config_path = Path(new_by_name[method_name]["config"])

        print(f"src_config: {src_config_path}")
        print(f"new_config: {new_config_path}")
        print(f"new_config_exists: {new_config_path.exists()}")

        if not new_config_path.exists():
            all_ok = False
            continue

        src_cfg = load_yaml(src_config_path)
        new_cfg = load_yaml(new_config_path)

        core_unchanged = (
            strip_allowed_blocks(src_cfg) == strip_allowed_blocks(new_cfg)
        )

        profile = runtime_profile_from_cfg(new_cfg, world_size=1)
        expected_path = expected_save_path(method_name, profile)
        actual_path = actual_save_path_from_config(new_config_path, new_cfg, profile)
        save_path_ok = actual_path == expected_path

        run_ok = audit_run(method_name, new_cfg)
        wandb_ok = audit_wandb(new_cfg)
        hf_ok = audit_hf(method_name, new_cfg)

        print(f"core_training_blocks_unchanged: {core_unchanged}")
        print(f"trainer.epochs: {new_cfg.get('trainer', {}).get('epochs')}")
        print(
            "dataset.train.crop.size:",
            new_cfg.get("dataset", {}).get("train", {}).get("crop", {}).get("size"),
        )
        print(
            "dataset.train.batch_size:",
            new_cfg.get("dataset", {}).get("train", {}).get("batch_size"),
        )

        print(f"runtime_profile: {profile}")
        print(f"saver.snapshot_dir: {new_cfg.get('saver', {}).get('snapshot_dir')}")
        print(f"saver.auto_profile_dir: {new_cfg.get('saver', {}).get('auto_profile_dir')}")
        print(f"expected_save_path: {expected_path}")
        print(f"actual_save_path:   {actual_path}")
        print(f"save_path_ok: {save_path_ok}")

        print(f"run_ok: {run_ok}")
        print(f"wandb_ok: {wandb_ok}")
        print(f"hf_ok: {hf_ok}")

        print("hf summary:")
        print(f"  repo_id: {new_cfg.get('hf', {}).get('repo_id')}")
        print(f"  auto_download: {new_cfg.get('hf', {}).get('auto_download')}")
        print(f"  auto_upload: {new_cfg.get('hf', {}).get('auto_upload')}")
        print(f"  upload_every_epoch: {new_cfg.get('hf', {}).get('upload_every_epoch')}")
        print(f"  keep_only_latest: {new_cfg.get('hf', {}).get('keep_only_latest')}")
        print(f"  path_in_repo: {new_cfg.get('hf', {}).get('path_in_repo')}")
        print(f"  auto_profile_path: {new_cfg.get('hf', {}).get('auto_profile_path')}")
        print(f"  squash_after_upload: {new_cfg.get('hf', {}).get('squash_after_upload')}")

        method_ok = (
            core_unchanged
            and save_path_ok
            and run_ok
            and wandb_ok
            and hf_ok
        )

        print(f"METHOD_OK: {method_ok}")

        if not method_ok:
            all_ok = False

    print()
    print("=" * 110)
    print(f"FINAL_AUDIT_OK: {all_ok}")

    if not all_ok:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())