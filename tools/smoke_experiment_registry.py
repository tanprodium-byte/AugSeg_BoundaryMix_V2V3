#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_experiment_suite import (  # noqa: E402
    dry_run_commands,
    load_registry,
    load_yaml,
    parse_args,
    resolve_path,
    validate_full_config,
)

REGISTRY = ROOT / "configs/experiment_registry_voc662_12_methods.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def cfg(method: dict) -> dict:
    return load_yaml(resolve_path(method["config"]))


def enabled(config: dict, section: str) -> bool:
    value = config.get(section, {}).get("enabled", False)
    require(isinstance(value, bool), f"{section}.enabled must be bool, got {value!r}")
    return value


def test_registry_shape(registry: dict) -> None:
    methods = registry["methods"]
    names = [m["name"] for m in methods]
    require(len(methods) == 12, f"expected 12 methods, got {len(methods)}")
    require(len(names) == len(set(names)), "method names must be unique")
    for method in methods:
        require(resolve_path(method["config"]).is_file(), f"missing config: {method['config']}")
        crop, global_batch = validate_full_config(method, registry, nproc_per_node=1)
        require(crop == [321, 321], f"{method['name']} crop mismatch: {crop}")
        require(global_batch == 8, f"{method['name']} global batch mismatch: {global_batch}")


def test_config_matrix(registry: dict) -> None:
    by_name = {m["name"]: m for m in registry["methods"]}
    for name in ("s1_saliency_box_cutmix", "s2_saliency_component_box_cutmix"):
        config = cfg(by_name[name])
        require(enabled(config, "saliency_cutmix"), f"{name} must enable saliency_cutmix")
        require(not enabled(config, "boundary_compatibility"), f"{name} must not enable V3")
        require(not enabled(config, "csl"), f"{name} must not enable CSL")

    for name in ("c1_csl_pseudo_selection", "c2_csl_random_reliable_masking"):
        config = cfg(by_name[name])
        require(enabled(config, "csl"), f"{name} must enable CSL")
        require(config["csl"].get("use_csl_for_ce_weight") is True, f"{name} must weight CE with CSL")
        require(config["csl"].get("use_csl_for_cutmix") is False, f"{name} must not use CSL CutMix")
        require(not enabled(config, "boundary_compatibility"), f"{name} must not enable V3")
        require(not enabled(config, "saliency_cutmix"), f"{name} must not enable saliency")

    s3 = cfg(by_name["s3_saliency_component_box_plus_v3_d2"])
    require(enabled(s3, "saliency_cutmix"), "S3 must enable saliency_cutmix")
    require(enabled(s3, "boundary_compatibility"), "S3 must enable V3")
    require(int(s3["boundary_compatibility"].get("pair_radius")) == 2, "S3 must be V3-d2")
    require(not enabled(s3, "csl"), "S3 must not enable CSL")

    c3 = cfg(by_name["c3_csl_guided_cutmix_plus_v3_d2"])
    require(enabled(c3, "csl"), "C3 must enable CSL")
    require(enabled(c3, "csl_cutmix"), "C3 must enable CSL CutMix")
    require(enabled(c3, "boundary_compatibility"), "C3 must enable V3")
    require(int(c3["boundary_compatibility"].get("pair_radius")) == 2, "C3 must be V3-d2")
    require(c3["csl"].get("use_csl_for_ce_weight") is False, "C3 use_csl_for_ce_weight must be false")

    for method in registry["methods"]:
        config = cfg(method)
        if not method["name"].startswith("c"):
            require(not enabled(config, "csl"), f"{method['name']} must not enable CSL")


def test_dry_run(registry: dict) -> None:
    args = parse_args(
        [
            "--registry",
            str(REGISTRY),
            "--gpu",
            "0",
            "--mode",
            "dry-run",
            "--nproc-per-node",
            "1",
        ]
    )
    commands = dry_run_commands(registry, args)
    require(len(commands) == 12, f"expected 12 dry-run commands, got {len(commands)}")
    for cmd in commands:
        joined = " ".join(cmd)
        require("train_semi.py" in cmd, f"command missing train_semi.py: {joined}")
        require("--config" in cmd, f"command missing --config: {joined}")


def test_postgres_cli_args() -> None:
    args = parse_args(
        [
            "--registry",
            str(REGISTRY),
            "--mode",
            "full",
            "--queue-backend",
            "postgres",
            "--db-url-env",
            "AUGSEG_SCHEDULER_DB_URL",
            "--worker-id",
            "supermaster:gpu0",
            "--server-name",
            "supermaster",
            "--init-queue",
            "--status",
        ]
    )
    require(args.queue_backend == "postgres", "postgres backend arg did not parse")
    require(args.init_queue is True, "init_queue arg did not parse")
    require(args.status is True, "status arg did not parse")
    require(args.worker_id == "supermaster:gpu0", "worker_id arg did not parse")


def main() -> int:
    registry = load_registry(REGISTRY)
    test_registry_shape(registry)
    test_config_matrix(registry)
    test_dry_run(registry)
    test_postgres_cli_args()
    print("VOC662 12-method registry smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
