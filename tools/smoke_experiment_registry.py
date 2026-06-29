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
    parse_epoch_targets,
    parse_args,
    resolve_path,
    validate_full_config,
)

REGISTRY = ROOT / "configs/experiment_registry_voc662_12_methods.yaml"
DIRECT_REGISTRY = ROOT / "configs/experiment_registry_voc662_12_methods_direct_s1s2s3_20_40_60_80.yaml"
CSLFIX_DIRECT_REGISTRY = ROOT / "configs/experiment_registry_voc662_12_methods_direct_s1s2s3_cslfix_20_40_60_80.yaml"
S1_RELOCATED_DIRECT_REGISTRY = ROOT / "configs/experiment_registry_voc662_13_methods_direct_s1_relocated_s1s2s3_cslfix_20_40_60_80.yaml"
S1_ADAPTIVE_RELOCATED_REGISTRY = ROOT / "configs/experiment_registry_voc662_14_methods_s1_relocated_adaptive_s1s2s3_cslfix_20_40_60_80.yaml"


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
    test_registry_shape_with_count(registry, 12)


def test_registry_shape_with_count(registry: dict, expected_count: int) -> None:
    methods = registry["methods"]
    names = [m["name"] for m in methods]
    require(len(methods) == expected_count, f"expected {expected_count} methods, got {len(methods)}")
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


def test_direct_config_matrix(
    registry: dict,
    expected_suite_name: str = "voc662_12_methods_direct_s1s2s3_20_40_60_80",
) -> None:
    require(registry["suite_name"] == expected_suite_name, "direct suite name mismatch")
    by_name = {m["name"]: m for m in registry["methods"]}
    for legacy_name in (
        "s1_saliency_box_cutmix",
        "s2_saliency_component_box_cutmix",
        "s3_saliency_component_box_plus_v3_d2",
    ):
        require(legacy_name not in by_name, f"direct suite must replace legacy method {legacy_name}")

    s1 = cfg(by_name["s1_saliency_box_direct_cutmix"])
    require(s1["saliency_cutmix"]["mode"] == "box", "direct S1 must use box mode")
    require(s1["saliency_cutmix"]["direct_labeled_mix"] is True, "direct S1 must enable direct_labeled_mix")
    require(not enabled(s1, "boundary_compatibility"), "direct S1 must not enable V3")
    require(not enabled(s1, "csl"), "direct S1 must not enable CSL")

    s2 = cfg(by_name["s2_saliency_component_mask_direct_cutmix"])
    require(s2["saliency_cutmix"]["mode"] == "component_mask", "direct S2 must use component_mask mode")
    require(s2["saliency_cutmix"]["direct_labeled_mix"] is True, "direct S2 must enable direct_labeled_mix")
    require(s2["saliency_cutmix"]["paste_mode"] == "mask", "direct S2 must paste masks")
    require(not enabled(s2, "boundary_compatibility"), "direct S2 must not enable V3")
    require(not enabled(s2, "csl"), "direct S2 must not enable CSL")

    s3 = cfg(by_name["s3_saliency_component_mask_direct_plus_v3_d2"])
    require(s3["saliency_cutmix"]["mode"] == "component_mask", "direct S3 must use component_mask mode")
    require(s3["saliency_cutmix"]["direct_labeled_mix"] is True, "direct S3 must enable direct_labeled_mix")
    require(s3["saliency_cutmix"]["paste_mode"] == "mask", "direct S3 must paste masks")
    require(enabled(s3, "boundary_compatibility"), "direct S3 must enable V3")
    require(int(s3["boundary_compatibility"].get("pair_radius")) == 2, "direct S3 must be V3-d2")
    require(float(s3["boundary_compatibility"].get("lambda_bcr")) == 0.01, "direct S3 BCR lambda mismatch")
    require(s3["boundary_compatibility"].get("use_component_gate") is False, "direct S3 must not use component gate")
    require(not enabled(s3, "boundary_component"), "direct S3 must not enable V2")
    require(not enabled(s3, "csl"), "direct S3 must not enable CSL")


def test_cslfix_direct_config_matrix(registry: dict) -> None:
    require(
        registry["suite_name"] == "voc662_12_methods_direct_s1s2s3_cslfix_20_40_60_80",
        "cslfix direct suite name mismatch",
    )
    by_name = {m["name"]: m for m in registry["methods"]}
    require("c1_csl_pseudo_selection" not in by_name, "cslfix suite must not use legacy C1")
    require("c2_csl_random_reliable_masking" not in by_name, "cslfix suite must not use legacy C2")
    require(
        "c1_csl_official_reliability_replace_confidence" in by_name,
        "cslfix suite missing official C1",
    )
    require(
        "c2_csl_official_reliable_mask_perturbation" in by_name,
        "cslfix suite missing official C2",
    )

    test_direct_config_matrix(registry, "voc662_12_methods_direct_s1s2s3_cslfix_20_40_60_80")

    c1 = cfg(by_name["c1_csl_official_reliability_replace_confidence"])
    require(enabled(c1, "csl"), "official C1 must enable CSL")
    require(c1["csl"]["mode"] == "official_reliability_replace_confidence", "official C1 mode mismatch")
    require(c1["csl"]["reliability_mode"] == "official_pcos", "official C1 reliability mismatch")
    require(c1["csl"]["use_csl_for_ce_weight"] is True, "official C1 must weight CE")
    require(c1["csl"]["use_csl_for_mix_confidence"] is True, "official C1 must replace mix confidence")
    require(c1["csl"]["random_mask_reliable"] is False, "official C1 must not random-mask reliable pixels")
    require(c1["csl"]["use_csl_for_cutmix"] is False, "official C1 must not use CSL CutMix")
    require(not enabled(c1, "boundary_compatibility"), "official C1 must not enable V3")
    require(not enabled(c1, "saliency_cutmix"), "official C1 must not enable saliency")

    c2 = cfg(by_name["c2_csl_official_reliable_mask_perturbation"])
    require(enabled(c2, "csl"), "official C2 must enable CSL")
    require(c2["csl"]["mode"] == "official_reliable_mask_perturbation", "official C2 mode mismatch")
    require(c2["csl"]["reliability_mode"] == "official_pcos", "official C2 reliability mismatch")
    require(c2["csl"]["use_csl_for_ce_weight"] is True, "official C2 must weight CE")
    require(c2["csl"]["use_csl_for_mix_confidence"] is True, "official C2 must replace mix confidence")
    require(c2["csl"]["random_mask_reliable"] is True, "official C2 must enable reliable masking flag")
    require(c2["csl"]["perturb_input"] is True, "official C2 must perturb input")
    require(c2["csl"]["use_csl_for_cutmix"] is False, "official C2 must not use CSL CutMix")
    require(not enabled(c2, "boundary_compatibility"), "official C2 must not enable V3")
    require(not enabled(c2, "saliency_cutmix"), "official C2 must not enable saliency")


def test_s1_relocated_direct_config_matrix(registry: dict) -> None:
    require(
        registry["suite_name"] == "voc662_13_methods_direct_s1_relocated_s1s2s3_cslfix_20_40_60_80",
        "s1 relocated direct suite name mismatch",
    )
    by_name = {m["name"]: m for m in registry["methods"]}
    require("s1_saliency_box_relocated_cutmix" in by_name, "s1 relocated suite missing relocated S1")
    require("c1_csl_official_reliability_replace_confidence" in by_name, "s1 relocated suite missing official C1")
    require("c2_csl_official_reliable_mask_perturbation" in by_name, "s1 relocated suite missing official C2")

    s1_relocated = cfg(by_name["s1_saliency_box_relocated_cutmix"])
    require(enabled(s1_relocated, "saliency_cutmix"), "relocated S1 must enable saliency_cutmix")
    require(s1_relocated["saliency_cutmix"]["mode"] == "box", "relocated S1 must use box mode")
    require(s1_relocated["saliency_cutmix"]["direct_labeled_mix"] is True, "relocated S1 must enable direct_labeled_mix")
    require(
        s1_relocated["saliency_cutmix"]["direct_paste_policy"] == "random_target",
        "relocated S1 must use random_target direct paste policy",
    )
    require(not enabled(s1_relocated, "boundary_compatibility"), "relocated S1 must not enable V3")
    require(not enabled(s1_relocated, "boundary_component"), "relocated S1 must not enable V2")
    require(not enabled(s1_relocated, "csl"), "relocated S1 must not enable CSL")


def test_s1_adaptive_relocated_config_matrix(registry: dict) -> None:
    require(
        registry["suite_name"] == "voc662_14_methods_s1_relocated_adaptive_s1s2s3_cslfix_20_40_60_80",
        "s1 adaptive relocated suite name mismatch",
    )
    by_name = {m["name"]: m for m in registry["methods"]}
    for required_name in (
        "s1_saliency_box_direct_cutmix",
        "s1_saliency_box_relocated_cutmix",
        "s1_saliency_box_adaptive_relocated_cutmix",
        "s2_saliency_component_mask_direct_cutmix",
        "s3_saliency_component_mask_direct_plus_v3_d2",
        "c1_csl_official_reliability_replace_confidence",
        "c2_csl_official_reliable_mask_perturbation",
        "c3_csl_guided_cutmix_plus_v3_d2",
    ):
        require(required_name in by_name, f"14-method suite missing {required_name}")

    s1_relocated = cfg(by_name["s1_saliency_box_relocated_cutmix"])
    require(
        s1_relocated["saliency_cutmix"].get("direct_confidence_gate", False) is False,
        "relocated S1 must remain no-gate",
    )

    s1_adaptive = cfg(by_name["s1_saliency_box_adaptive_relocated_cutmix"])
    require(enabled(s1_adaptive, "saliency_cutmix"), "adaptive relocated S1 must enable saliency_cutmix")
    require(s1_adaptive["saliency_cutmix"]["mode"] == "box", "adaptive relocated S1 must use box mode")
    require(s1_adaptive["saliency_cutmix"]["direct_labeled_mix"] is True, "adaptive relocated S1 must enable direct_labeled_mix")
    require(
        s1_adaptive["saliency_cutmix"]["direct_paste_policy"] == "random_target",
        "adaptive relocated S1 must use random_target direct paste policy",
    )
    require(
        s1_adaptive["saliency_cutmix"]["direct_confidence_gate"] is True,
        "adaptive relocated S1 must enable direct confidence gate",
    )
    require(not enabled(s1_adaptive, "boundary_compatibility"), "adaptive relocated S1 must not enable V3")
    require(not enabled(s1_adaptive, "boundary_component"), "adaptive relocated S1 must not enable V2")
    require(not enabled(s1_adaptive, "csl"), "adaptive relocated S1 must not enable CSL")


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
            "--launcher",
            "python-module",
        ]
    )
    commands = dry_run_commands(registry, args)
    require(len(commands) == 12, f"expected 12 dry-run commands, got {len(commands)}")
    for cmd in commands:
        joined = " ".join(cmd)
        require("train_semi.py" in cmd, f"command missing train_semi.py: {joined}")
        require("--config" in cmd, f"command missing --config: {joined}")
        require(cmd[0] == sys.executable, f"python-module launcher must use sys.executable: {joined}")
        require(cmd[1:4] == ["-m", "torch.distributed.run", "--standalone"], f"unexpected launcher: {joined}")


def test_segment_dry_run(registry: dict) -> None:
    args = parse_args(
        [
            "--registry",
            str(REGISTRY),
            "--gpu",
            "0",
            "--mode",
            "dry-run",
            "--schedule-mode",
            "segments",
            "--epoch-targets",
            "20,40,60,80",
            "--suite-name",
            "voc662_12_methods_segments_20_40_60_80",
            "--nproc-per-node",
            "1",
            "--launcher",
            "python-module",
        ]
    )
    require(parse_epoch_targets(args.epoch_targets) == [20, 40, 60, 80], "epoch targets did not parse")
    commands = dry_run_commands(registry, args)
    require(len(commands) == 12, f"expected 12 segment dry-run commands, got {len(commands)}")
    first = commands[0]
    require("--config" in first, "segment command missing --config")
    cfg_path = Path(first[first.index("--config") + 1])
    segment_cfg = load_yaml(cfg_path)
    original_cfg = cfg(registry["methods"][0])
    require(segment_cfg["trainer"]["epochs"] == 20, "segment temp config must target epoch 20")
    for key in ("hf", "saver", "wandb"):
        require(segment_cfg.get(key) == original_cfg.get(key), f"segment temp config changed {key}")


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
            "--launcher",
            "python-module",
            "--init-queue",
            "--status",
        ]
    )
    require(args.queue_backend == "postgres", "postgres backend arg did not parse")
    require(args.init_queue is True, "init_queue arg did not parse")
    require(args.status is True, "status arg did not parse")
    require(args.worker_id == "supermaster:gpu0", "worker_id arg did not parse")
    require(args.launcher == "python-module", "launcher arg did not parse")
    seg_args = parse_args(
        [
            "--registry",
            str(REGISTRY),
            "--mode",
            "full",
            "--schedule-mode",
            "segments",
            "--epoch-targets",
            "20,40,60,80",
            "--suite-name",
            "voc662_12_methods_segments_20_40_60_80",
            "--queue-backend",
            "postgres",
            "--db-url-env",
            "AUGSEG_SCHEDULER_DB_URL",
            "--worker-id",
            "islab-server3:gpu0",
            "--server-name",
            "islab-server3",
            "--launcher",
            "python-module",
            "--init-queue",
        ]
    )
    require(seg_args.schedule_mode == "segments", "segment schedule arg did not parse")
    require(seg_args.suite_name == "voc662_12_methods_segments_20_40_60_80", "suite_name arg did not parse")


def main() -> int:
    registry = load_registry(REGISTRY)
    test_registry_shape(registry)
    test_config_matrix(registry)
    test_dry_run(registry)
    test_segment_dry_run(registry)
    test_postgres_cli_args()
    direct_registry = load_registry(DIRECT_REGISTRY)
    test_registry_shape(direct_registry)
    test_direct_config_matrix(direct_registry)
    cslfix_direct_registry = load_registry(CSLFIX_DIRECT_REGISTRY)
    test_registry_shape(cslfix_direct_registry)
    test_cslfix_direct_config_matrix(cslfix_direct_registry)
    s1_relocated_direct_registry = load_registry(S1_RELOCATED_DIRECT_REGISTRY)
    test_registry_shape_with_count(s1_relocated_direct_registry, 13)
    test_s1_relocated_direct_config_matrix(s1_relocated_direct_registry)
    s1_adaptive_relocated_registry = load_registry(S1_ADAPTIVE_RELOCATED_REGISTRY)
    test_registry_shape_with_count(s1_adaptive_relocated_registry, 14)
    test_s1_adaptive_relocated_config_matrix(s1_adaptive_relocated_registry)
    print("VOC662 registry smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
