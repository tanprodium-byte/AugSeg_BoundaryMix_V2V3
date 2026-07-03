#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import copy
import yaml


SUITE_ID = "voc662_8sc_officialc3_rerun01_trainsemi_wandb_hf_20260701"

SRC_REGISTRY = Path(
    "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml"
)

RERUN_CONFIG_ROOT = (
    Path("exps/boundary_mix_v2_v3/voc_semi662_reruns") / SUITE_ID
)

OUT_REGISTRY = Path("configs") / f"experiment_registry_{SUITE_ID}.yaml"

HF_REPO_ID = "tanprodium/augseg-boundarymix-v2v3-runs"

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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")

    return data


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def validate_source_config(method_name: str, cfg: dict) -> None:
    epochs = int(cfg.get("trainer", {}).get("epochs", 0))
    if epochs < 80:
        raise ValueError(
            f"{method_name}: trainer.epochs={epochs}, expected >= 80"
        )

    crop = cfg.get("dataset", {}).get("train", {}).get("crop", {}).get("size")
    if crop != [321, 321]:
        raise ValueError(
            f"{method_name}: crop={crop}, expected [321, 321]"
        )

    batch_size = int(
        cfg.get("dataset", {}).get("train", {}).get("batch_size", 0)
    )
    if batch_size != 8:
        raise ValueError(
            f"{method_name}: batch_size={batch_size}, expected 8"
        )


def make_rerun_config(method_name: str, src_config_path: Path) -> dict:
    cfg = load_yaml(src_config_path)

    validate_source_config(method_name, cfg)

    cfg.setdefault("saver", {})
    cfg["saver"]["snapshot_dir"] = (
        f"../../../../../exp_boundary_mix_v2_v3/reruns/"
        f"{SUITE_ID}/{method_name}"
    )
    cfg["saver"]["auto_profile_dir"] = True

    cfg["run"] = {
        "suite_id": SUITE_ID,
        "name": method_name,
        "log_every": 50,
    }

    cfg["wandb"] = {
        "enable": True,
        "project": "augseg-voc662",
        "entity": "tanprodium-uit",
        "tags": [
            "voc662",
            "official_c3",
            "rerun01",
            "train_semi_wandb_hf",
        ],
    }

    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["auto_resume"] = True
    cfg["checkpoint"]["save_latest"] = True
    cfg["checkpoint"]["save_best"] = True

    cfg["hf"] = {
        "enabled": True,
        "repo_id": HF_REPO_ID,
        "repo_type": "model",
        "auto_download": True,
        "auto_upload": True,
        "upload_every_epoch": True,
        "keep_only_latest": True,
        "bundle_name": "latest.tar.gz",
        "path_in_repo": f"{SUITE_ID}/{method_name}",
        "auto_profile_path": False,
        "squash_after_upload": True,
    }

    return cfg


def main() -> int:
    if not SRC_REGISTRY.exists():
        raise FileNotFoundError(f"missing source registry: {SRC_REGISTRY}")

    registry = load_yaml(SRC_REGISTRY)
    methods = registry.get("methods", [])

    actual_names = [m.get("name") for m in methods]
    if actual_names != EXPECTED_METHODS:
        raise ValueError(
            "source registry method order/name mismatch\n"
            f"expected={EXPECTED_METHODS}\n"
            f"actual={actual_names}"
        )

    new_registry = copy.deepcopy(registry)
    new_registry["suite_name"] = SUITE_ID

    new_methods = []

    for method in methods:
        method_name = method["name"]
        src_config_path = Path(method["config"])

        if not src_config_path.exists():
            raise FileNotFoundError(
                f"missing source config for {method_name}: {src_config_path}"
            )

        cfg = make_rerun_config(method_name, src_config_path)

        out_config_path = RERUN_CONFIG_ROOT / method_name / "config.yaml"
        dump_yaml(out_config_path, cfg)

        new_method = copy.deepcopy(method)
        new_method["config"] = str(out_config_path)
        new_methods.append(new_method)

        print(f"[OK] {method_name}")
        print(f"     src: {src_config_path}")
        print(f"     out: {out_config_path}")
        print(f"     snapshot_dir: {cfg['saver']['snapshot_dir']}")
        print(f"     hf.path_in_repo: {cfg['hf']['path_in_repo']}")

    new_registry["methods"] = new_methods
    dump_yaml(OUT_REGISTRY, new_registry)

    print()
    print(f"[OK] wrote registry: {OUT_REGISTRY}")
    print(f"[OK] suite_id: {SUITE_ID}")
    print(f"[OK] method_count: {len(new_methods)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())