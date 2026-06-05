#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "exps/boundary_mix_v2_v3/voc_semi662_c321_gbs8"
BANNED_PATH_MARKERS = (
    "/home/jupyter-iec2024iot04",
    "/home/islabworker3",
    "/mnt/chautm",
    ".codex_smoke",
    ".scheduler_runs",
)
CONFIGS = {
    "baseline": "baseline-voc662-r101-c321-bs8",
    "v2_component_weighting": "boundarymix-v2-component-weighting-voc662-r101-c321-bs8",
    "v2_v3_best_template": "boundarymix-v2-v3-best-template-voc662-r101-c321-bs8",
    "v3_js_bcr_d1": "boundarymix-v3-js-bcr-d1-voc662-r101-c321-bs8",
    "v3_js_bcr_d2": "boundarymix-v3-js-bcr-d2-voc662-r101-c321-bs8",
    "v3_js_bcr_d3": "boundarymix-v3-js-bcr-d3-voc662-r101-c321-bs8",
}


def fail(message: str) -> None:
    print(f"VERIFY_FAIL {message}")
    raise SystemExit(1)


def load_config(name: str) -> dict[str, Any]:
    path = PROFILE_ROOT / name / "config.yaml"
    if not path.is_file():
        fail(f"missing_config {path}")
    text = path.read_text()
    for marker in BANNED_PATH_MARKERS:
        if marker in text:
            fail(f"{name} banned_path_marker_in_config marker={marker} path={path}")
    return yaml.safe_load(text)


def require_portable_path(name: str, field: str, value: Any, *, allow_empty: bool = False) -> str:
    if value is None:
        if allow_empty:
            return ""
        fail(f"{name} missing_path_field {field}")
    raw = str(value)
    if raw == "":
        if allow_empty:
            return raw
        fail(f"{name} empty_path_field {field}")
    for marker in BANNED_PATH_MARKERS:
        if marker in raw:
            fail(f"{name} banned_path_marker field={field} marker={marker} value={raw}")
    if Path(raw).is_absolute():
        fail(f"{name} absolute_path field={field} value={raw}")
    return raw


def resolve_from_repo(path_text: str) -> Path:
    return (ROOT / path_text).resolve()


def require_existing_path(name: str, field: str, value: Any, *, kind: str) -> Path:
    raw = require_portable_path(name, field, value)
    resolved = resolve_from_repo(raw)
    if kind == "dir":
        ok = resolved.is_dir()
    elif kind == "file":
        ok = resolved.is_file()
    else:
        raise ValueError(f"unknown path kind: {kind}")
    if not ok:
        fail(f"{name} missing_{kind} field={field} value={raw} resolved={resolved}")
    return resolved


def require_input_paths(name: str, cfg: dict[str, Any]) -> None:
    train = cfg["dataset"]["train"]
    val = cfg["dataset"]["val"]
    require_existing_path(name, "dataset.train.data_root", train.get("data_root"), kind="dir")
    require_existing_path(name, "dataset.val.data_root", val.get("data_root"), kind="dir")
    train_list = require_existing_path(name, "dataset.train.data_list", train.get("data_list"), kind="file")
    require_existing_path(name, "dataset.val.data_list", val.get("data_list"), kind="file")
    require_existing_path(name, "net.encoder.pretrain", cfg["net"]["encoder"].get("pretrain"), kind="file")

    unlabeled = Path(str(train.get("data_list")).replace("labeled.txt", "unlabeled.txt"))
    if str(unlabeled) == str(train.get("data_list")):
        fail(f"{name} cannot_derive_unlabeled_split field=dataset.train.data_list value={train.get('data_list')}")
    require_existing_path(name, "dataset.train.data_list.derived_unlabeled", str(unlabeled), kind="file")

    expected = {
        "dataset.train.data_root": ROOT / "data/VOC2012",
        "dataset.val.data_root": ROOT / "data/VOC2012",
        "dataset.train.data_list": ROOT / "data/splitsall/pascal_u2pl/662/labeled.txt",
        "dataset.train.data_list.derived_unlabeled": ROOT / "data/splitsall/pascal_u2pl/662/unlabeled.txt",
        "dataset.val.data_list": ROOT / "data/splitsall/pascal_u2pl/val.txt",
    }
    actual = {
        "dataset.train.data_root": resolve_from_repo(str(train.get("data_root"))),
        "dataset.val.data_root": resolve_from_repo(str(val.get("data_root"))),
        "dataset.train.data_list": train_list,
        "dataset.train.data_list.derived_unlabeled": resolve_from_repo(str(unlabeled)),
        "dataset.val.data_list": resolve_from_repo(str(val.get("data_list"))),
    }
    for field, expected_path in expected.items():
        if actual[field] != expected_path.resolve():
            fail(f"{name} wrong_portable_path field={field} expected={expected_path} actual={actual[field]}")


def require_common(name: str, cfg: dict[str, Any], slug: str) -> None:
    train = cfg["dataset"]["train"]
    val = cfg["dataset"]["val"]
    if cfg["dataset"]["type"] != "pascal_semi" or cfg["dataset"].get("n_sup") != 662:
        fail(f"{name} dataset_not_voc_semi_662")
    if train["data_list"] != "./data/splitsall/pascal_u2pl/662/labeled.txt":
        fail(f"{name} wrong_train_split {train['data_list']}")
    if train["batch_size"] != 8:
        fail(f"{name} train_batch_size={train['batch_size']}")
    if train["crop"]["size"] != [321, 321]:
        fail(f"{name} crop_size={train['crop']['size']}")
    if train.get("resize_base_size") != 500:
        fail(f"{name} resize_base_size={train.get('resize_base_size')}")
    if train.get("rand_resize") != [0.5, 2.0]:
        fail(f"{name} rand_resize={train.get('rand_resize')}")
    if val["batch_size"] != 1:
        fail(f"{name} val_batch_size={val['batch_size']}")
    if cfg["net"]["encoder"]["type"] != "augseg.models.resnet.resnet101":
        fail(f"{name} encoder={cfg['net']['encoder']['type']}")

    require_portable_path(name, "saver.snapshot_dir", cfg.get("saver", {}).get("snapshot_dir"), allow_empty=True)
    require_input_paths(name, cfg)

    hf_path = cfg.get("hf", {}).get("path_in_repo", "")
    require_portable_path(name, "hf.path_in_repo", hf_path)
    if "c321" not in hf_path or slug not in hf_path or not hf_path.endswith("/latest.tar.gz"):
        fail(f"{name} bad_hf_path {hf_path}")
    if "voc_semi662_c321_gbs8" not in hf_path and "voc662-c321-gbs8" not in hf_path:
        fail(f"{name} hf_path_missing_c321_profile {hf_path}")


def require_modules(name: str, cfg: dict[str, Any]) -> None:
    boundary_mix_enabled = bool(cfg.get("boundary_mix", {}).get("enabled", False))
    component_enabled = bool(cfg.get("boundary_component", {}).get("enabled", False))
    compatibility_enabled = bool(cfg.get("boundary_compatibility", {}).get("enabled", False))

    if name == "baseline":
        if boundary_mix_enabled or component_enabled or compatibility_enabled:
            fail("baseline_boundary_modules_enabled")
        return

    if boundary_mix_enabled:
        fail(f"{name} legacy_boundary_mix_enabled")
    if name == "v2_component_weighting":
        if not component_enabled or compatibility_enabled:
            fail(f"{name} expected_v2_only")
        return
    if name == "v2_v3_best_template":
        if not component_enabled or not compatibility_enabled:
            fail(f"{name} expected_v2_plus_v3")
        if not bool(cfg["boundary_compatibility"].get("use_component_gate", False)):
            fail(f"{name} component_gate_disabled")
        return

    if not compatibility_enabled or component_enabled:
        fail(f"{name} expected_v3_only")
    if bool(cfg["boundary_compatibility"].get("use_component_gate", False)):
        fail(f"{name} standalone_v3_uses_component_gate")


def main() -> int:
    for name, slug in CONFIGS.items():
        cfg = load_config(name)
        require_common(name, cfg, slug)
        require_modules(name, cfg)
        print(f"VERIFY_OK {name}")
    print("VERIFY_VOC6_C321_GBS8_CONFIGS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
