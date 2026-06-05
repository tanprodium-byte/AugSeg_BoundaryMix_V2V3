#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Any
import sys
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "exps/boundary_mix_v2_v3/voc_semi662"
REAL_DIR = ROOT / ".codex_smoke/real_configs/voc5_single_gpu_gbs8"

VERSIONS = [
    "v2_component_weighting",
    "v2_v3_best_template",
    "v3_js_bcr_d1",
    "v3_js_bcr_d2",
    "v3_js_bcr_d3",
]

ALLOWED_EXACT = {
    ("dataset", "train", "batch_size"),
    ("dataset", "val", "batch_size"),
    ("saver", "snapshot_dir"),
    ("saver", "auto_resume"),
    ("checkpoint", "auto_resume"),
}
ALLOWED_PREFIXES = {
    ("hf",),
    ("wandb",),
}

PROTECTED_EXACT = {
    ("dataset", "train", "strong_aug"),
    ("dataset", "train", "crop"),
    ("dataset", "train", "resize_base_size"),
    ("dataset", "train", "cutmix"),
    ("dataset", "train", "mix"),
    ("trainer", "optimizer"),
    ("trainer", "lr_scheduler"),
    ("trainer", "epochs"),
    ("trainer", "unsupervised"),
    ("trainer", "threshold"),
    ("boundary_component",),
    ("boundary_compatibility",),
    ("model",),
    ("net",),
}


def load_yaml(path: Path) -> Any:
    with path.open("r") as f:
        return yaml.safe_load(f)


def is_prefix(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(path) >= len(prefix) and path[: len(prefix)] == prefix


def is_allowed(path: tuple[str, ...]) -> bool:
    if path in ALLOWED_EXACT:
        return True
    return any(is_prefix(path, prefix) for prefix in ALLOWED_PREFIXES)


def is_protected(path: tuple[str, ...]) -> bool:
    return any(is_prefix(path, prefix) for prefix in PROTECTED_EXACT)


def scalar_repr(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 180 else text[:177] + "..."


def diff(a: Any, b: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any, Any]]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[tuple[tuple[str, ...], Any, Any]] = []
        for key in sorted(set(a) | set(b)):
            out.extend(diff(a.get(key, "<MISSING>"), b.get(key, "<MISSING>"), path + (str(key),)))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if a == b:
            return []
        return [(path, a, b)]
    if a != b:
        return [(path, a, b)]
    return []


def main() -> int:
    failed = False
    for version in VERSIONS:
        base_path = BASE_DIR / version / "config.yaml"
        real_path = REAL_DIR / version / "config.yaml"
        base_cfg = load_yaml(base_path)
        real_cfg = load_yaml(real_path)
        diffs = diff(base_cfg, real_cfg)

        print(f"== {version}")
        if not diffs:
            print("NO_DIFF")
            continue

        for path, old, new in diffs:
            dotted = ".".join(path)
            status = "ALLOWED" if is_allowed(path) else "DISALLOWED"
            if is_protected(path):
                status = "PROTECTED_FAIL"
            print(f"{status} {dotted}: {scalar_repr(old)} -> {scalar_repr(new)}")

            if not is_allowed(path) or is_protected(path):
                failed = True

    if failed:
        print("VERIFY_REAL_CONFIGS: FAIL")
        return 1

    print("VERIFY_REAL_CONFIGS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
