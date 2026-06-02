#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import copy
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "exps/boundary_mix_v2_v3/voc_semi662"
OUT_ROOT = ROOT / ".codex_smoke/real_configs/voc5_single_gpu_gbs8"

VERSIONS = [
    "v2_component_weighting",
    "v2_v3_best_template",
    "v3_js_bcr_d1",
    "v3_js_bcr_d2",
    "v3_js_bcr_d3",
]


def main() -> int:
    written: list[Path] = []
    for version in VERSIONS:
        src = BASE_DIR / version / "config.yaml"
        if not src.is_file():
            raise FileNotFoundError(src)

        with src.open("r") as f:
            cfg = yaml.safe_load(f)

        cfg = copy.deepcopy(cfg)
        cfg["dataset"]["train"]["batch_size"] = 8
        cfg["dataset"]["val"]["batch_size"] = 1
        cfg.setdefault("hf", {})["enabled"] = False
        cfg["hf"]["auto_download"] = False
        cfg["hf"]["auto_upload"] = False
        cfg.setdefault("wandb", {})["enable"] = False
        cfg.setdefault("saver", {})["snapshot_dir"] = f"./exp_smoke/real_single_gpu_gbs8/{version}"
        cfg["saver"]["snapshot_dir"] = f"./exp_smoke/real_single_gpu_gbs8/{version}"

        out = OUT_ROOT / version / "config.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written.append(out)

    print("Created single-GPU global batch 8 configs. Not launched:")
    for path in written:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
