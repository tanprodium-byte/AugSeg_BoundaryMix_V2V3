#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.csl_cutmix import get_csl_guided_boxes  # noqa: E402
from util.csl_official import compute_csl_official_selection  # noqa: E402
from util.boundary_mix import _rand_bbox  # noqa: E402


CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/c3_csl_official_guided_cutmix_plus_v3_d2/config.yaml"
TRAIN = ROOT / "train_semi.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_config() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    require(cfg["run"]["name"] == "c3_csl_official_guided_cutmix_plus_v3_d2", "run name mismatch")
    require(cfg["wandb"]["name"] == "c3_csl_official_guided_cutmix_plus_v3_d2", "wandb name mismatch")
    require("c3_csl_official_guided_cutmix_plus_v3_d2" in cfg["saver"]["snapshot_dir"], "snapshot mismatch")
    require(cfg["csl"]["enabled"] is True, "CSL must be enabled")
    require(cfg["csl"]["mode"] == "official_guided_cutmix", "C3 official mode mismatch")
    require(cfg["csl"]["reliability_mode"] == "official_pcos", "C3 must use official_pcos")
    require(cfg["csl"].get("use_csl_for_cutmix") is True, "C3 must use CSL for CutMix")
    require(cfg["csl"].get("use_csl_for_ce_weight") is False, "C3 must not weight CE")
    require(cfg["csl"].get("use_csl_for_mix_confidence") is False, "C3 must not replace mix confidence")
    require(cfg["csl_cutmix"]["enabled"] is True, "CSL CutMix must be enabled")
    require(cfg["csl_cutmix"]["target_policy"] == "low_reliability", "target policy mismatch")
    require(cfg["saliency_cutmix"]["enabled"] is False, "C3 official must not enable saliency CutMix")
    bcr = cfg["boundary_compatibility"]
    require(bcr["enabled"] is True, "BCR must be enabled")
    require(int(bcr["pair_radius"]) == 2, "BCR must be d2")
    require(float(bcr["lambda_bcr"]) == 0.01, "BCR lambda mismatch")
    require(bcr["use_component_gate"] is False, "C3 official must not use component gate")
    require("entropy_margin" not in CONFIG.read_text(), "C3 official config must not contain entropy_margin")


def test_train_path_markers() -> None:
    source = TRAIN.read_text()
    require("official_guided_cutmix" in source, "train_semi missing official guided mode marker")
    require("csl_official_guided_cutmix_enabled" in source, "train_semi missing official guided guard")
    require("compute_csl_official_selection" in source, "train_semi missing official CSL selection")
    require('csl_selection["weight"]' in source, "train_semi must expose official weight map")
    require("get_csl_guided_boxes" in source, "train_semi missing CSL CutMix box selection")
    require("target_boxes=csl_target_boxes" in source, "train_semi must pass target boxes")


def test_tensor_path() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 4, 16, 16)
    probs = torch.softmax(logits, dim=1)
    ignore_mask = torch.full((2, 16, 16), 255, dtype=torch.long)
    ignore_mask[:, 2:14, 3:15] = 0
    selection = compute_csl_official_selection(probs, ignore_mask=ignore_mask, alpha=8.0)
    reliability = selection["weight"]
    require(reliability.shape == (2, 16, 16), "official reliability shape mismatch")
    require(reliability.dtype.is_floating_point, "official reliability must be floating point")
    require(float(reliability.min()) >= 0.0 and float(reliability.max()) <= 1.0, "reliability range mismatch")
    require(torch.count_nonzero(reliability[:, :2, :]) == 0, "ignore pixels should have zero reliability")

    boxes, stats = get_csl_guided_boxes(
        reliability,
        _rand_bbox,
        num_candidates=4,
        temperature=0.2,
        policy="low_reliability",
    )
    require(tuple(boxes.shape) == (2, 4), f"target box shape mismatch: {tuple(boxes.shape)}")
    require(boxes.dtype == torch.long, "target boxes must be long")
    require("csl_cutmix/target_reliability_selected" in stats, "missing csl_cutmix stats")
    require(stats["csl_cutmix/fallback_ratio"] in (0.0, 1.0), "unexpected fallback stat")


def main() -> int:
    test_config()
    test_train_path_markers()
    test_tensor_path()
    print("C3 official guided CutMix smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
