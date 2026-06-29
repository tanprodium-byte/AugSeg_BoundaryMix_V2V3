#!/usr/bin/env python
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.csl_official import (  # noqa: E402
    apply_csl_reliable_mask_perturbation,
    compute_csl_official_selection,
)


def _load_config(rel_path):
    with open(os.path.join(ROOT, rel_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_compute_csl_official_selection():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(2, 4, 8, 9), dim=1)
    ignore_mask = torch.zeros(2, 8, 9, dtype=torch.long)
    ignore_mask[:, :2, :3] = 255
    out = compute_csl_official_selection(probs, ignore_mask=ignore_mask, alpha=8.0)

    assert out["pseudo_label"].shape == (2, 8, 9)
    assert out["pseudo_label"].dtype == torch.long
    assert out["raw_confidence"].shape == (2, 8, 9)
    assert out["residual_variance"].shape == (2, 8, 9)
    assert out["weight"].shape == (2, 8, 9)
    assert out["reliable_mask"].shape == (2, 8, 9)
    assert out["reliable_mask"].dtype == torch.bool
    assert out["sample_reliability"].shape == (2,)
    assert torch.isfinite(out["weight"]).all()
    assert torch.isfinite(out["sample_reliability"]).all()
    assert float(out["weight"].min()) >= 0.0
    assert float(out["weight"].max()) <= 1.0
    assert torch.equal(out["weight"][ignore_mask == 255], torch.zeros_like(out["weight"][ignore_mask == 255]))


def test_apply_csl_reliable_mask_perturbation():
    torch.manual_seed(1)
    image = torch.ones(2, 3, 8, 8)
    reliable = torch.zeros(2, 8, 8, dtype=torch.bool)
    reliable[:, :4, :4] = True
    perturbed, perturb_mask, stats = apply_csl_reliable_mask_perturbation(
        image,
        reliable,
        mask_prob=1.0,
        block_size=1,
        cover_ratio=1.0,
        mode="zero",
    )
    assert perturbed.shape == image.shape
    assert perturb_mask.shape == reliable.shape
    assert torch.equal(perturb_mask, reliable)
    assert torch.equal(perturbed[:, :, :4, :4], torch.zeros_like(perturbed[:, :, :4, :4]))
    assert torch.equal(perturbed[:, :, 4:, 4:], torch.ones_like(perturbed[:, :, 4:, 4:]))
    assert stats["csl/perturb_reliable_ratio"] == 1.0


def test_configs():
    c1 = _load_config(
        "exps/boundary_mix_v2_v3/voc_semi662/c1_csl_official_reliability_replace_confidence/config.yaml"
    )
    assert c1["csl"]["enabled"] is True
    assert c1["csl"]["mode"] == "official_reliability_replace_confidence"
    assert c1["csl"]["reliability_mode"] == "official_pcos"
    assert c1["csl"]["use_csl_for_ce_weight"] is True
    assert c1["csl"]["use_csl_for_mix_confidence"] is True
    assert c1["csl"]["random_mask_reliable"] is False
    assert c1["csl"]["use_csl_for_cutmix"] is False

    c2 = _load_config(
        "exps/boundary_mix_v2_v3/voc_semi662/c2_csl_official_reliable_mask_perturbation/config.yaml"
    )
    assert c2["csl"]["enabled"] is True
    assert c2["csl"]["mode"] == "official_reliable_mask_perturbation"
    assert c2["csl"]["reliability_mode"] == "official_pcos"
    assert c2["csl"]["use_csl_for_ce_weight"] is True
    assert c2["csl"]["use_csl_for_mix_confidence"] is True
    assert c2["csl"]["random_mask_reliable"] is True
    assert c2["csl"]["perturb_input"] is True
    assert float(c2["csl"]["mask_prob"]) == 0.3
    assert c2["csl"]["mask_mode"] == "zero"
    assert c2["csl"]["use_csl_for_cutmix"] is False


def main():
    test_compute_csl_official_selection()
    test_apply_csl_reliable_mask_perturbation()
    test_configs()
    print("Official CSL smoke tests passed")


if __name__ == "__main__":
    main()
