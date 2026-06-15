import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.csl_cutmix import get_csl_guided_boxes
from util.csl_reliability import apply_csl_random_reliable_mask, compute_csl_reliability
from util.boundary_mix import cut_mix_label_adaptive_with_mask


def _fixed_sampler(size, lam=None):
    del lam
    batch = size[0]
    return (
        np.zeros(batch, dtype=np.int64),
        np.zeros(batch, dtype=np.int64),
        np.full(batch, size[2] // 2, dtype=np.int64),
        np.full(batch, size[3] // 2, dtype=np.int64),
    )


class AlternatingSampler:
    def __init__(self):
        self.count = 0

    def __call__(self, size, lam=None):
        del lam
        batch, height, width = size[0], size[2], size[3]
        if self.count % 2 == 0:
            box = (0, 0, height, width // 2)
        else:
            box = (0, width // 2, height, width)
        self.count += 1
        return tuple(np.full(batch, value, dtype=np.int64) for value in box)


def test_compute_csl_reliability():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(2, 4, 6, 7, requires_grad=True), dim=1)
    reliability, stats = compute_csl_reliability(probs, cfg={"reliability_mode": "entropy_margin"})
    assert reliability.shape == (2, 6, 7)
    assert reliability.requires_grad is False
    assert torch.isfinite(reliability).all()
    assert float(reliability.min()) >= 0.0
    assert float(reliability.max()) <= 1.0
    for key in (
        "csl/mean_reliability",
        "csl/reliability_std",
        "csl/reliability_min",
        "csl/reliability_max",
        "csl/confidence_mean",
        "csl/entropy_mean",
        "csl/margin_mean",
    ):
        assert key in stats
        assert np.isfinite(stats[key])


def test_apply_csl_random_reliable_mask():
    torch.manual_seed(1)
    reliability = torch.full((4, 12, 12), 0.8)
    effective, stats = apply_csl_random_reliable_mask(reliability, mask_prob=0.5, training=True)
    assert effective.shape == reliability.shape
    assert effective.requires_grad is False
    assert torch.isfinite(effective).all()
    assert torch.all(effective <= reliability)
    assert float(effective.mean()) < float(reliability.mean())
    for key in (
        "csl/mask_prob",
        "csl/masked_ratio",
        "csl/raw_reliability_mean",
        "csl/effective_weight_mean",
    ):
        assert key in stats
        assert np.isfinite(stats[key])


def test_get_csl_guided_boxes_low_reliability():
    torch.manual_seed(2)
    np.random.seed(2)
    reliability = torch.ones(2, 8, 8)
    reliability[:, :, :4] = 0.0
    boxes, stats = get_csl_guided_boxes(
        reliability,
        AlternatingSampler(),
        num_candidates=2,
        temperature=0.01,
        policy="low_reliability",
    )
    assert boxes.shape == (2, 4)
    assert torch.equal(boxes[:, 1], torch.zeros(2, dtype=torch.long))
    assert torch.equal(boxes[:, 3], torch.full((2,), 4, dtype=torch.long))
    assert stats["csl_cutmix/score_candidate_max"] > stats["csl_cutmix/score_candidate_min"]
    for key in (
        "csl_cutmix/score_selected",
        "csl_cutmix/score_candidate_mean",
        "csl_cutmix/score_candidate_max",
        "csl_cutmix/score_candidate_min",
        "csl_cutmix/score_candidate_std",
        "csl_cutmix/prob_selected",
        "csl_cutmix/prob_max",
        "csl_cutmix/selection_entropy",
        "csl_cutmix/target_reliability_selected",
        "csl_cutmix/fallback_ratio",
    ):
        assert key in stats
        assert np.isfinite(stats[key])


def test_get_csl_guided_boxes_fallback():
    reliability = torch.zeros(2, 8, 8)
    boxes, stats = get_csl_guided_boxes(
        reliability,
        _fixed_sampler,
        num_candidates=0,
        temperature=0.2,
        policy="low_reliability",
    )
    assert boxes.shape == (2, 4)
    assert stats["csl_cutmix/fallback_ratio"] == 1.0


def test_cutmix_preserves_labeled_weight_one():
    torch.manual_seed(3)
    np.random.seed(3)
    batch, height, width = 2, 8, 8
    unlabeled_image = torch.zeros(batch, 3, height, width)
    unlabeled_mask = torch.zeros(batch, height, width, dtype=torch.long)
    unlabeled_logits = torch.full((batch, height, width), 0.2)
    unlabeled_weight = torch.full((batch, height, width), 0.4)
    labeled_image = torch.ones(batch, 3, height, width)
    labeled_mask = torch.ones(batch, height, width, dtype=torch.long)
    confidence = [0.0, 0.0]

    _, _, _, source_mask, mixed_weight = cut_mix_label_adaptive_with_mask(
        unlabeled_image,
        unlabeled_mask,
        unlabeled_logits,
        labeled_image,
        labeled_mask,
        confidence,
        unlabeled_weight=unlabeled_weight,
        labeled_boxes=torch.tensor([[0, 0, 4, 4], [0, 0, 4, 4]]),
        target_boxes=torch.tensor([[0, 0, 4, 4], [0, 0, 4, 4]]),
    )
    assert source_mask.sum() > 0
    assert torch.allclose(mixed_weight[source_mask.bool()], torch.ones_like(mixed_weight[source_mask.bool()]))


def _load_config(rel_path):
    with open(os.path.join(ROOT, rel_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_configs():
    c1 = _load_config("exps/boundary_mix_v2_v3/voc_semi662/c1_csl_pseudo_selection/config.yaml")
    assert c1["csl"]["enabled"] is True
    assert c1["csl"]["use_csl_for_ce_weight"] is True
    assert c1["csl"]["use_csl_for_cutmix"] is False
    assert c1["saliency_cutmix"]["enabled"] is False
    assert c1["boundary_component"]["enabled"] is False
    assert c1["boundary_compatibility"]["enabled"] is False

    c2 = _load_config("exps/boundary_mix_v2_v3/voc_semi662/c2_csl_random_reliable_masking/config.yaml")
    assert c2["csl"]["enabled"] is True
    assert c2["csl"]["use_csl_for_ce_weight"] is True
    assert c2["csl"]["random_mask_reliable"] is True
    assert float(c2["csl"]["mask_prob"]) == 0.3
    assert c2["csl"]["mask_labeled_pixels"] is False
    assert c2["saliency_cutmix"]["enabled"] is False
    assert c2["boundary_component"]["enabled"] is False
    assert c2["boundary_compatibility"]["enabled"] is False

    c3 = _load_config("exps/boundary_mix_v2_v3/voc_semi662/c3_csl_guided_cutmix_plus_v3_d2/config.yaml")
    assert c3["csl_cutmix"]["enabled"] is True
    assert c3["csl"]["use_csl_for_cutmix"] is True
    assert c3["csl"]["use_csl_for_ce_weight"] is False
    assert c3["saliency_cutmix"]["enabled"] is False
    assert c3["boundary_component"]["enabled"] is False
    assert c3["boundary_compatibility"]["enabled"] is True
    assert c3["boundary_compatibility"]["pair_radius"] == 2
    assert float(c3["boundary_compatibility"]["lambda_bcr"]) == 0.01
    assert c3["boundary_compatibility"]["use_component_gate"] is False


def main():
    test_compute_csl_reliability()
    test_apply_csl_random_reliable_mask()
    test_get_csl_guided_boxes_low_reliability()
    test_get_csl_guided_boxes_fallback()
    test_cutmix_preserves_labeled_weight_one()
    test_configs()
    print("CSL C1/C2/C3 smoke tests passed")


if __name__ == "__main__":
    main()
