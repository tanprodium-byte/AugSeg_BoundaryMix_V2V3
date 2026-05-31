import os
import sys

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.boundary_component import compute_component_weights


def _weighted_ce_denominator(weight, target, ignore_index=255, eps=1e-6):
    valid = target.ne(ignore_index)
    denom = (weight * valid.to(dtype=weight.dtype)).sum().clamp_min(eps)
    return denom


def test_no_affected_component_returns_base_weight():
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    target[:, 2:6, 2:6] = 1
    confidence = torch.full((1, 8, 8), 0.75)
    mix_mask = torch.zeros((1, 8, 8), dtype=torch.float32)

    weight, stats = compute_component_weights(
        target,
        confidence,
        mix_mask,
        ignore_index=255,
        connectivity=8,
        foreground_only=True,
        base_pixel_weight="one",
    )

    assert torch.allclose(weight, torch.ones_like(weight))
    assert stats["num_affected_components"] == 0
    assert stats["num_affected_pixels"] == 0


def test_affected_target_component_gets_soft_q_below_one():
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    target[:, 2:6, 2:6] = 1
    confidence = torch.full((1, 8, 8), 0.5)
    mix_mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mix_mask[:, 2:6, 2:5] = 1.0

    weight, stats = compute_component_weights(
        target,
        confidence,
        mix_mask,
        ignore_index=255,
        connectivity=8,
        foreground_only=True,
        tau_visible_low=0.2,
        tau_visible_high=0.6,
        area_min=1,
        area_max=16,
        use_mean_confidence_in_q=True,
        base_pixel_weight="one",
    )

    visible_fragment = weight[0, 2:6, 5:6]
    assert stats["num_affected_components"] == 1
    assert stats["num_affected_pixels"] == 4
    assert float(visible_fragment.max().item()) < 1.0
    assert float(visible_fragment.min().item()) >= 0.0


def test_force_q_one_is_neutral_and_denominator_nonzero():
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    target[:, 2:6, 2:6] = 1
    confidence = torch.full((1, 8, 8), 0.5)
    mix_mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mix_mask[:, 2:6, 2:5] = 1.0

    weight, stats = compute_component_weights(
        target,
        confidence,
        mix_mask,
        ignore_index=255,
        connectivity=8,
        foreground_only=True,
        area_min=1,
        area_max=16,
        base_pixel_weight="one",
        force_q_one=True,
    )

    denom = _weighted_ce_denominator(weight, target)
    pred = torch.randn((1, 2, 8, 8), dtype=torch.float32)
    loss = (F.cross_entropy(pred, target, ignore_index=255, reduction="none") * weight).sum() / denom

    assert stats["num_affected_components"] == 1
    assert torch.allclose(weight, torch.ones_like(weight))
    assert torch.isfinite(loss)
    assert float(denom.item()) > 0.0


def main():
    test_no_affected_component_returns_base_weight()
    test_affected_target_component_gets_soft_q_below_one()
    test_force_q_one_is_neutral_and_denominator_nonzero()
    print("BoundaryMix V2 smoke tests passed.")


if __name__ == "__main__":
    main()
