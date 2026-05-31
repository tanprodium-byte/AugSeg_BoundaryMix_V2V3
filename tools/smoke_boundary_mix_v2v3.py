import os
import sys

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.boundary_component import compute_component_weights
from util.boundary_compatibility import compute_js_boundary_compatibility_loss


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


def _base_v3_inputs(num_classes=3):
    features = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    features[:, 0, :, :4] = 1.0
    features[:, 0, :, 4:] = 1.0
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    target[:, :, :4] = 1
    target[:, :, 4:] = 1
    teacher_probs = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    confidence = torch.ones((1, 8, 8), dtype=torch.float32)
    mix_mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mix_mask[:, :, :4] = 1.0
    return features, target, teacher_probs, confidence, mix_mask


def test_v3_js_finite_with_zero_probabilities():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()
    teacher_probs.zero_()
    teacher_probs[:, 1, :, 4:] = 1.0e-12

    loss, stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=32,
    )

    assert torch.isfinite(loss)
    assert stats["num_pairs_per_image"] > 0
    assert stats["mean_JS"] >= 0.0


def test_v3_same_semantic_pairs_produce_valid_same_loss():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()
    features[:, 1, :, 4:] = 1.0

    loss, stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
    )

    assert torch.isfinite(loss)
    assert stats["same_pairs_ratio"] > 0.0
    assert stats["diff_pairs_ratio"] == 0.0
    assert loss.item() >= 0.0


def test_v3_different_semantic_pairs_produce_valid_diff_loss():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()
    target[:, :, 4:] = 2
    teacher_probs = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()

    loss, stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
        margin=0.4,
    )

    assert torch.isfinite(loss)
    assert stats["diff_pairs_ratio"] > 0.0
    assert loss.item() > 0.0


def test_v3_uncertain_pairs_are_ignored():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs(num_classes=2)
    target[:, :, :4] = 0
    teacher_probs = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    teacher_probs[:, 0, :, 4:] = 0.2
    teacher_probs[:, 1, :, 4:] = 0.8

    loss, stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        num_classes=2,
        pair_radius=1,
        max_pairs_per_image=64,
        tau_same=0.8,
        tau_diff=0.3,
    )

    assert torch.isfinite(loss)
    assert stats["uncertain_pairs_ratio"] > 0.0
    assert loss.item() == 0.0


def test_v3_pair_count_is_capped():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()

    loss, stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        num_classes=3,
        pair_radius=3,
        max_pairs_per_image=5,
    )

    assert torch.isfinite(loss)
    assert stats["num_pairs_per_image"] <= 5.0


def test_v3_standalone_does_not_use_component_q_c():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()
    component_weight = torch.zeros_like(confidence)

    loss_without_q, _ = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        component_weight_map_or_none=None,
        use_component_gate=False,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
    )
    loss_with_ignored_q, _ = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        component_weight_map_or_none=component_weight,
        use_component_gate=False,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
    )

    assert torch.allclose(loss_without_q, loss_with_ignored_q)


def test_combined_v2_v3_passes_component_weight_map_into_bcr():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()
    confidence = torch.full_like(confidence, 0.8)
    component_weight, component_stats = compute_component_weights(
        target,
        confidence,
        mix_mask,
        ignore_index=255,
        connectivity=8,
        foreground_only=True,
        tau_visible_low=0.2,
        tau_visible_high=0.6,
        area_min=1,
        area_max=32,
        use_mean_confidence_in_q=True,
        base_pixel_weight="one",
    )

    loss, stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        component_weight_map_or_none=component_weight,
        use_component_gate=True,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
    )

    assert torch.isfinite(loss)
    assert component_stats["num_affected_components"] == 1
    assert stats["same_pairs_ratio"] > 0.0
    assert 0.0 < stats["mean_r_ab"] < 1.0


def test_combined_v2_v3_component_gate_finite_loss():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()
    target[:, :, 4:] = 2
    teacher_probs = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    component_weight = torch.ones_like(confidence)
    component_weight[:, :, 4:] = 0.25

    loss, stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        component_weight_map_or_none=component_weight,
        use_component_gate=True,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
    )

    assert torch.isfinite(loss)
    assert stats["diff_pairs_ratio"] > 0.0
    assert loss.item() >= 0.0


def test_v3_standalone_accepts_none_component_weight():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()

    loss, stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        component_weight_map_or_none=None,
        use_component_gate=False,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
    )

    assert torch.isfinite(loss)
    assert stats["num_pairs_per_image"] > 0.0


def test_combined_without_component_gate_is_valid():
    features, target, teacher_probs, confidence, mix_mask = _base_v3_inputs()
    component_weight = torch.zeros_like(confidence)

    loss_without_gate, _ = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        component_weight_map_or_none=component_weight,
        use_component_gate=False,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
    )
    loss_standalone, _ = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_mask,
        component_weight_map_or_none=None,
        use_component_gate=False,
        num_classes=3,
        pair_radius=1,
        max_pairs_per_image=64,
    )

    assert torch.isfinite(loss_without_gate)
    assert torch.allclose(loss_without_gate, loss_standalone)


def main():
    test_no_affected_component_returns_base_weight()
    test_affected_target_component_gets_soft_q_below_one()
    test_force_q_one_is_neutral_and_denominator_nonzero()
    test_v3_js_finite_with_zero_probabilities()
    test_v3_same_semantic_pairs_produce_valid_same_loss()
    test_v3_different_semantic_pairs_produce_valid_diff_loss()
    test_v3_uncertain_pairs_are_ignored()
    test_v3_pair_count_is_capped()
    test_v3_standalone_does_not_use_component_q_c()
    test_combined_v2_v3_passes_component_weight_map_into_bcr()
    test_combined_v2_v3_component_gate_finite_loss()
    test_v3_standalone_accepts_none_component_weight()
    test_combined_without_component_gate_is_valid()
    print("BoundaryMix V2/V3 smoke tests passed.")


if __name__ == "__main__":
    main()
