import os
import sys
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import util.boundary_mix as boundary_mix
from util.boundary_compatibility import compute_js_boundary_compatibility_loss


def _boxes(batch, row1, col1, row2, col2):
    return tuple(np.full((batch,), value, dtype=np.int64) for value in (row1, col1, row2, col2))


def _inputs(size=6):
    image = torch.stack((torch.full((3, size, size), 10.0), torch.full((3, size, size), 20.0)))
    target = torch.stack((torch.zeros((size, size), dtype=torch.long), torch.ones((size, size), dtype=torch.long)))
    confidence = torch.stack((torch.full((size, size), 0.6), torch.full((size, size), 0.8)))
    probs = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    labeled_image = torch.stack((torch.full((3, size, size), 100.0), torch.full((3, size, size), 200.0)))
    labeled_target = torch.stack(
        (torch.full((size, size), 2, dtype=torch.long), torch.zeros((size, size), dtype=torch.long))
    )
    return image, target, confidence, probs, labeled_image, labeled_target


def _run_mixer(labeled_box, final_box, sample_confidence, gate_draws):
    values = _inputs()
    image, target, confidence, probs, labeled_image, labeled_target = [value.clone() for value in values]
    with mock.patch.object(boundary_mix.torch, "randperm", return_value=torch.tensor([1, 0])), mock.patch.object(
        boundary_mix.np.random, "beta", side_effect=[0.8, 0.5]
    ), mock.patch.object(
        boundary_mix.np.random, "random", side_effect=list(gate_draws)
    ), mock.patch.object(
        boundary_mix, "_rand_bbox", side_effect=[labeled_box, final_box]
    ):
        result = boundary_mix.cut_mix_label_adaptive_with_mask(
            image,
            target,
            confidence,
            labeled_image,
            labeled_target,
            sample_confidence,
            unlabeled_probs=probs,
            return_labeled_origin_mask=True,
        )
    return values, result


def _expected_box_mask(row1, col1, row2, col2, size=6):
    result = torch.zeros((2, size, size), dtype=torch.float32)
    result[:, row1:row2, col1:col2] = 1.0
    return result


def test_a_u2u_provenance_without_labeled_enrichment():
    original, result = _run_mixer(
        _boxes(2, 0, 0, 2, 2), _boxes(2, 1, 1, 5, 5), [1.0, 1.0], [0.5, 0.5]
    )
    mixed_image, mixed_target, mixed_confidence, mix_source_mask, labeled_origin_mask, mixed_probs = result
    expected = _expected_box_mask(1, 1, 5, 5)
    assert torch.equal(mix_source_mask, expected)
    assert not labeled_origin_mask.bool().any()
    assert torch.equal(mixed_image[0, :, 1:5, 1:5], original[0][1, :, 1:5, 1:5])
    assert torch.equal(mixed_target[0, 1:5, 1:5], original[1][1, 1:5, 1:5])
    assert torch.equal(mixed_confidence[0, 1:5, 1:5], original[2][1, 1:5, 1:5])
    assert torch.equal(mixed_probs[0, :, 1:5, 1:5], original[3][1, :, 1:5, 1:5])

    features = torch.zeros((2, 2, 6, 6), dtype=torch.float32)
    features[:, 0] = 1.0
    features[:, 0][mix_source_mask.bool()] = 0.0
    features[:, 1][mix_source_mask.bool()] = 1.0
    teacher_probs = torch.full((2, 3, 6, 6), 1.0 / 3.0)
    loss_a, stats_a = compute_js_boundary_compatibility_loss(
        features,
        mixed_target,
        teacher_probs,
        mixed_confidence,
        mix_source_mask,
        labeled_origin_mask=labeled_origin_mask,
        num_classes=3,
        band_width=1,
        pair_radius=2,
    )
    changed_target = (mixed_target + 1) % 3
    loss_b, stats_b = compute_js_boundary_compatibility_loss(
        features,
        changed_target,
        teacher_probs,
        mixed_confidence,
        mix_source_mask,
        labeled_origin_mask=labeled_origin_mask,
        num_classes=3,
        band_width=1,
        pair_radius=2,
    )
    assert stats_a["bcr/num_pairs_candidate"] > 0
    assert stats_a["bcr/num_pairs_active"] > 0
    assert loss_a.item() > 0.0
    assert torch.allclose(loss_a, loss_b)
    assert stats_a["bcr/num_pairs_same"] == stats_b["bcr/num_pairs_same"]


def test_b_labeled_enrichment_survives_and_stays_separate():
    original, result = _run_mixer(
        _boxes(2, 2, 2, 4, 4), _boxes(2, 1, 1, 5, 5), [0.0, 1.0], [0.5, 0.5]
    )
    mixed_image, mixed_target, mixed_confidence, mix_source_mask, labeled_origin_mask, mixed_probs = result
    assert torch.equal(mix_source_mask, _expected_box_mask(1, 1, 5, 5))
    expected_origin = torch.zeros_like(labeled_origin_mask)
    expected_origin[1, 2:4, 2:4] = 1.0
    assert torch.equal(labeled_origin_mask, expected_origin)
    assert torch.equal(mixed_image[1, :, 2:4, 2:4], original[4][1, :, 2:4, 2:4])
    assert torch.equal(mixed_target[1, 2:4, 2:4], original[5][1, 2:4, 2:4])
    assert torch.equal(mixed_confidence[1, 2:4, 2:4], torch.ones((2, 2)))
    assert torch.equal(mixed_probs[1, :, 2:4, 2:4], torch.zeros((3, 2, 2)))
    ordinary = mix_source_mask.bool() & ~labeled_origin_mask.bool()
    assert ordinary.any()
    assert torch.all(mixed_confidence[ordinary] < 1.0)
    assert torch.allclose(mixed_probs.sum(dim=1)[ordinary], torch.ones_like(mixed_confidence[ordinary]))


def test_c_nonoverlapping_enrichment_does_not_remove_u2u_boundary():
    _, result = _run_mixer(
        _boxes(2, 0, 0, 1, 1), _boxes(2, 2, 2, 5, 5), [0.0, 0.0], [0.5, 0.5]
    )
    _, mixed_target, mixed_confidence, mix_source_mask, labeled_origin_mask, mixed_probs = result
    assert torch.equal(mix_source_mask, _expected_box_mask(2, 2, 5, 5))
    assert not labeled_origin_mask.bool().any()
    features = torch.randn((2, 3, 6, 6), generator=torch.Generator().manual_seed(9))
    _, stats = compute_js_boundary_compatibility_loss(
        features,
        mixed_target,
        mixed_probs,
        mixed_confidence,
        mix_source_mask,
        labeled_origin_mask=labeled_origin_mask,
        num_classes=3,
        band_width=1,
        pair_radius=2,
    )
    assert stats["bcr/num_pairs_candidate"] > 0


def test_d_zero_area_final_paste_is_a_noop():
    original, result = _run_mixer(
        _boxes(2, 0, 0, 1, 1), _boxes(2, 3, 3, 3, 3), [1.0, 1.0], [0.5, 0.5]
    )
    mixed_image, mixed_target, mixed_confidence, mix_source_mask, labeled_origin_mask, mixed_probs = result
    assert torch.equal(mixed_image, original[0])
    assert torch.equal(mixed_target, original[1])
    assert torch.equal(mixed_confidence, original[2])
    assert torch.equal(mixed_probs, original[3])
    assert not mix_source_mask.bool().any()
    assert not labeled_origin_mask.bool().any()


def test_e_final_tensor_alignment():
    original, result = _run_mixer(
        _boxes(2, 0, 0, 1, 1), _boxes(2, 1, 2, 5, 5), [1.0, 1.0], [0.5, 0.5]
    )
    mixed_image, mixed_target, mixed_confidence, mix_source_mask, labeled_origin_mask, mixed_probs = result
    for receiver, donor in ((0, 1), (1, 0)):
        region = mix_source_mask[receiver].bool()
        assert torch.equal(mixed_image[receiver, :, region], original[0][donor, :, region])
        assert torch.equal(mixed_target[receiver, region], original[1][donor, region])
        assert torch.equal(mixed_confidence[receiver, region], original[2][donor, region])
        assert torch.equal(mixed_probs[receiver, :, region], original[3][donor, :, region])
    assert not labeled_origin_mask.bool().any()


def test_f_bcr_formulas_and_origin_confidence_semantics():
    mix_source_mask = torch.zeros((1, 4, 4), dtype=torch.float32)
    mix_source_mask[:, :, :2] = 1.0
    labeled_origin_mask = torch.zeros_like(mix_source_mask)
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    confidence = torch.full((1, 4, 4), 0.8)
    confidence[:, :, :2] = 0.5
    teacher_probs = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
    features = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    features[:, 0, :, :2] = 1.0
    features[:, 1, :, 2:] = 1.0

    same_loss, same_stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_source_mask,
        labeled_origin_mask=labeled_origin_mask,
        num_classes=2,
        band_width=1,
        pair_radius=1,
        tau_same=0.8,
        tau_diff=0.3,
        margin=0.4,
    )
    assert torch.allclose(same_loss, torch.tensor(0.25), atol=1e-6)
    assert abs(same_stats["bcr/mean_r_ab"] - 0.4) < 1e-6

    teacher_probs[:, :, :, 2:] = F.one_hot(
        torch.ones((1, 4, 2), dtype=torch.long), num_classes=2
    ).permute(0, 3, 1, 2).float()
    features[:, :, :, 2:] = features[:, :, :, :2]
    diff_loss, diff_stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_source_mask,
        labeled_origin_mask=labeled_origin_mask,
        num_classes=2,
        band_width=1,
        pair_radius=1,
        tau_same=0.8,
        tau_diff=0.3,
        margin=0.4,
    )
    assert torch.allclose(diff_loss, torch.tensor(0.36), atol=1e-6)
    assert diff_stats["bcr/num_pairs_diff"] > 0

    labeled_origin_mask[:, :, :2] = 1.0
    teacher_probs = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
    teacher_probs[:, :, :, :2] = F.one_hot(
        torch.ones((1, 4, 2), dtype=torch.long), num_classes=2
    ).permute(0, 3, 1, 2).float()
    _, origin_stats = compute_js_boundary_compatibility_loss(
        features,
        target,
        teacher_probs,
        confidence,
        mix_source_mask,
        labeled_origin_mask=labeled_origin_mask,
        num_classes=2,
        band_width=1,
        pair_radius=1,
    )
    assert origin_stats["bcr/num_pairs_same"] > 0
    assert abs(origin_stats["bcr/mean_r_ab"] - 0.8) < 1e-6


def test_g_u_method_dispatch_source_is_unchanged_by_provenance_api():
    with open(os.path.join(ROOT, "train_semi.py"), encoding="utf-8") as handle:
        source = handle.read()
    u_call = source[source.index("image_u_aug, label_u_aug, logits_u_aug, u1_step_diagnostics") :]
    u_call = u_call[: u_call.index("u1_diagnostics_accumulator.add_")]
    assert "mix_source_mask" not in u_call
    assert "labeled_origin_mask" not in u_call


def test_h_opt_in_return_is_rng_neutral():
    values = _inputs()

    def run(return_origin):
        np.random.seed(731)
        torch.manual_seed(731)
        inputs = [value.clone() for value in values]
        output = boundary_mix.cut_mix_label_adaptive_with_mask(
            inputs[0], inputs[1], inputs[2], inputs[4], inputs[5], [0.4, 0.7],
            unlabeled_probs=inputs[3], return_labeled_origin_mask=return_origin,
        )
        return output, np.random.get_state(), torch.random.get_rng_state()

    without_origin, numpy_without, torch_without = run(False)
    with_origin, numpy_with, torch_with = run(True)
    assert all(torch.equal(a, b) for a, b in zip(without_origin[:3], with_origin[:3]))
    assert torch.equal(without_origin[3], with_origin[4])
    assert torch.equal(without_origin[4], with_origin[5])
    assert numpy_without[0] == numpy_with[0]
    assert np.array_equal(numpy_without[1], numpy_with[1])
    assert numpy_without[2:] == numpy_with[2:]
    assert torch.equal(torch_without, torch_with)


def main():
    test_a_u2u_provenance_without_labeled_enrichment()
    test_b_labeled_enrichment_survives_and_stays_separate()
    test_c_nonoverlapping_enrichment_does_not_remove_u2u_boundary()
    test_d_zero_area_final_paste_is_a_noop()
    test_e_final_tensor_alignment()
    test_f_bcr_formulas_and_origin_confidence_semantics()
    test_g_u_method_dispatch_source_is_unchanged_by_provenance_api()
    test_h_opt_in_return_is_rng_neutral()
    print("V3_PROVENANCE_SPLIT_SMOKE: PASS")


if __name__ == "__main__":
    main()
