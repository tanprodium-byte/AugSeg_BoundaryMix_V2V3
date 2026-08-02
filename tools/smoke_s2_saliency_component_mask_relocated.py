#!/usr/bin/env python3
"""Synthetic smoke coverage for S2 exact-mask relocation."""

import os
import sys
import ast
from pathlib import Path

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import util.boundary_mix as boundary_mix  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Patch:
    def __init__(self, obj, name, value):
        self.obj = obj
        self.name = name
        self.value = value

    def __enter__(self):
        self.old = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self

    def __exit__(self, exc_type, exc, tb):
        setattr(self.obj, self.name, self.old)


def legacy_inputs():
    batch, height, width = 2, 6, 7
    unlabeled_image = torch.arange(batch * 3 * height * width, dtype=torch.float32).view(batch, 3, height, width) * -1
    unlabeled_target = torch.arange(batch * height * width, dtype=torch.long).view(batch, height, width)
    unlabeled_confidence = torch.full((batch, height, width), 0.25, dtype=torch.float16)
    labeled_image = torch.arange(batch * 3 * height * width, dtype=torch.float32).view(batch, 3, height, width) + 1000
    labeled_target = torch.arange(batch * height * width, dtype=torch.long).view(batch, height, width) + 100
    masks = torch.zeros(batch, height, width, dtype=torch.bool)
    masks[0, 1:4, 2] = True
    masks[0, 3, 2:5] = True
    masks[1, 0:2, 0:3] = True
    probs = torch.ones(batch, 4, height, width)
    weight = torch.full((batch, height, width), 0.5)
    return unlabeled_image, unlabeled_target, unlabeled_confidence, labeled_image, labeled_target, masks, probs, weight


def call_legacy(*, probs=False, weight=False, metadata=False, policy="same_coordinate"):
    values = legacy_inputs()
    u_image, u_target, u_conf, l_image, l_target, masks, prob_map, weight_map = values
    calls = {"randperm": 0, "randint": 0}

    def fixed_randperm(n, *args, **kwargs):
        calls["randperm"] += 1
        return torch.tensor([1, 0], dtype=torch.long)

    def forbidden_randint(*args, **kwargs):
        calls["randint"] += 1
        raise AssertionError("legacy direct mask path consumed destination RNG")

    def forbidden_bbox(*args, **kwargs):
        raise AssertionError("legacy direct mask path reached _rand_bbox")

    with Patch(torch, "randperm", fixed_randperm), Patch(torch, "randint", forbidden_randint), Patch(boundary_mix, "_rand_bbox", forbidden_bbox):
        result = boundary_mix.cut_mix_label_adaptive_with_mask(
            u_image,
            u_target,
            u_conf,
            l_image,
            l_target,
            [1.0, 1.0],
            return_target_metadata=metadata,
            unlabeled_probs=prob_map if probs else None,
            unlabeled_weight=weight_map if weight else None,
            labeled_masks=masks,
            direct_labeled_mix=True,
            direct_paste_policy=policy,
            direct_confidence_gate=True,
        )
    return result, values, calls


def test_legacy_same_coordinate_snapshot():
    result, values, calls = call_legacy(probs=True, weight=True)
    image, target, confidence, provenance, probs, weight = result
    u_image, u_target, u_conf, l_image, l_target, masks, prob_input, weight_input = values
    donor = torch.tensor([1, 0])
    expected_image = u_image.clone()
    expected_target = u_target.clone()
    expected_conf = u_conf.clone()
    expected_source = torch.zeros_like(u_target, dtype=torch.float32)
    expected_probs = prob_input.clone()
    expected_weight = weight_input.clone()
    for i, src in enumerate(donor.tolist()):
        mask = masks[src]
        expected_image[i, :, mask] = l_image[src, :, mask]
        expected_target[i, mask] = l_target[src, mask]
        expected_conf[i, mask] = 1
        expected_source[i, mask] = 1
        expected_probs[i, :, mask] = 0
        expected_weight[i, mask] = 1
    require(torch.equal(image, expected_image), "legacy RGB snapshot")
    require(torch.equal(target, expected_target), "legacy target snapshot")
    require(torch.equal(confidence, expected_conf), "legacy confidence snapshot")
    require(torch.equal(provenance, expected_source), "legacy provenance snapshot")
    require(torch.equal(probs, expected_probs), "legacy probability snapshot")
    require(torch.equal(weight, expected_weight), "legacy weight snapshot")
    require(calls == {"randperm": 1, "randint": 0}, "legacy RNG and donor-permutation counts")
    print("PASS legacy bitwise RGB/target/confidence/provenance/probability/weight")
    print("PASS legacy donor_permutation_calls=1 destination_rng_calls=0 early_return=1")


def test_legacy_tuple_matrix():
    cases = [
        (False, False, False, 4, ()),
        (True, False, False, 5, (4,)),
        (False, True, False, 5, (4,)),
        (True, True, False, 6, (4, 5)),
        (False, False, True, 6, (4, 5)),
        (True, False, True, 7, (4, 5, 6)),
        (False, True, True, 7, (4, 5, 6)),
        (True, True, True, 8, (4, 5, 6, 7)),
    ]
    for has_probs, has_weight, metadata, length, optional_positions in cases:
        result, values, _ = call_legacy(probs=has_probs, weight=has_weight, metadata=metadata)
        require(len(result) == length, f"tuple length {has_probs=} {has_weight=} {metadata=}")
        require(result[0].shape == values[0].shape, "tuple image order")
        require(result[1].shape == values[1].shape, "tuple target order")
        require(result[2].shape == values[2].shape, "tuple confidence order")
        require(result[3].shape == values[1].shape, "tuple provenance order")
        if metadata:
            require(result[4].shape == values[1].shape and result[5].shape == values[2].shape, "tuple metadata order")
        if has_probs:
            prob_position = 6 if metadata else 4
            require(result[prob_position].dim() == 4, "tuple probability order")
        if has_weight:
            weight_position = length - 1
            require(result[weight_position].dim() == 3, "tuple weight order")
        require(optional_positions == tuple(range(4, length)), "declared optional positions")
    print("PASS legacy tuple matrix lengths=4,5,5,6,6,7,7,8 and element order")


def generator_for_destination(mask, destination, search_limit=100000):
    for seed in range(search_limit):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = boundary_mix._translate_mask_to_random_valid_destination(mask, generator)
        if result["destination_extent"][:2] == destination:
            return seed, result
    raise AssertionError(f"destination {destination} not found")


def assert_translation_invariants(mask, result, name):
    translated = result["translated_mask"]
    src = mask.nonzero(as_tuple=False)
    dst = translated.nonzero(as_tuple=False)
    require(len(src) == len(dst), f"{name}: area")
    src_extent = result["source_extent"]
    dst_extent = result["destination_extent"]
    require(src_extent[2] - src_extent[0] == dst_extent[2] - dst_extent[0], f"{name}: height")
    require(src_extent[3] - src_extent[1] == dst_extent[3] - dst_extent[1], f"{name}: width")
    recovered = torch.zeros_like(mask)
    recovered[src_extent[0]:src_extent[2], src_extent[1]:src_extent[3]] = translated[
        dst_extent[0]:dst_extent[2], dst_extent[1]:dst_extent[3]
    ]
    require(torch.equal(recovered, mask), f"{name}: inverse translation")
    src_offsets = src - src.min(dim=0).values
    dst_offsets = dst - dst.min(dim=0).values
    require(torch.equal(src_offsets, dst_offsets), f"{name}: relative offsets")
    require(bool(((dst[:, 0] >= 0) & (dst[:, 0] < mask.shape[0])).all()), f"{name}: rows in bounds")
    require(bool(((dst[:, 1] >= 0) & (dst[:, 1] < mask.shape[1])).all()), f"{name}: cols in bounds")


def geometry_masks():
    masks = {}
    l_mask = torch.zeros(7, 8, dtype=torch.bool)
    l_mask[1:5, 2] = True
    l_mask[4, 2:6] = True
    masks["irregular_L"] = l_mask
    concave = torch.zeros(7, 8, dtype=torch.bool)
    concave[1:6, 1] = True
    concave[1, 1:6] = True
    concave[5, 1:6] = True
    masks["concave"] = concave
    ring = torch.zeros(7, 8, dtype=torch.bool)
    ring[1:6, 2:7] = True
    ring[2:5, 3:6] = False
    masks["ring_hole"] = ring
    fallback = torch.zeros(7, 8, dtype=torch.bool)
    fallback[2:5, 1:6] = True
    masks["fallback_rectangle"] = fallback
    masks["top_edge"] = torch.zeros(7, 8, dtype=torch.bool); masks["top_edge"][0, 2:5] = True
    masks["bottom_edge"] = torch.zeros(7, 8, dtype=torch.bool); masks["bottom_edge"][-1, 2:5] = True
    masks["left_edge"] = torch.zeros(7, 8, dtype=torch.bool); masks["left_edge"][2:5, 0] = True
    masks["right_edge"] = torch.zeros(7, 8, dtype=torch.bool); masks["right_edge"][2:5, -1] = True
    for name, row, col in (("top_left", 0, 0), ("top_right", 0, 7), ("bottom_left", 6, 0), ("bottom_right", 6, 7)):
        masks[name] = torch.zeros(7, 8, dtype=torch.bool); masks[name][row, col] = True
    masks["one_pixel"] = torch.zeros(7, 8, dtype=torch.bool); masks["one_pixel"][3, 4] = True
    masks["horizontal_thin"] = torch.zeros(7, 8, dtype=torch.bool); masks["horizontal_thin"][3, 1:7] = True
    masks["vertical_thin"] = torch.zeros(7, 8, dtype=torch.bool); masks["vertical_thin"][1:6, 4] = True
    masks["full_image"] = torch.ones(7, 8, dtype=torch.bool)
    return masks


def test_geometry_and_signed_displacements():
    for index, (name, mask) in enumerate(geometry_masks().items()):
        result = boundary_mix._translate_mask_to_random_valid_destination(
            mask, torch.Generator(device="cpu").manual_seed(100 + index)
        )
        assert_translation_invariants(mask, result, name)
        local = result["local_mask"]
        require(int(local.sum()) == int(mask.sum()), f"{name}: no rectangle leakage")
    full = geometry_masks()["full_image"]
    full_result = boundary_mix._translate_mask_to_random_valid_destination(full, torch.Generator().manual_seed(1))
    require(full_result["num_valid_translations"] == 1, "full-image support")
    require((full_result["delta_r"], full_result["delta_c"]) == (0, 0), "full-image zero")

    mask = geometry_masks()["irregular_L"]
    src_top, src_left = 1, 2
    destinations = {
        "positive_row": (2, 2), "negative_row": (0, 2),
        "positive_col": (1, 3), "negative_col": (1, 0),
        "mixed_sign": (0, 3), "partial_overlap": (2, 3),
    }
    observed = {}
    for name, destination in destinations.items():
        _, result = generator_for_destination(mask, destination)
        assert_translation_invariants(mask, result, name)
        observed[name] = (result["delta_r"], result["delta_c"], result["source_destination_iou"])
    require(observed["positive_row"][0] > 0 and observed["negative_row"][0] < 0, "signed rows")
    require(observed["positive_col"][1] > 0 and observed["negative_col"][1] < 0, "signed cols")
    require(observed["mixed_sign"][0] < 0 < observed["mixed_sign"][1], "mixed signs")
    require(0 < observed["partial_overlap"][2] < 1, "partial overlap IoU")
    print(f"PASS geometry masks={','.join(geometry_masks())}")
    print(f"PASS signed_displacements={observed}")


def test_relocated_tensor_provenance_and_diagnostics():
    batch, height, width = 2, 7, 8
    target_image = torch.full((batch, 3, height, width), -10.0)
    target_gt = torch.full((batch, height, width), -10, dtype=torch.long)
    target_conf = torch.full((batch, height, width), 0.25, dtype=torch.float16)
    probs_in = torch.ones(batch, 4, height, width)
    weight_in = torch.full((batch, height, width), 0.5)
    row = torch.arange(height).view(height, 1).expand(height, width)
    col = torch.arange(width).view(1, width).expand(height, width)
    donor_image = torch.zeros(batch, 3, height, width)
    donor_gt = torch.zeros(batch, height, width, dtype=torch.long)
    for b in range(batch):
        donor_image[b, 0] = b * 10000 + row * 100 + col
        donor_image[b, 1] = b * 10000 + row * 100 + col + 20000
        donor_image[b, 2] = b * 10000 + row * 100 + col + 40000
        donor_gt[b] = b * 1000 + row * 10 + col
    donor_gt[0, 4, 2] = 255
    masks = torch.zeros(batch, height, width, dtype=torch.bool)
    masks[0] = geometry_masks()["irregular_L"]
    masks[1, 1:4, 1:5] = True
    diagnostics = {}
    calls = {"randperm": 0, "randint": 0}
    original_randint = torch.randint

    def fixed_randperm(n, *args, **kwargs):
        calls["randperm"] += 1
        return torch.tensor([0, 1])

    def counted_randint(*args, **kwargs):
        calls["randint"] += 1
        return original_randint(*args, **kwargs)

    context = {"base_seed": 17, "rank": 0, "epoch": 2, "iteration": 9}
    with Patch(torch, "randperm", fixed_randperm), Patch(torch, "randint", counted_randint):
        result = boundary_mix.cut_mix_label_adaptive_with_mask(
            target_image, target_gt, target_conf, donor_image, donor_gt, [1.0, 1.0],
            unlabeled_probs=probs_in, unlabeled_weight=weight_in,
            labeled_masks=masks, direct_labeled_mix=True,
            direct_paste_policy="component_mask_random_valid_destination",
            direct_confidence_gate=False, destination_context=context,
            destination_diagnostics=diagnostics,
        )
    image, gt, conf, provenance, probs, weight = result
    require(calls == {"randperm": 1, "randint": 4}, "one permutation and two coordinates per nonempty target")
    require(diagnostics["relocation_draw_count"] == 2, "draw count")
    for i in range(batch):
        seed = boundary_mix._stable_s2_destination_seed(17, 0, 2, 9, i)
        tr = boundary_mix._translate_mask_to_random_valid_destination(masks[i], torch.Generator().manual_seed(seed))
        sh1, sw1, sh2, sw2 = tr["source_extent"]
        dh1, dw1, dh2, dw2 = tr["destination_extent"]
        local = tr["local_mask"]
        translated = tr["translated_mask"]
        require(torch.equal(image[i, :, dh1:dh2, dw1:dw2][:, local], donor_image[i, :, sh1:sh2, sw1:sw2][:, local]), "RGB coordinates")
        require(torch.equal(gt[i, dh1:dh2, dw1:dw2][local], donor_gt[i, sh1:sh2, sw1:sw2][local]), "GT coordinates")
        require(torch.equal(torch.sort(gt[i][translated]).values, torch.sort(donor_gt[i][masks[i]]).values), "GT multiset")
        require(bool((conf[i][translated] == 1).all()), "confidence convention")
        require(bool((probs[i, :, translated] == 0).all()), "probability convention")
        require(bool((weight[i][translated] == 1).all()), "weight convention")
        require(bool((provenance[i][translated] == 1).all()), "provenance convention")
        outside = ~translated
        require(torch.equal(image[i, :, outside], target_image[i, :, outside]), "outside RGB")
        require(torch.equal(gt[i, outside], target_gt[i, outside]), "outside target")
        require(torch.equal(conf[i, outside], target_conf[i, outside]), "outside confidence")
        require(torch.equal(probs[i, :, outside], probs_in[i, :, outside]), "outside probabilities")
        require(torch.equal(weight[i, outside], weight_in[i, outside]), "outside weight")
        require(bool((provenance[i, outside] == 0).all()), "outside provenance")
    require(bool((conf[gt == 255] == 1).all()), "ignore GT confidence")
    require(bool((weight[gt == 255] == 1).all()), "ignore GT weight")
    print(f"PASS tensor_provenance diagnostics={diagnostics} control_calls={calls}")


def test_stateless_rng_isolation():
    sequence_a = [boundary_mix._stable_s2_destination_seed(3, 1, 2, iteration, target) for iteration in range(8) for target in range(4)]
    sequence_b = [boundary_mix._stable_s2_destination_seed(4, 1, 2, iteration, target) for iteration in range(8) for target in range(4)]
    require(sequence_a == [boundary_mix._stable_s2_destination_seed(3, 1, 2, i, t) for i in range(8) for t in range(4)], "stable replay")
    require(sequence_a != sequence_b, "different base seed sequence")
    require(len(set(sequence_a[:4])) == 4, "independent target streams")

    mask = geometry_masks()["irregular_L"]
    torch.manual_seed(1234)
    cpu_before = torch.get_rng_state().clone()
    np.random.seed(1234)
    numpy_before = np.random.get_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    for seed in sequence_a:
        boundary_mix._translate_mask_to_random_valid_destination(mask, torch.Generator(device="cpu").manual_seed(seed))
    require(torch.equal(cpu_before, torch.get_rng_state()), "global CPU RNG isolation")
    numpy_after = np.random.get_state()
    require(numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1]) and numpy_before[2:] == numpy_after[2:], "global NumPy RNG isolation")
    if cuda_before is not None:
        require(all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all())), "global CUDA RNG isolation")
    print(f"PASS stateless_rng seeds={len(sequence_a)} cpu_unchanged=1 numpy_unchanged=1 cuda_unchanged={int(cuda_before is not None)}")


def test_zero_policy_and_empty_mask():
    full = torch.ones(3, 4, dtype=torch.bool)
    full_result = boundary_mix._translate_mask_to_random_valid_destination(full, torch.Generator().manual_seed(8))
    require(full_result["num_valid_translations"] == 1 and full_result["delta_r"] == 0 and full_result["delta_c"] == 0, "zero-only support")
    two_state = torch.ones(2, 3, dtype=torch.bool)
    two_state = torch.nn.functional.pad(two_state, (0, 0, 0, 1))
    states = set()
    for seed in range(50):
        result = boundary_mix._translate_mask_to_random_valid_destination(two_state, torch.Generator().manual_seed(seed))
        states.add(result["destination_extent"][:2])
    require(states == {(0, 0), (1, 0)}, "exactly two states including zero")

    empty = torch.zeros(1, 3, 4, dtype=torch.bool)
    diagnostics = {}
    calls = {"randint": 0}
    original_randint = torch.randint
    def counted_randint(*args, **kwargs):
        calls["randint"] += 1
        return original_randint(*args, **kwargs)
    with Patch(torch, "randperm", lambda n, *args, **kwargs: torch.tensor([0])), Patch(torch, "randint", counted_randint):
        boundary_mix.cut_mix_label_adaptive_with_mask(
            torch.zeros(1, 3, 3, 4), torch.zeros(1, 3, 4, dtype=torch.long), torch.zeros(1, 3, 4),
            torch.ones(1, 3, 3, 4), torch.ones(1, 3, 4, dtype=torch.long), [0.0],
            labeled_masks=empty, direct_labeled_mix=True,
            direct_paste_policy="component_mask_random_valid_destination",
            destination_context={"base_seed": 1, "rank": 0, "epoch": 0, "iteration": 0},
            destination_diagnostics=diagnostics,
        )
    require(diagnostics == {"relocation_empty_mask_count": 1}, "empty diagnostics")
    require(calls["randint"] == 0, "empty no coordinate draws")

    zero_mask = torch.zeros(3, 4, dtype=torch.bool); zero_mask[1:, 1:] = True
    _, zero_result = generator_for_destination(zero_mask, (1, 1))
    require(zero_result["num_valid_translations"] > 1 and (zero_result["delta_r"], zero_result["delta_c"]) == (0, 0), "random zero with alternatives")

    def mixer_diagnostics(mask, base_seed):
        stats = {}
        with Patch(torch, "randperm", lambda n, *args, **kwargs: torch.tensor([0])):
            boundary_mix.cut_mix_label_adaptive_with_mask(
                torch.zeros(1, 3, *mask.shape), torch.zeros(1, *mask.shape, dtype=torch.long), torch.zeros(1, *mask.shape),
                torch.ones(1, 3, *mask.shape), torch.ones(1, *mask.shape, dtype=torch.long), [0.0],
                labeled_masks=mask.unsqueeze(0), direct_labeled_mix=True,
                direct_paste_policy="component_mask_random_valid_destination",
                destination_context={"base_seed": base_seed, "rank": 0, "epoch": 0, "iteration": 0},
                destination_diagnostics=stats,
            )
        return stats

    zero_only_stats = mixer_diagnostics(full, 1)
    require(zero_only_stats["relocation_draw_count"] == 1, "zero-only draw")
    require(zero_only_stats["relocation_zero_count"] == 1 and zero_only_stats["relocation_zero_only_count"] == 1, "zero-only counters")
    require(zero_only_stats["relocation_random_zero_count"] == 0 and zero_only_stats["relocation_success_count"] == 0, "zero-only classifications")
    random_zero_stats = None
    for base_seed in range(10000):
        candidate = mixer_diagnostics(zero_mask, base_seed)
        if candidate["relocation_zero_count"] == 1:
            random_zero_stats = candidate
            break
    require(random_zero_stats is not None, "find random-zero context")
    require(random_zero_stats["relocation_random_zero_count"] == 1, "random-zero counter")
    require(random_zero_stats["relocation_zero_only_count"] == 0 and random_zero_stats["relocation_success_count"] == 0, "random-zero classification")
    for stats in (zero_only_stats, random_zero_stats):
        require(stats["relocation_zero_count"] + stats["relocation_nonzero_count"] == stats["relocation_draw_count"], "zero/nonzero accounting")
        require(stats["relocation_success_count"] == stats["relocation_nonzero_count"], "success accounting")
    print(f"PASS zero_policy two_state={sorted(states)} zero_only={zero_only_stats} random_zero={random_zero_stats} empty_draws=0")


def test_uniformity():
    sample_count = 12000
    absolute_tolerance = 0.08
    mask = torch.ones(2, 3, dtype=torch.bool)
    mask = torch.nn.functional.pad(mask, (0, 1, 0, 1))
    expected_states = {(r, c) for r in range(2) for c in range(2)}
    frequencies = {state: 0 for state in expected_states}
    for sample in range(sample_count):
        generator = torch.Generator(device="cpu").manual_seed(900000 + sample)
        result = boundary_mix._translate_mask_to_random_valid_destination(mask, generator)
        state = result["destination_extent"][:2]
        require(state in expected_states, "uniformity invalid state")
        frequencies[state] += 1
    expected_frequency = sample_count / len(expected_states)
    for state, frequency in frequencies.items():
        relative_error = abs(frequency - expected_frequency) / expected_frequency
        require(relative_error <= absolute_tolerance, f"uniformity {state} {relative_error}")
    observed_zero_rate = frequencies[(0, 0)] / sample_count
    expected_zero_rate = 1 / len(expected_states)
    print(f"PASS uniformity state_count={len(expected_states)} sample_count={sample_count} frequencies={dict(sorted(frequencies.items()))}")
    print(f"PASS uniformity expected_frequency={expected_frequency} tolerance={absolute_tolerance} observed_zero_rate={observed_zero_rate:.6f} expected_zero_rate={expected_zero_rate:.6f}")


def test_source_control_flow_and_logging():
    boundary_source = Path(ROOT, "util/boundary_mix.py").read_text()
    train_source = Path(ROOT, "train_semi.py").read_text()
    ast.parse(boundary_source)
    ast.parse(train_source)
    require("and direct_paste_policy == \"component_mask_random_valid_destination\"" in boundary_source, "isolated policy condition")
    require("return _return_with_optional_metadata()" in boundary_source, "direct early return")
    require("saliency_selector_exception_batch = 0" in train_source, "selector exception reset")
    handler_fragment = train_source[train_source.index("except Exception as exc:", train_source.index("if saliency_cutmix_enabled:")):]
    handler_fragment = handler_fragment[:handler_fragment.index("if csl_cutmix_enabled:")]
    require("saliency_selector_exception_batch = 1" in handler_fragment, "selector exception assignment")
    require("saliency_labeled_boxes = None" in handler_fragment and "saliency_labeled_masks = None" in handler_fragment, "selector outputs cleared")
    require("fallback=random_box" in handler_fragment, "legacy exception fallback logging")
    for forbidden in ("compute_csl", "thresholded_boundary_mix_loss", "compute_component_weights", "compute_js_boundary_compatibility_loss"):
        relocated_branch = boundary_source[boundary_source.index("component_mask_random_valid_destination", boundary_source.index("def cut_mix_label_adaptive_with_mask")):boundary_source.index("if direct_labeled_mix and labeled_masks is not None:", boundary_source.index("component_mask_random_valid_destination", boundary_source.index("def cut_mix_label_adaptive_with_mask")))]
        require(forbidden not in relocated_branch, f"forbidden path {forbidden}")
    print("PASS control_flow isolated_policy=1 early_return=1 no_gate=1 no_CSL=1 no_BoundaryMix=1 no_BCR=1 no_component_weight=1")
    print("PASS selector_exception clear_outputs=1 legacy_fallback_reachable=1 scalar_assignment_only=1")


def test_runtime_seed_plumbing():
    train_source = Path(ROOT, "train_semi.py").read_text()
    train_tree = ast.parse(train_source)
    require('cfg["_runtime_seed"]' not in train_source, "runtime seed leaked into shared config")

    train_defs = [node for node in train_tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"]
    require(len(train_defs) == 1, "single train definition")
    train_def = train_defs[0]
    require(train_def.args.args[-1].arg == "runtime_seed", "runtime_seed is trailing train argument")
    require(isinstance(train_def.args.defaults[-1], ast.Constant) and train_def.args.defaults[-1].value == 0, "runtime_seed default")

    main_def = next(node for node in train_tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    production_calls = [
        node for node in ast.walk(main_def)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "train"
    ]
    require(len(production_calls) == 1, "single production train call")
    seed_keywords = [keyword for keyword in production_calls[0].keywords if keyword.arg == "runtime_seed"]
    require(len(seed_keywords) == 1, "explicit runtime_seed keyword")
    seed_value = seed_keywords[0].value
    require(
        isinstance(seed_value, ast.Call)
        and isinstance(seed_value.func, ast.Name)
        and seed_value.func.id == "int"
        and len(seed_value.args) == 1
        and isinstance(seed_value.args[0], ast.Attribute)
        and isinstance(seed_value.args[0].value, ast.Name)
        and seed_value.args[0].value.id == "args"
        and seed_value.args[0].attr == "seed",
        "main passes int(args.seed)",
    )
    require('"base_seed": runtime_seed' in train_source, "destination context uses runtime_seed")
    require("if s2_relocated_policy_active:" in train_source, "relocation plumbing policy guard")
    require(train_source.count("**destination_kwargs,") == 2, "both mixer calls use guarded destination kwargs")
    print("PASS runtime_seed explicit_train_argument=1 shared_config_mutation=0 destination_context=1")
    print("PASS destination_plumbing guarded_call_sites=2 legacy_active_diagnostics=0")


def main():
    test_legacy_same_coordinate_snapshot()
    test_legacy_tuple_matrix()
    test_geometry_and_signed_displacements()
    test_relocated_tensor_provenance_and_diagnostics()
    test_stateless_rng_isolation()
    test_zero_policy_and_empty_mask()
    test_uniformity()
    test_source_control_flow_and_logging()
    test_runtime_seed_plumbing()
    print("PASS S2 RELOCATED COMPLETE SMOKE")


if __name__ == "__main__":
    main()
