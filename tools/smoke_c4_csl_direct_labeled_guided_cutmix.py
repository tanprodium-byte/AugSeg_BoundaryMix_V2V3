#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import util.boundary_mix as boundary_mix  # noqa: E402
from util.boundary_mix import (  # noqa: E402
    compute_c4_direct_mix_stats,
    cut_mix_label_adaptive_c4_direct_labeled,
)
from util.csl_cutmix import get_csl_guided_boxes  # noqa: E402


CONFIG = (
    ROOT
    / "exps/boundary_mix_v2_v3/voc_semi662"
    / "c4_csl_official_direct_labeled_guided_cutmix_plus_ce_weight/config.yaml"
)
TRAIN = ROOT / "train_semi.py"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def make_inputs():
    batch, height, width = 2, 5, 6
    unlabeled_image = torch.empty(batch, 3, height, width)
    unlabeled_image[0].fill_(10.0)
    unlabeled_image[1].fill_(20.0)
    unlabeled_target = torch.empty(batch, height, width, dtype=torch.long)
    unlabeled_target[0].fill_(2)
    unlabeled_target[1].fill_(3)
    unlabeled_logits = torch.empty(batch, height, width, dtype=torch.float16)
    unlabeled_logits[0].fill_(0.2)
    unlabeled_logits[1].fill_(0.8)
    unlabeled_weight = torch.empty(batch, height, width)
    unlabeled_weight[0].fill_(0.25)
    unlabeled_weight[1].fill_(0.75)

    labeled_image = torch.empty(batch, 3, height, width)
    labeled_target = torch.empty(batch, height, width, dtype=torch.long)
    for donor in range(batch):
        for y in range(height):
            for x in range(width):
                marker = 1000 * (donor + 1) + 10 * y + x
                labeled_image[donor, :, y, x] = float(marker)
                labeled_target[donor, y, x] = 10 * (donor + 1) + y
    labeled_target[1, 2, 2] = 255

    # Box contract: [row1, col1, row2, col2]
    boxes = torch.tensor(
        [
            [1, 2, 4, 5],
            [0, 2, 2, 6],
        ],
        dtype=torch.long,
    )
    unlabeled_target[0, 1:4, 2:5] = labeled_target[1, 1:4, 2:5]
    return (
        unlabeled_image,
        unlabeled_target,
        unlabeled_logits,
        labeled_image,
        labeled_target,
        unlabeled_weight,
        boxes,
    )


class ControlledRNG:
    def __init__(self, draws, permutation):
        self.draws = iter(draws)
        self.permutation = torch.as_tensor(permutation, dtype=torch.long)
        self.random_calls = 0
        self.randperm_calls = 0

    def random(self, *args, **kwargs):
        del args, kwargs
        self.random_calls += 1
        return next(self.draws)

    def randperm(self, n, *args, **kwargs):
        del args
        self.randperm_calls += 1
        require(n == self.permutation.numel(), "unexpected randperm size")
        return self.permutation.to(kwargs.get("device", torch.device("cpu")))

    def forbidden(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("C4 stats helper must not call any additional RNG")


def run_controlled(reliabilities, draws=(0.5, 0.5), permutation=(1, 0)):
    inputs = make_inputs()
    originals = tuple(t.clone() for t in inputs)
    rng = ControlledRNG(draws, permutation)
    old_random = np.random.random
    old_uniform = np.random.uniform
    old_randperm = torch.randperm
    old_rand = torch.rand
    old_multinomial = torch.multinomial
    old_rand_bbox = boundary_mix._rand_bbox
    try:
        np.random.random = rng.random
        np.random.uniform = rng.forbidden
        torch.randperm = rng.randperm
        torch.rand = rng.forbidden
        torch.multinomial = rng.forbidden

        def forbidden_source_box_rng(*args, **kwargs):
            del args, kwargs
            raise AssertionError("C4 must not sample a source box")

        boundary_mix._rand_bbox = forbidden_source_box_rng
        result = cut_mix_label_adaptive_c4_direct_labeled(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            reliabilities,
            inputs[6],
            unlabeled_weight=inputs[5],
            ignore_index=255,
        )
        stats = compute_c4_direct_mix_stats(
            result[3],
            result[1],
            inputs[6],
            ignore_index=255,
        )
    finally:
        np.random.random = old_random
        np.random.uniform = old_uniform
        torch.randperm = old_randperm
        torch.rand = old_rand
        torch.multinomial = old_multinomial
        boundary_mix._rand_bbox = old_rand_bbox
    return inputs, originals, rng, result, stats


def test_gate_pairing_geometry_and_provenance():
    inputs, originals, rng, result, stats = run_controlled([0.2, 0.8])
    image, target, logits, source_mask, weight = result
    u_image, u_target, u_logits, labeled_image, labeled_target, u_weight, boxes = inputs

    require(rng.randperm_calls == 1, "C4 must use exactly one labeled donor permutation")
    require(rng.random_calls == 2, "C4 must draw one adaptive gate value per final target, in order")
    require(len(result) == 5, "C4 weighted-CE caller tuple arity must be five")
    require(image.shape == u_image.shape and image.dtype == u_image.dtype and image.device == u_image.device, "image contract")
    require(target.shape == u_target.shape and target.dtype == u_target.dtype and target.device == u_target.device, "target contract")
    require(logits.shape == u_logits.shape and logits.dtype == u_logits.dtype, "confidence metadata contract")
    require(source_mask.shape == u_target.shape and source_mask.dtype == torch.float32, "source-mask contract")
    require(weight.shape == u_weight.shape and weight.dtype == u_weight.dtype, "weight contract")

    row1, col1, row2, col2 = boxes[0].tolist()
    require(
        torch.equal(image[0, :, row1:row2, col1:col2], labeled_image[1, :, row1:row2, col1:col2]),
        "target 0 must receive same-coordinate RGB from donor 1",
    )
    require(
        torch.equal(target[0, row1:row2, col1:col2], labeled_target[1, row1:row2, col1:col2]),
        "RGB and GT must use the same labeled donor and coordinates",
    )
    require(source_mask[0, row1:row2, col1:col2].eq(1).all(), "successful rectangle must be labeled provenance")
    outside = torch.ones_like(source_mask[0], dtype=torch.bool)
    outside[row1:row2, col1:col2] = False
    require(torch.equal(image[0, :, outside], u_image[0, :, outside]), "outside image must remain final target 0")
    require(torch.equal(target[0, outside], u_target[0, outside]), "outside pseudo-label must remain final target 0")
    require(torch.equal(weight[0, outside], u_weight[0, outside]), "outside CSL weight must remain final target 0")
    require(source_mask[0, outside].eq(0).all(), "outside provenance must remain zero")

    require(torch.equal(image[1], u_image[1]), "gate-fail target image must remain unchanged")
    require(torch.equal(target[1], u_target[1]), "gate-fail target pseudo-label must remain unchanged")
    require(torch.equal(logits[1], u_logits[1]), "gate-fail confidence metadata must remain unchanged")
    require(torch.equal(weight[1], u_weight[1]), "gate-fail CSL weight must remain unchanged")
    require(source_mask[1].eq(0).all(), "gate-fail provenance must be all zero")

    valid = target[0, row1:row2, col1:col2].ne(255)
    pasted_weight = weight[0, row1:row2, col1:col2]
    require(pasted_weight[valid].eq(1).all(), "valid labeled pixels must receive exact weight one")
    require(pasted_weight[~valid].eq(0).all(), "labeled ignore pixels must store zero weight")
    require(target[0, 2, 2].item() == 255, "labeled ignore target must remain 255")
    require(
        torch.equal(target[0, row1:row2, col1:col2], originals[1][0, row1:row2, col1:col2]),
        "fixture must prove pasted labels can equal the previous pseudo-labels",
    )

    expected_stats = {
        "c4/gate_attempted_count": 2.0,
        "c4/gate_pass_count": 1.0,
        "c4/gate_pass_ratio": 0.5,
        "c4/mixed_sample_count": 1.0,
        "c4/mixed_sample_ratio": 0.5,
        "c4/pasted_pixel_count": 9.0,
        "c4/pasted_pixel_ratio": 9.0 / (2 * 5 * 6),
        "c4/selected_box_area_mean": 8.5,
        "c4/selected_box_area_ratio_mean": 8.5 / (5 * 6),
        "c4/valid_labeled_pasted_pixel_count": 8.0,
        "c4/valid_labeled_pixel_ratio": 8.0 / 9.0,
        "c4/ignore_labeled_pasted_pixel_count": 1.0,
        "c4/ignore_labeled_pixel_ratio": 1.0 / 9.0,
    }
    require(set(stats) == set(expected_stats), "C4 stats must return exactly the production metric keys")
    for key, expected in expected_stats.items():
        require(abs(stats[key] - expected) < 1e-7, f"unexpected {key}: {stats[key]} != {expected}")
    require(
        stats["c4/pasted_pixel_count"] == float(source_mask.sum().item()),
        "pasted-pixel count must come from provenance",
    )
    require(
        stats["c4/pasted_pixel_ratio"] == float(source_mask.sum().item()) / float(source_mask.numel()),
        "pasted-pixel ratio must come from provenance, not label differences",
    )
    require(
        stats["c4/valid_labeled_pasted_pixel_count"] + stats["c4/ignore_labeled_pasted_pixel_count"]
        == stats["c4/pasted_pixel_count"],
        "valid and ignore pasted counts must partition provenance",
    )
    require(rng.randperm_calls == 1, "stats helper must not add a torch.randperm call")
    require(rng.random_calls == 2, "stats helper must not add an np.random.random call")

    for original, after in zip(originals, inputs):
        require(torch.equal(original, after), "C4 must not mutate any input tensor")


def test_gate_is_owned_by_final_target():
    _, _, _, result, _ = run_controlled([0.8, 0.2], draws=(0.5, 0.5), permutation=(1, 0))
    image, target, _, source_mask, weight = result
    base = make_inputs()
    require(source_mask[0].sum().item() == 0, "target 0 must use reliability[0], not donor/target 1 reliability")
    require(source_mask[1].sum().item() == 8, "target 1 must use reliability[1]")
    require(torch.equal(image[0], base[0][0]), "failed target 0 must remain itself")
    require(torch.equal(target[0], base[1][0]), "failed target 0 pseudo-label must remain itself")
    require(torch.equal(weight[0], base[5][0]), "failed target 0 weight must remain itself")


def expect_value_error(call, text):
    try:
        call()
    except ValueError as exc:
        require(text in str(exc), f"missing error context {text!r}: {exc}")
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def test_fail_fast_validation():
    inputs = make_inputs()
    common = dict(unlabeled_weight=inputs[5], ignore_index=255)

    expect_value_error(
        lambda: cut_mix_label_adaptive_c4_direct_labeled(
            inputs[0], inputs[1], inputs[2], inputs[3][:1], inputs[4][:1], [0.0, 0.0], inputs[6], **common
        ),
        "equal labeled and unlabeled batch sizes",
    )
    expect_value_error(
        lambda: cut_mix_label_adaptive_c4_direct_labeled(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4][:, :-1],
            [0.0, 0.0],
            inputs[6],
            **common,
        ),
        "labeled image/GT spatial mismatch",
    )

    invalid = inputs[6].clone()
    invalid[0] = torch.tensor([-1, 1, 4, 4])
    rng = ControlledRNG((0.5, 0.5), (1, 0))
    old_random, old_randperm = np.random.random, torch.randperm
    try:
        np.random.random, torch.randperm = rng.random, rng.randperm
        expect_value_error(
            lambda: cut_mix_label_adaptive_c4_direct_labeled(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], [0.0, 0.0], invalid, **common
            ),
            "invalid target box for target 0",
        )
    finally:
        np.random.random, torch.randperm = old_random, old_randperm


def test_ignore_weighted_ce_semantics():
    _, _, _, result, _ = run_controlled([0.0, 1.0], draws=(0.5, 0.5))
    _, target, _, _, weight = result
    logits = torch.randn(2, 32, 5, 6)
    valid = target.ne(255)
    effective_weight = weight * valid
    ce = F.cross_entropy(logits, target, reduction="none", ignore_index=255)
    denominator = effective_weight.sum().clamp_min(1e-6)
    loss = (ce * effective_weight).sum() / denominator
    require(torch.isfinite(loss), "weighted CE must be finite")
    require(effective_weight[target == 255].eq(0).all(), "ignore pixels must have zero effective CE contribution")
    require(
        denominator.item() == effective_weight[target != 255].sum().item(),
        "ignore pixels must not increase weighted denominator",
    )


class FailingSampler:
    def __init__(self):
        self.calls = 0

    def __call__(self, size, lam=None):
        del size, lam
        self.calls += 1
        raise ValueError("controlled candidate failure")


def fixed_fallback_sampler(size, lam=None):
    del lam
    batch = size[0]
    return (
        np.zeros(batch, dtype=np.int64),
        np.zeros(batch, dtype=np.int64),
        np.full(batch, 2, dtype=np.int64),
        np.full(batch, 2, dtype=np.int64),
    )


def test_strict_failure_and_default_fallback():
    reliability = torch.zeros(2, 5, 6)
    failing = FailingSampler()
    try:
        get_csl_guided_boxes(reliability, failing, num_candidates=2, strict=True)
    except RuntimeError as exc:
        require("strict CSL guided box selection failed" in str(exc), "strict error must include batch context")
        require(isinstance(exc.__cause__, ValueError), "strict error must preserve original exception chaining")
    else:
        raise AssertionError("C4 strict CSL selection must raise instead of falling back")
    require(failing.calls == 1, "strict path must not consume fallback RNG")

    boxes, stats = get_csl_guided_boxes(reliability, fixed_fallback_sampler, num_candidates=0)
    require(tuple(boxes.shape) == (2, 4), "default fallback box shape")
    require(stats["csl_cutmix/fallback_ratio"] == 1.0, "default/C3 fallback behavior must remain enabled")


def test_config_and_isolated_dispatch():
    cfg = yaml.safe_load(CONFIG.read_text())
    csl = cfg["csl"]
    require(csl["mode"] == "official_direct_labeled_guided_cutmix_plus_ce_weight", "exact C4 mode")
    require(cfg["trainer"]["unsupervised"]["use_cutmix"] is True, "outer CutMix enabled")
    require(float(cfg["trainer"]["unsupervised"]["use_cutmix_trigger_prob"]) == 1.0, "C1 trigger parity")
    require(csl["use_csl_for_mix_confidence"] is True, "CSL sample gate")
    require(csl["use_csl_for_ce_weight"] is True, "CSL weighted CE")
    require(csl["use_csl_for_cutmix"] is True and cfg["csl_cutmix"]["enabled"] is True, "guided boxes")
    require(csl["perturb_input"] is False, "no input perturbation")
    require(cfg["boundary_compatibility"]["enabled"] is False, "no BCR")
    require(cfg["boundary_mix"]["enabled"] is False, "no BoundaryMix")
    require(cfg["boundary_component"]["enabled"] is False, "no component weighting")
    require(cfg["saliency_cutmix"]["enabled"] is False, "no saliency CutMix")

    source = TRAIN.read_text()
    exact_guard = 'csl_mode == "official_direct_labeled_guided_cutmix_plus_ce_weight"'
    require(exact_guard in source, "C4 dispatch must be guarded by the exact opt-in mode")
    require("if csl_c4_direct_labeled_enabled:" in source, "missing isolated C4 dispatch")
    require(
        "c4_target_boxes = csl_target_boxes[:, [1, 0, 3, 2]]"
        not in source,
        "C4 must not transpose row/col boxes",
    )
    require(
        any(
            line.strip() == "c4_target_boxes = csl_target_boxes"
            for line in source.splitlines()
        ),
        "C4 must pass row/col boxes directly",
    )
    require(
        source.index("if ar_applied:") < source.index("if csl_c4_direct_labeled_enabled:", source.index("if ar_applied:")),
        "C4 RNG dispatch must remain inside the existing outer trigger",
    )
    c4_mode = "official_direct_labeled_guided_cutmix_plus_ce_weight"
    for old_mode in (
        "official_reliability_replace_confidence",
        "official_reliable_mask_perturbation",
        "official_guided_cutmix",
    ):
        require(old_mode != c4_mode, f"{old_mode} must not satisfy the exact C4 mode predicate")

    require("c4_stats = None" in source, "C4 stats must remain None unless the C4 branch populates it")
    require(
        source.count("c4_stats = compute_c4_direct_mix_stats(") == 1,
        "C4 stats must have exactly one production call site",
    )
    stats_call = source.index("c4_stats = compute_c4_direct_mix_stats(")
    c4_branch = source.rfind("if csl_c4_direct_labeled_enabled:", 0, stats_call)
    next_branch = source.index("elif boundary_component_enabled:", stats_call)
    require(c4_branch < stats_call < next_branch, "C4 stats computation must remain inside the exact C4 branch")

    log_guard = source.index("if c4_stats is not None:")
    log_guard_end = source.index("if torch.cuda.is_available():", log_guard)
    require(
        "log_dict[key] = value" in source[log_guard:log_guard_end],
        "C4 values must enter log_dict only when c4_stats is populated",
    )
    require(
        '"official_direct_labeled_guided_cutmix_plus_ce_weight": 6.0' in source,
        "C4 mode metadata must map to 6.0 instead of the -1.0 fallback",
    )

    c4_instrumentation = source[stats_call:next_branch]
    require("[c4]" in c4_instrumentation, "missing C4 terminal debug prefix")
    require("rank == 0" in c4_instrumentation, "C4 terminal debug line must be rank-zero only")
    require(
        "csl_cutmix_debug_enabled" in c4_instrumentation,
        "C4 terminal debug line must use csl_cutmix.debug_log",
    )
    require(
        'step < int(csl_cutmix_cfg.get("debug_first_batches", 3))' in c4_instrumentation
        and "or do_log_now" in c4_instrumentation,
        "C4 terminal debug cadence must use first batches or the existing log cadence",
    )


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    test_gate_pairing_geometry_and_provenance()
    test_gate_is_owned_by_final_target()
    test_fail_fast_validation()
    test_ignore_weighted_ce_semantics()
    test_strict_failure_and_default_fallback()
    test_config_and_isolated_dispatch()
    print("C4 CSL DIRECT LABELED GUIDED CUTMIX SMOKE PASSED")


if __name__ == "__main__":
    main()
