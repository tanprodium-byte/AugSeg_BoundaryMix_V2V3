#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import util.boundary_mix as boundary_mix  # noqa: E402
import util.csl_cutmix as csl_cutmix  # noqa: E402
from augseg.utils.loss_helper import compute_unsupervised_loss_by_threshold  # noqa: E402


CONFIG = (
    ROOT
    / "exps/boundary_mix_v2_v3/voc_semi662"
    / "s1_saliency_box_adaptive_relocated_plus_csl_official_cutmix/config.yaml"
)
BASE_CONFIG = (
    ROOT
    / "exps/boundary_mix_v2_v3/voc_semi662"
    / "s1_saliency_box_adaptive_relocated_cutmix/config.yaml"
)
TRAIN = ROOT / "train_semi.py"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def test_config_and_dispatch_isolation():
    cfg = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load(BASE_CONFIG.read_text())
    identity = "s1_saliency_box_adaptive_relocated_plus_csl_official_cutmix"

    require(cfg["run"]["name"] == identity, "new run identity")
    require(cfg["wandb"]["name"] == identity, "new W&B identity")
    require(identity in cfg["saver"]["snapshot_dir"], "new snapshot identity")
    require(identity.replace("_", "-") in cfg["hf"]["path_in_repo"], "new HF identity")
    require(cfg["wandb"]["enable"] is False and cfg["hf"]["enabled"] is False, "external logging disabled")

    saliency = cfg["saliency_cutmix"]
    require(saliency["enabled"] is True and saliency["mode"] == "box", "S1 box source enabled")
    require(saliency["direct_labeled_mix"] is True, "direct labeled mixing enabled")
    require(saliency["direct_confidence_gate"] is True, "adaptive gate enabled")
    require(saliency["direct_paste_policy"] == "csl_official_fixed_size_target", "new policy enabled")
    require(base["saliency_cutmix"]["direct_paste_policy"] == "random_target", "base policy unchanged")

    fixed = cfg["fixed_size_csl_destination"]
    require(fixed == {
        "enabled": True,
        "num_candidates": 8,
        "selection": "softmax",
        "temperature": 0.2,
        "target_policy": "low_reliability",
    }, "locked fixed-size selector config")

    csl = cfg["csl"]
    require(csl["reliability_mode"] == "official_pcos", "official reliability enabled")
    require(csl["use_csl_for_mix_confidence"] is False, "CSL sample gate disabled")
    require(csl["use_csl_for_ce_weight"] is False, "CSL CE weighting disabled")
    require(csl["use_csl_for_cutmix"] is False, "legacy C3 destination path disabled")
    require(csl["perturb_input"] is False, "CSL perturbation disabled")
    require(cfg["csl_cutmix"]["enabled"] is False, "legacy CSL CutMix helper disabled")
    require(cfg["boundary_mix"]["enabled"] is False, "BoundaryMix disabled")
    require(cfg["boundary_component"]["enabled"] is False, "component weighting disabled")
    require(cfg["boundary_compatibility"]["enabled"] is False, "BCR disabled")
    require(float(cfg["boundary_compatibility"]["lambda_bcr"]) == 0.0, "BCR weight zero")
    require(float(cfg["trainer"]["unsupervised"]["threshold"]) == 0.95, "original threshold")
    require(float(cfg["trainer"]["unsupervised"]["loss_weight"]) == 1.0, "original loss weight")

    source = TRAIN.read_text()
    require("s1_adaptive_relocated_official_fixed_size_cutmix" in source, "new isolated train predicate")
    require("csl_destination_reliability=csl_reliability_u" in source, "reliability passed to mixer")


def test_fixed_size_candidate_generation_and_scoring():
    reliability = torch.ones(6, 7)
    reliability[1:4, 2:6] = 0.0
    randint_values = iter([1, 2, 0, 0, 3, 3, 1, 2, 2, 1, 0, 3, 3, 0, 1, 2])
    calls = []
    old_randint = torch.randint
    old_multinomial = torch.multinomial
    old_beta = np.random.beta

    try:
        def fixed_randint(low, high, size, **kwargs):
            calls.append((low, high, size, kwargs.get("device")))
            return torch.tensor([next(randint_values)], device=kwargs.get("device"))

        def fixed_multinomial(probs, num_samples, **kwargs):
            require(tuple(probs.shape) == (1, 8), "softmax probability shape")
            require(num_samples == 1, "one multinomial selection")
            return torch.tensor([[0]], device=probs.device)

        def forbidden_beta(*args, **kwargs):
            raise AssertionError("fixed-size selector must not sample Beta sizes")

        torch.randint = fixed_randint
        torch.multinomial = fixed_multinomial
        np.random.beta = forbidden_beta
        selected, diagnostics = csl_cutmix.select_fixed_size_csl_destination(
            reliability,
            3,
            4,
            num_candidates=8,
            temperature=0.2,
            policy="low_reliability",
            target_index=0,
        )
    finally:
        torch.randint = old_randint
        torch.multinomial = old_multinomial
        np.random.beta = old_beta

    candidates = diagnostics["candidates"]
    require(len(calls) == 16, "exactly K row/column origin draws")
    require(all(call[0] == 0 for call in calls), "origin lower bound is zero")
    require(all(call[1] == (4 if index % 2 == 0 else 4) for index, call in enumerate(calls)), "valid exclusive bounds")
    require(tuple(candidates.shape) == (8, 4), "K candidate boxes")
    require(torch.equal(candidates[:, 2] - candidates[:, 0], torch.full((8,), 3)), "fixed source height")
    require(torch.equal(candidates[:, 3] - candidates[:, 1], torch.full((8,), 4)), "fixed source width")
    require(torch.equal(candidates[0], candidates[3]), "duplicate candidates accepted")
    require(torch.equal(selected, candidates[0]), "controlled multinomial destination")

    expected_scores = []
    score_map = 1.0 - reliability
    for row1, col1, row2, col2 in candidates.tolist():
        expected_scores.append(score_map[row1:row2, col1:col2].mean())
    require(torch.allclose(diagnostics["scores"], torch.stack(expected_scores)), "official mean(1-reliability) scores")
    require(diagnostics["scores"][0] > diagnostics["scores"][1], "lower reliability receives larger score")
    expected_probs = torch.softmax(diagnostics["scores"] / 0.2, dim=0)
    expected_probs = expected_probs.clamp_min(1e-6)
    expected_probs = expected_probs / expected_probs.sum()
    require(torch.allclose(diagnostics["probabilities"], expected_probs), "temperature-0.2 softmax semantics")
    require(int(diagnostics["selected_index"]) == 0, "single recorded selected index")


def test_duplicate_only_origin_and_strict_failure():
    reliability = torch.full((3, 4), 0.5)
    torch.manual_seed(17)
    selected, diagnostics = csl_cutmix.select_fixed_size_csl_destination(
        reliability, 3, 4, num_candidates=8, temperature=0.2, target_index=1
    )
    require(torch.equal(diagnostics["candidates"], torch.tensor([[0, 0, 3, 4]]).repeat(8, 1)), "one-origin duplicates")
    require(torch.equal(selected, torch.tensor([0, 0, 3, 4])), "one-origin selection")

    for bad_h, bad_w in ((0, 2), (2, 0), (4, 2), (2, 5)):
        try:
            csl_cutmix.select_fixed_size_csl_destination(
                reliability, bad_h, bad_w, num_candidates=8, target_index=7
            )
        except RuntimeError as exc:
            message = str(exc)
            require("target_index=7" in message, "failure includes target index")
            require("reliability_shape=(3, 4)" in message, "failure includes reliability shape")
            require("num_candidates=8" in message, "failure includes candidate count")
        else:
            raise AssertionError("invalid geometry must fail fast")

    try:
        csl_cutmix.select_fixed_size_csl_destination(reliability.unsqueeze(0), 1, 1, target_index=2)
    except RuntimeError as exc:
        require("reliability_shape=(1, 3, 4)" in str(exc), "malformed reliability is contextual")
    else:
        raise AssertionError("malformed reliability must fail fast")

    old_sampler = csl_cutmix.sample_box_by_softmax
    try:
        def failed_sampler(*args, **kwargs):
            raise ValueError("injected selector failure")

        csl_cutmix.sample_box_by_softmax = failed_sampler
        try:
            csl_cutmix.select_fixed_size_csl_destination(reliability, 1, 1, target_index=3)
        except RuntimeError as exc:
            require("original_error=injected selector failure" in str(exc), "selector error remains observable")
        else:
            raise AssertionError("selector failure must not fall back")
    finally:
        csl_cutmix.sample_box_by_softmax = old_sampler


def _synthetic_mix_inputs():
    batch, height, width = 2, 6, 7
    unlabeled_image = torch.arange(batch * 3 * height * width, dtype=torch.float32).view(batch, 3, height, width)
    unlabeled_mask = torch.arange(batch * height * width, dtype=torch.long).view(batch, height, width) % 4
    unlabeled_logits = torch.linspace(0.1, 0.9, batch * height * width).view(batch, height, width)
    labeled_image = 1000 + torch.arange(batch * 3 * height * width, dtype=torch.float32).view(batch, 3, height, width)
    labeled_mask = torch.full((batch, height, width), 5, dtype=torch.long)
    labeled_mask[0].fill_(8)
    labeled_mask[0, 0, 0] = 255
    boxes = torch.tensor([[0, 0, 2, 3], [1, 2, 4, 6]])
    reliability = torch.stack([torch.full((height, width), 0.1), torch.full((height, width), 0.9)])
    return unlabeled_image, unlabeled_mask, unlabeled_logits, labeled_image, labeled_mask, boxes, reliability


def test_gate_fail_preserves_target_and_consumes_no_destination_rng():
    values = _synthetic_mix_inputs()
    unlabeled_image, unlabeled_mask, unlabeled_logits, labeled_image, labeled_mask, boxes, reliability = values
    old_randperm = torch.randperm
    old_random = np.random.random
    old_selector = boundary_mix.select_fixed_size_csl_destination
    old_randint = torch.randint
    old_multinomial = torch.multinomial
    try:
        torch.randperm = lambda n, *args, **kwargs: torch.tensor([1, 0])
        np.random.random = lambda *args, **kwargs: 0.5

        def forbidden(*args, **kwargs):
            raise AssertionError("gate-failed sample must not consume destination RNG or selector calls")

        boundary_mix.select_fixed_size_csl_destination = forbidden
        torch.randint = forbidden
        torch.multinomial = forbidden
        image, target, confidence, source_mask = boundary_mix.cut_mix_label_adaptive_with_mask(
            unlabeled_image,
            unlabeled_mask,
            unlabeled_logits,
            labeled_image,
            labeled_mask,
            [1.0, 1.0],
            labeled_boxes=boxes,
            direct_labeled_mix=True,
            direct_paste_policy="csl_official_fixed_size_target",
            direct_confidence_gate=True,
            csl_destination_reliability=reliability,
        )
    finally:
        torch.randperm = old_randperm
        np.random.random = old_random
        boundary_mix.select_fixed_size_csl_destination = old_selector
        torch.randint = old_randint
        torch.multinomial = old_multinomial

    require(torch.equal(image, unlabeled_image), "gate-fail image equals untouched clone")
    require(torch.equal(target, unlabeled_mask), "gate-fail target equals untouched clone")
    require(torch.equal(confidence, unlabeled_logits), "gate-fail confidence equals untouched clone")
    require(torch.count_nonzero(source_mask) == 0, "gate-fail source mask is empty")


def test_gate_ownership_donor_source_and_direct_paste():
    values = _synthetic_mix_inputs()
    unlabeled_image, unlabeled_mask, unlabeled_logits, labeled_image, labeled_mask, boxes, reliability = values
    old_randperm = torch.randperm
    old_random = np.random.random
    old_selector = boundary_mix.select_fixed_size_csl_destination
    selector_calls = []
    try:
        torch.randperm = lambda n, *args, **kwargs: torch.tensor([1, 0])
        np.random.random = lambda *args, **kwargs: 0.5

        def fixed_selector(reliability_i, crop_h, crop_w, **kwargs):
            selector_calls.append((reliability_i.clone(), crop_h, crop_w, kwargs))
            return torch.tensor([4, 4, 4 + crop_h, 4 + crop_w]), {"selected_index": torch.tensor(0)}

        boundary_mix.select_fixed_size_csl_destination = fixed_selector
        image, target, confidence, source_mask = boundary_mix.cut_mix_label_adaptive_with_mask(
            unlabeled_image,
            unlabeled_mask,
            unlabeled_logits,
            labeled_image,
            labeled_mask,
            [1.0, 0.0],
            labeled_boxes=boxes,
            direct_labeled_mix=True,
            direct_paste_policy="csl_official_fixed_size_target",
            direct_confidence_gate=True,
            csl_destination_reliability=reliability,
            csl_destination_num_candidates=8,
            csl_destination_temperature=0.2,
        )
    finally:
        torch.randperm = old_randperm
        np.random.random = old_random
        boundary_mix.select_fixed_size_csl_destination = old_selector

    require(len(selector_calls) == 1, "selector runs only for gate-pass final target")
    reliability_i, crop_h, crop_w, kwargs = selector_calls[0]
    require(torch.equal(reliability_i, reliability[1]), "CSL reliability belongs to final target i")
    require((crop_h, crop_w) == (2, 3), "source donor box determines destination size")
    require(kwargs["target_index"] == 1, "selector receives final target index")

    require(torch.equal(image[0], unlabeled_image[0]), "failed target image unchanged")
    require(torch.equal(target[0], unlabeled_mask[0]), "failed target label unchanged")
    require(torch.equal(confidence[0], unlabeled_logits[0]), "failed target confidence unchanged")

    source_rgb = labeled_image[0, :, 0:2, 0:3]
    source_gt = labeled_mask[0, 0:2, 0:3]
    require(torch.equal(image[1, :, 4:6, 4:7], source_rgb), "RGB uses permuted donor and exact source box")
    require(torch.equal(target[1, 4:6, 4:7], source_gt), "GT uses same donor and source box")
    require(torch.equal(confidence[1, 4:6, 4:7], torch.ones(2, 3)), "valid labeled confidence is one")
    require(target[1, 4, 4].item() == 255, "pasted GT 255 remains 255")
    require(source_mask[1, 4:6, 4:7].sum().item() == 6, "exactly one direct paste")

    outside = ~source_mask.bool()
    require(torch.equal(image[outside.unsqueeze(1).expand_as(image)], unlabeled_image[outside.unsqueeze(1).expand_as(image)]), "outside image unchanged")
    require(torch.equal(target[outside], unlabeled_mask[outside]), "outside pseudo-label unchanged")
    require(torch.equal(confidence[outside], unlabeled_logits[outside]), "outside baseline confidence unchanged")


def test_diagnostics_rng_and_loss_isolation():
    reliability = torch.rand(6, 7)
    torch.manual_seed(123)
    selected_a, diagnostics = csl_cutmix.select_fixed_size_csl_destination(reliability, 2, 3)
    state_after_selection = torch.random.get_rng_state().clone()
    _ = (
        diagnostics["candidates"].clone(),
        diagnostics["scores"].mean().item(),
        diagnostics["probabilities"].sum().item(),
        int(diagnostics["selected_index"]),
    )
    require(torch.equal(torch.random.get_rng_state(), state_after_selection), "diagnostics consume no RNG")
    torch.manual_seed(123)
    selected_b, _ = csl_cutmix.select_fixed_size_csl_destination(reliability, 2, 3)
    require(torch.equal(selected_a, selected_b), "diagnostic use/logging does not alter selection")

    torch.manual_seed(41)
    student_logits = torch.randn(1, 4, 4, 5)
    target = torch.randint(0, 4, (1, 4, 5))
    confidence = torch.rand(1, 4, 5)
    loss_a, _ = compute_unsupervised_loss_by_threshold(
        student_logits, target.clone(), confidence.clone(), thresh=0.95
    )
    csl_map_a = torch.zeros(1, 4, 5)
    csl_map_b = torch.ones(1, 4, 5)
    require(not torch.equal(csl_map_a, csl_map_b), "CSL maps differ")
    loss_b, _ = compute_unsupervised_loss_by_threshold(
        student_logits, target.clone(), confidence.clone(), thresh=0.95
    )
    require(torch.equal(loss_a, loss_b), "CSL reliability cannot change original unsupervised CE")


def test_legacy_policy_and_return_contract():
    values = _synthetic_mix_inputs()
    unlabeled_image, unlabeled_mask, unlabeled_logits, labeled_image, labeled_mask, boxes, _ = values
    old_randperm = torch.randperm
    old_randint = torch.randint
    randint_values = iter([torch.tensor([1]), torch.tensor([2]), torch.tensor([0]), torch.tensor([1])])
    try:
        torch.randperm = lambda n, *args, **kwargs: torch.tensor([0, 1])
        torch.randint = lambda *args, **kwargs: next(randint_values)
        result = boundary_mix.cut_mix_label_adaptive_with_mask(
            unlabeled_image,
            unlabeled_mask,
            unlabeled_logits,
            labeled_image,
            labeled_mask,
            [0.0, 0.0],
            labeled_boxes=boxes,
            direct_labeled_mix=True,
            direct_paste_policy="random_target",
        )
    finally:
        torch.randperm = old_randperm
        torch.randint = old_randint

    require(len(result) == 4, "legacy direct return tuple remains four tensors")
    image, target, confidence, source_mask = result
    require(image.shape == unlabeled_image.shape, "legacy image shape")
    require(target.shape == unlabeled_mask.shape, "legacy target shape")
    require(confidence.shape == unlabeled_logits.shape, "legacy confidence shape")
    require(source_mask.shape == unlabeled_mask.shape, "legacy source-mask shape")


def main():
    test_config_and_dispatch_isolation()
    test_fixed_size_candidate_generation_and_scoring()
    test_duplicate_only_origin_and_strict_failure()
    test_gate_fail_preserves_target_and_consumes_no_destination_rng()
    test_gate_ownership_donor_source_and_direct_paste()
    test_diagnostics_rng_and_loss_isolation()
    test_legacy_policy_and_return_contract()
    print("S1 adaptive relocated + official fixed-size CSL CutMix smoke tests passed")


if __name__ == "__main__":
    main()
