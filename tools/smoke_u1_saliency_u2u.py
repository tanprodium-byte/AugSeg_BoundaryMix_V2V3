#!/usr/bin/env python3
"""Focused, side-effect-free U1 correctness smoke tests."""

from __future__ import annotations

import hashlib
import argparse
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util import u1_saliency_u2u as u1
from util.boundary_mix import _rand_bbox as legacy_s1_rand_bbox


BASELINE_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/baseline_augseg_fair80_rerun01/config.yaml"
U1_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/u1_self_pseudo_saliency_u2u_cutmix/config.yaml"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def snapshot_global_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def assert_rng_equal(before, after):
    require(before["python"] == after["python"], "Python global RNG changed")
    for left, right in zip(before["numpy"], after["numpy"]):
        if isinstance(left, np.ndarray):
            require(np.array_equal(left, right), "NumPy global RNG changed")
        else:
            require(left == right, "NumPy global RNG changed")
    require(torch.equal(before["torch_cpu"], after["torch_cpu"]), "Torch CPU global RNG changed")
    require(len(before["torch_cuda"]) == len(after["torch_cuda"]), "CUDA RNG device count changed")
    for left, right in zip(before["torch_cuda"], after["torch_cuda"]):
        require(torch.equal(left, right), "Torch CUDA global RNG changed")


def test_seed_policy():
    fields = (2, 0, 17, u1.STREAM_CANDIDATES)
    serialized = struct.pack("<QQQQ", *fields)
    expected = hashlib.sha256(serialized).hexdigest()
    require(u1._stream_digest(*fields).hex() == expected, "SHA-256 test vector mismatch")
    words = np.frombuffer(bytes.fromhex(expected), dtype="<u4").copy()
    actual = u1.make_numpy_rng(*fields).get_state()[1][:8]
    require(np.array_equal(actual, np.random.RandomState(words).get_state()[1][:8]), "all digest words not used")
    require(u1.U1_RNG_POLICY_VERSION == "u1_rng_policy_v1", "wrong policy version")
    require(
        (u1.STREAM_DERANGEMENT, u1.STREAM_CANDIDATES, u1.STREAM_SELECTION, u1.STREAM_DESTINATION)
        == (1, 2, 3, 4),
        "wrong stream IDs",
    )
    replay_a = u1.make_torch_generator(2, 0, 17, 1, "cpu")
    replay_b = u1.make_torch_generator(2, 0, 17, 1, "cpu")
    require(torch.equal(torch.randperm(16, generator=replay_a), torch.randperm(16, generator=replay_b)), "replay failed")
    variants = [u1._stream_digest(2, 0, 17, 1), u1._stream_digest(3, 0, 17, 1), u1._stream_digest(2, 1, 17, 1), u1._stream_digest(2, 0, 18, 1)]
    require(len(set(variants)) == 4, "seed/rank/iteration variation failed")


def test_derangement():
    for batch in (2, 3, 8, 17):
        generator = u1.make_torch_generator(2, 0, batch, u1.STREAM_DERANGEMENT, "cpu")
        permutation, attempts = u1.sample_derangement(batch, generator, torch.device("cpu"))
        require(sorted(permutation.tolist()) == list(range(batch)), "not a bijection")
        require(not torch.any(permutation == torch.arange(batch)), "fixed point")
        require(1 <= attempts <= 128, "attempt bound")
    try:
        u1.sample_derangement(1, torch.Generator().manual_seed(0), torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("B<2 did not fail")
    original = torch.randperm
    try:
        torch.randperm = lambda n, **kwargs: torch.arange(n, device=kwargs.get("device"))
        try:
            u1.sample_derangement(4, torch.Generator(), torch.device("cpu"))
        except RuntimeError as exc:
            require("128" in str(exc), "wrong exhaustion error")
        else:
            raise AssertionError("forced exhaustion did not fail")
    finally:
        torch.randperm = original


class BoundaryRng:
    def __init__(self, lam, center=160):
        self.lam = lam
        self.center = center
        self.beta_calls = 0
        self.randint_calls = 0

    def beta(self, alpha, beta):
        require((alpha, beta) == (8, 2), "wrong Beta distribution")
        self.beta_calls += 1
        return self.lam

    def randint(self, low, high, size):
        require(low == int(321 / 8) and high == 321, "wrong center range")
        self.randint_calls += 1
        return np.full(size, self.center, dtype=np.int64)


def test_candidates_and_zero_area():
    local_rng = u1.make_numpy_rng(2, 0, 11, u1.STREAM_CANDIDATES)
    local_boxes, local_valid = u1.sample_legacy_s1_candidates(3, 321, 321, local_rng)
    global_state = np.random.get_state()
    try:
        reference_rng = u1.make_numpy_rng(2, 0, 11, u1.STREAM_CANDIDATES)
        np.random.set_state(reference_rng.get_state())
        legacy_candidates = []
        for _ in range(8):
            coords = legacy_s1_rand_bbox((3, 3, 321, 321), lam=np.random.beta(8, 2))
            legacy_candidates.append(np.stack(coords, axis=1))
        legacy_boxes = torch.from_numpy(np.stack(legacy_candidates, axis=1))
    finally:
        np.random.set_state(global_state)
    require(torch.equal(local_boxes, legacy_boxes), "U1 geometry differs from exact legacy S1 helper")
    expected_valid = (legacy_boxes[:, :, 2] > legacy_boxes[:, :, 0]) & (legacy_boxes[:, :, 3] > legacy_boxes[:, :, 1])
    require(torch.equal(local_valid, expected_valid), "validity differs from legacy bounds")

    for cut in (0, 1, 2):
        lam = 1.0 - (cut / 321.0) ** 2 if cut else 1.0
        rng = BoundaryRng(lam)
        boxes, valid = u1.sample_legacy_s1_candidates(2, 321, 321, rng)
        require(boxes.shape == (2, 8, 4), "candidate shape/count")
        require(rng.beta_calls == 8 and rng.randint_calls == 16, "not exactly eight complete draws")
        expected_valid = cut >= 2
        require(bool(valid.all()) == expected_valid, f"cut={cut} validity")
    saliency = torch.arange(2 * 321 * 321, dtype=torch.float32).reshape(2, 321, 321)
    zero_boxes, _ = u1.sample_legacy_s1_candidates(2, 321, 321, BoundaryRng(1.0))
    scores, valid = u1.score_candidates(saliency, zero_boxes)
    require(not valid.any() and torch.equal(scores, torch.zeros_like(scores)), "zero-area score must be zero")
    probabilities = u1.candidate_probabilities(scores, torch.zeros(2, dtype=torch.bool))
    require(torch.equal(probabilities, torch.full_like(probabilities, 1 / 8)), "invalid candidates removed")

    rgb = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    pseudo = torch.arange(2 * 4 * 4, dtype=torch.long).reshape(2, 4, 4)
    confidence = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    boxes = torch.zeros((2, 8, 4), dtype=torch.long)
    selected = torch.zeros(2, dtype=torch.long)
    generator = torch.Generator().manual_seed(91)
    state_before = generator.get_state().clone()
    mixed = u1.relocate_selected(rgb, rgb.flip(0), pseudo, pseudo.flip(0), confidence, confidence.flip(0), boxes, selected, generator)
    require(not torch.equal(state_before, generator.get_state()), "destination RNG not consumed")
    require(torch.equal(mixed[0], rgb) and torch.equal(mixed[1], pseudo) and torch.equal(mixed[2], confidence), "empty paste not a no-op")
    require(not mixed[3].any(), "empty paste counted nonempty")

    receiver_rgb = torch.zeros(2, 3, 5, 6)
    donor_rgb = torch.arange(2 * 3 * 5 * 6, dtype=torch.float32).reshape(2, 3, 5, 6)
    receiver_pseudo = torch.zeros(2, 5, 6, dtype=torch.long)
    donor_pseudo = torch.arange(2 * 5 * 6, dtype=torch.long).reshape(2, 5, 6)
    receiver_confidence = torch.zeros(2, 5, 6)
    donor_confidence = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6) + 0.5
    boxes = torch.zeros((2, 8, 4), dtype=torch.long)
    boxes[0, 0] = torch.tensor([1, 2, 4, 5])
    boxes[1, 0] = torch.tensor([0, 1, 2, 5])
    selected = torch.zeros(2, dtype=torch.long)
    expected_generator = torch.Generator().manual_seed(123)
    expected_destinations = []
    for crop_h, crop_w in ((3, 3), (2, 4)):
        row = int(torch.randint(0, 5 - crop_h + 1, (1,), generator=expected_generator).item())
        col = int(torch.randint(0, 6 - crop_w + 1, (1,), generator=expected_generator).item())
        expected_destinations.append((row, col, row + crop_h, col + crop_w))
    aligned = u1.relocate_selected(
        receiver_rgb, donor_rgb, receiver_pseudo, donor_pseudo,
        receiver_confidence, donor_confidence, boxes, selected,
        torch.Generator().manual_seed(123),
    )
    sources = ((1, 2, 4, 5), (0, 1, 2, 5))
    for index, (source, destination) in enumerate(zip(sources, expected_destinations)):
        sr1, sc1, sr2, sc2 = source
        dr1, dc1, dr2, dc2 = destination
        require(torch.equal(aligned[0][index, :, dr1:dr2, dc1:dc2], donor_rgb[index, :, sr1:sr2, sc1:sc2]), "RGB alignment")
        require(torch.equal(aligned[1][index, dr1:dr2, dc1:dc2], donor_pseudo[index, sr1:sr2, sc1:sc2]), "pseudo alignment")
        require(torch.equal(aligned[2][index, dr1:dr2, dc1:dc2], donor_confidence[index, sr1:sr2, sc1:sc2]), "confidence alignment")
        outside = torch.ones(5, 6, dtype=torch.bool)
        outside[dr1:dr2, dc1:dc2] = False
        require(not aligned[0][index, :, outside].any(), "RGB outside destination changed")
        require(not aligned[1][index, outside].any(), "pseudo outside destination changed")
        require(not aligned[2][index, outside].any(), "confidence outside destination changed")


def test_saliency_math():
    gradient = torch.tensor([[[[3.0]], [[4.0]], [[0.0]]]])
    saliency = u1.rgb_l2_saliency(gradient)
    require(saliency.item() == 5.0, "RGB square-root L2 is not exact")
    raw = torch.tensor([[[2.0, 4.0], [6.0, 8.0]], [[7.0, 7.0], [7.0, 7.0]]])
    normalized, near_flat = u1.normalize_saliency(raw)
    require(torch.equal(normalized[0], torch.tensor([[0.0, 1 / 3], [2 / 3, 1.0]])), "min-max mismatch")
    require(torch.equal(normalized[1], torch.zeros_like(normalized[1])), "constant saliency not exact zero")
    require(bool(near_flat[1]), "constant saliency not near-flat")
    known_scores = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    probs = u1.candidate_probabilities(known_scores, torch.tensor([True, False]))
    require(torch.equal(probs[0], torch.full((8,), 1 / 8)), "near-flat not exact uniform")
    require(torch.allclose(probs[1], torch.softmax(known_scores[1] / 0.2, 0)), "temperature softmax mismatch")


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1, bias=False)
        self.register_buffer("marker", torch.tensor([4.0]))

    def forward(self, x):
        return self.conv(x), None


class NonfiniteTeacher(TinyTeacher):
    def forward(self, x):
        logits = self.conv(x) * torch.tensor(float("nan"), device=x.device)
        return logits, None


def teacher_snapshot(model):
    return {
        "parameters": [p.detach().clone() for p in model.parameters()],
        "buffers": [b.detach().clone() for b in model.buffers()],
        "flags": [m.training for m in model.modules()],
        "requires_grad": [p.requires_grad for p in model.parameters()],
        "grads": [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
    }


def assert_teacher_equal(before, after):
    for key in ("parameters", "buffers"):
        require(all(torch.equal(a, b) for a, b in zip(before[key], after[key])), f"Teacher {key} changed")
    require(before["flags"] == after["flags"], "Teacher mode changed")
    require(before["requires_grad"] == after["requires_grad"], "requires_grad changed")
    for left, right in zip(before["grads"], after["grads"]):
        require((left is None) == (right is None), "Teacher grad presence changed")
        if left is not None:
            require(torch.equal(left, right), "Teacher grad value changed")


def test_probe_alignment_and_rng_neutrality():
    torch.manual_seed(123)
    teacher = TinyTeacher().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = torch.full_like(parameter, 9.0)
    weak = torch.randn(4, 3, 8, 8)
    strong = torch.randn(4, 3, 8, 8)
    weak_before = weak.clone()
    strong_before = strong.clone()
    with torch.no_grad():
        pseudo = teacher(weak)[0].argmax(1)
        confidence = teacher(weak)[0].softmax(1).max(1).values
    teacher_before = teacher_snapshot(teacher)
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(77)
    rng_before = snapshot_global_rng()
    mixed_rgb, mixed_pseudo, mixed_confidence, diagnostics = u1.apply_u1_saliency_u2u(
        teacher=teacher,
        weak_rgb=weak,
        strong_rgb=strong,
        hard_pseudo=pseudo,
        confidence=confidence,
        base_seed=2,
        rank=0,
        epoch=0,
        step=0,
        absolute_global_iteration=0,
    )
    assert_rng_equal(rng_before, snapshot_global_rng())
    assert_teacher_equal(teacher_before, teacher_snapshot(teacher))
    require(diagnostics.total_receivers == 4 and diagnostics.total_candidate_draws == 32, "diagnostics")
    require(diagnostics.paste_attempts == 4, "paste attempt rate")
    require(mixed_rgb.shape == strong.shape and mixed_pseudo.shape == pseudo.shape and mixed_confidence.shape == confidence.shape, "aligned output shapes")
    require(torch.equal(weak, weak_before) and torch.equal(strong, strong_before), "canonical input mutation")


def test_disabled_train_dispatch_boundary():
    # Import the real training boundary only after global RNG state is seeded.
    from train_semi import select_unlabeled_mix_branch

    class TinyState(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([3.0]))
            self.register_buffer("marker", torch.tensor([5.0]))

    teacher = TinyState().eval()
    student = TinyState().train()
    teacher_before = teacher_snapshot(teacher)
    student_before = teacher_snapshot(student)
    rgb = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    pseudo = torch.arange(2 * 4 * 4, dtype=torch.long).reshape(2, 4, 4)
    confidence = torch.linspace(0, 1, 2 * 4 * 4).reshape(2, 4, 4)
    inputs_before = (rgb.clone(), pseudo.clone(), confidence.clone())

    random.seed(811)
    np.random.seed(811)
    torch.manual_seed(811)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(811)
    initial_rng = snapshot_global_rng()

    # Independent reference for the exact pre-U1 legacy trigger operation.
    reference_draw = np.random.uniform(0, 1)
    expected = ("legacy", int(reference_draw < 0.75), int(reference_draw < 0.75))
    reference_rng = snapshot_global_rng()

    random.setstate(initial_rng["python"])
    np.random.set_state(initial_rng["numpy"])
    torch.random.set_rng_state(initial_rng["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(initial_rng["torch_cuda"])
    actual = select_unlabeled_mix_branch(False, True, 0.75)
    require(actual == expected, "disabled dispatch did not select the exact legacy branch/flags")
    assert_rng_equal(reference_rng, snapshot_global_rng())

    # The boundary itself passes the legacy mix/loss inputs through untouched.
    require(torch.equal(rgb, inputs_before[0]), "disabled dispatch changed RGB")
    require(torch.equal(pseudo, inputs_before[1]), "disabled dispatch changed pseudo-label")
    require(torch.equal(confidence, inputs_before[2]), "disabled dispatch changed confidence/loss input")
    assert_teacher_equal(teacher_before, teacher_snapshot(teacher))
    assert_teacher_equal(student_before, teacher_snapshot(student))


def test_visible_failures():
    weak = torch.randn(2, 3, 4, 4)
    pseudo = torch.zeros(2, 4, 4, dtype=torch.long)
    confidence = torch.ones(2, 4, 4)
    teacher = TinyTeacher().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    original = torch.randperm
    try:
        torch.randperm = lambda n, **kwargs: torch.arange(n, device=kwargs.get("device"))
        try:
            u1.apply_u1_saliency_u2u(
                teacher=teacher, weak_rgb=weak, strong_rgb=weak, hard_pseudo=pseudo,
                confidence=confidence, base_seed=2, rank=0, epoch=0, step=0,
                absolute_global_iteration=0,
            )
        except u1.U1SynchronizedError as exc:
            require("derangement_failure" in str(exc), "derangement failure not visible")
        else:
            raise AssertionError("derangement exhaustion did not synchronize/fail")
    finally:
        torch.randperm = original

    nonfinite_teacher = NonfiniteTeacher().eval()
    for parameter in nonfinite_teacher.parameters():
        parameter.requires_grad_(False)
    try:
        u1.apply_u1_saliency_u2u(
            teacher=nonfinite_teacher, weak_rgb=weak, strong_rgb=weak, hard_pseudo=pseudo,
            confidence=confidence, base_seed=2, rank=0, epoch=0, step=0,
            absolute_global_iteration=0,
        )
    except u1.U1SynchronizedError as exc:
        require("nonfinite_probe_or_saliency" in str(exc), "nonfinite failure not visible")
    else:
        raise AssertionError("nonfinite probe did not synchronize/fail")


def test_cuda_one_step_crop321():
    if not torch.cuda.is_available():
        print("UNVERIFIED test_cuda_one_step_crop321: CUDA unavailable")
        return
    device = torch.device("cuda", 0)
    teacher = TinyTeacher().to(device).eval()
    student = TinyTeacher().to(device).train()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    weak = torch.linspace(-1, 1, 2 * 3 * 321 * 321, device=device).reshape(2, 3, 321, 321)
    strong = weak.flip(-1).clone()
    with torch.no_grad():
        canonical_logits = teacher(weak)[0]
        canonical_probs = canonical_logits.softmax(1)
        confidence, pseudo = canonical_probs.max(1)
    teacher_before = teacher_snapshot(teacher)
    rng_before = snapshot_global_rng()
    student_before = [parameter.detach().clone() for parameter in student.parameters()]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    mixed_rgb, mixed_pseudo, mixed_confidence, diagnostics = u1.apply_u1_saliency_u2u(
        teacher=teacher,
        weak_rgb=weak,
        strong_rgb=strong,
        hard_pseudo=pseudo,
        confidence=confidence,
        base_seed=2,
        rank=0,
        epoch=0,
        step=0,
        absolute_global_iteration=0,
    )
    prediction = student(mixed_rgb)[0]
    valid = mixed_confidence.ge(0.0)
    loss_map = F.cross_entropy(prediction, mixed_pseudo, reduction="none")
    loss = (loss_map * valid).sum() / valid.sum()
    require(bool(torch.isfinite(loss)), "one-step Student loss nonfinite")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters()):
            teacher_parameter.data.mul_(0.999).add_(student_parameter.data, alpha=0.001)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device)
    require(any(not torch.equal(a, b) for a, b in zip(student_before, student.parameters())), "Student did not update")
    require(diagnostics.paste_attempts == 2, "not every receiver attempted paste")
    # Probe neutrality was established before the intentionally subsequent EMA update.
    require(teacher_before["flags"] == teacher_snapshot(teacher)["flags"], "Teacher mode changed")
    assert_rng_equal(rng_before, snapshot_global_rng())
    print(f"MEASURE crop321_helper_one_step_runtime_sec={elapsed:.6f} peak_allocated_bytes={peak}")


def test_ddp_failure_and_neutrality():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    device = torch.device("cpu")
    teacher = DistributedDataParallel(TinyTeacher().to(device).eval()).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    weak = torch.randn(2, 3, 16, 16, device=device)
    strong = torch.randn_like(weak)
    with torch.no_grad():
        probabilities = teacher(weak)[0].softmax(1)
        confidence, pseudo = probabilities.max(1)
    before = teacher_snapshot(teacher)
    u1.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=weak, strong_rgb=strong, hard_pseudo=pseudo,
        confidence=confidence, base_seed=2, rank=dist.get_rank(), epoch=0, step=0,
        absolute_global_iteration=0,
    )
    assert_teacher_equal(before, teacher_snapshot(teacher))
    try:
        u1.apply_u1_saliency_u2u(
            teacher=teacher, weak_rgb=weak[:1], strong_rgb=strong[:1], hard_pseudo=pseudo[:1],
            confidence=confidence[:1], base_seed=2, rank=dist.get_rank(), epoch=0, step=1,
            absolute_global_iteration=1,
        )
    except u1.U1SynchronizedError:
        pass
    else:
        raise AssertionError("synchronized B<2 failure missing")
    try:
        u1.synchronized_failure_check(True, "forced_nonfinite", device=device, rank=0, epoch=0, step=2, iteration=2)
    except u1.U1SynchronizedError as exc:
        require("forced_nonfinite" in str(exc), "wrong synchronized failure category")
    else:
        raise AssertionError("forced synchronized nonfinite failure missing")
    dist.destroy_process_group()


def recursive_diff(left, right, prefix=""):
    changes = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                changes.append(path)
            else:
                changes.extend(recursive_diff(left[key], right[key], path))
    elif left != right:
        changes.append(prefix)
    return changes


def test_config_and_static_integration():
    baseline = yaml.safe_load(BASELINE_CONFIG.read_text())
    config = yaml.safe_load(U1_CONFIG.read_text())
    require(config["dataset"]["train"]["crop"]["size"] == [321, 321], "crop drift")
    require(config["trainer"]["epochs"] == 80, "schedule drift")
    require(config["u1_saliency_u2u"]["enabled"] is True, "U1 not enabled")
    for section in ("saliency_cutmix", "fixed_size_csl_destination", "csl", "csl_cutmix", "boundary_mix", "boundary_component", "boundary_compatibility"):
        require(config[section]["enabled"] is False, f"incompatible extension {section}")
    unchanged = ("dataset", "trainer", "criterion", "net", "checkpoint")
    for section in unchanged:
        require(config[section] == baseline[section], f"scientific baseline drift in {section}")
    differences = recursive_diff(baseline, config)
    allowed_prefixes = (
        "name", "saver.snapshot_dir", "wandb.name", "run.name", "hf.path_in_repo", "u1_saliency_u2u"
    )
    require(all(any(path == prefix or path.startswith(prefix + ".") for prefix in allowed_prefixes) for path in differences), f"unexpected semantic config diff: {differences}")
    source = (ROOT / "train_semi.py").read_text()
    require('setdefault("enabled", False)' in source, "U1 not disabled by default")
    require("select_unlabeled_mix_branch(" in source, "actual train dispatch boundary not integrated")
    module_source = (ROOT / "util/u1_saliency_u2u.py").read_text()
    require(module_source.count("dist.all_reduce") == 2, "unexpected collectives")
    require("get_rng_state" not in module_source and "random.getstate" not in module_source, "production RNG snapshots present")
    require("state_dict" not in module_source, "production Teacher snapshots present")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    if args.ddp:
        test_ddp_failure_and_neutrality()
        print("PASS test_ddp_failure_and_neutrality")
        return
    tests = [
        test_seed_policy,
        test_derangement,
        test_candidates_and_zero_area,
        test_saliency_math,
        test_probe_alignment_and_rng_neutrality,
        test_disabled_train_dispatch_boundary,
        test_visible_failures,
        test_cuda_one_step_crop321,
        test_config_and_static_integration,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PASS smoke_u1_saliency_u2u")


if __name__ == "__main__":
    main()
