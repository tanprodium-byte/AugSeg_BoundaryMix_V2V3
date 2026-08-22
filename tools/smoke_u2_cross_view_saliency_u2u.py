#!/usr/bin/env python3
"""Focused, side-effect-free smoke tests for U2 cross-view saliency U2U."""

import argparse
import copy
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import util.u1_saliency_u2u as common  # noqa: E402


BASELINE = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/baseline_augseg_fair80_rerun01/config.yaml"
U1_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/u1_self_pseudo_saliency_u2u_cutmix/config.yaml"
U2_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/u2_cross_view_saliency_u2u_cutmix/config.yaml"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.register_buffer("marker", torch.tensor([7.0]))
        self.seen = None

    def forward(self, tensor):
        self.seen = tensor.detach().clone()
        return self.conv(tensor), None


def freeze_teacher(teacher):
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = torch.full_like(parameter, 3.0)
    return teacher


def model_snapshot(model):
    return {
        "parameters": [p.detach().clone() for p in model.parameters()],
        "buffers": [b.detach().clone() for b in model.buffers()],
        "modes": [m.training for m in model.modules()],
        "requires_grad": [p.requires_grad for p in model.parameters()],
        "grads": [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
    }


def require_model_equal(left, right):
    for field in ("parameters", "buffers"):
        require(all(torch.equal(a, b) for a, b in zip(left[field], right[field])), field)
    require(left["modes"] == right["modes"], "recursive modes")
    require(left["requires_grad"] == right["requires_grad"], "requires_grad")
    for a, b in zip(left["grads"], right["grads"]):
        require((a is None) == (b is None), "grad presence")
        if a is not None:
            require(torch.equal(a, b), "grad value")


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def require_rng_equal(left, right):
    require(left["python"] == right["python"], "Python RNG")
    require(left["numpy"][0] == right["numpy"][0], "NumPy RNG type")
    require(np.array_equal(left["numpy"][1], right["numpy"][1]), "NumPy RNG state")
    require(left["numpy"][2:] == right["numpy"][2:], "NumPy RNG metadata")
    require(torch.equal(left["cpu"], right["cpu"]), "Torch CPU RNG")
    if left["cuda"] is not None:
        require(all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])), "Torch CUDA RNG")


def canonical_inputs(batch=4, height=9, width=11, device="cpu"):
    geometry_only = torch.arange(batch * 3 * height * width, dtype=torch.float32, device=device).reshape(batch, 3, height, width)
    intensity_view = geometry_only.mul(-0.25).add(1000).clone()
    teacher = freeze_teacher(TinyTeacher().to(device))
    with torch.no_grad():
        probabilities = teacher(geometry_only)[0].softmax(1)
        confidence, pseudo = probabilities.max(1)
    return teacher, geometry_only, intensity_view, pseudo.detach(), confidence.detach()


def expected_permutation(batch, seed, rank, iteration, device):
    generator = common.make_torch_generator(seed, rank, iteration, common.STREAM_DERANGEMENT, device)
    return common.sample_derangement(batch, generator, torch.device(device))[0]


def test_probe_identity_and_neutrality():
    teacher, geometry_only, intensity_view, pseudo, confidence = canonical_inputs()
    student = nn.Conv2d(3, 4, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    student_before = model_snapshot(student)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    original_geometry = geometry_only.clone()
    original_intensity = intensity_view.clone()
    original_pseudo = pseudo.clone()
    original_confidence = confidence.clone()
    teacher_before = model_snapshot(teacher)
    random.seed(41); np.random.seed(41); torch.manual_seed(41)
    rng_before = rng_snapshot()
    permutation = expected_permutation(4, 2, 0, 17, "cpu")
    output = common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=geometry_only, strong_rgb=intensity_view,
        hard_pseudo=pseudo, confidence=confidence, base_seed=2, rank=0,
        epoch=1, step=3, absolute_global_iteration=17,
        saliency_probe_rgb=intensity_view,
    )
    require(torch.equal(teacher.seen, original_intensity[permutation]), "U2 probe is not exact pre-mix intensity donor")
    require(torch.equal(geometry_only, original_geometry), "geometry-only donor mutated")
    require(torch.equal(intensity_view, original_intensity), "pre-mix intensity tensor mutated")
    require(torch.equal(pseudo, original_pseudo), "canonical pseudo mutated")
    require(torch.equal(confidence, original_confidence), "canonical confidence mutated")
    require_model_equal(teacher_before, model_snapshot(teacher))
    require_model_equal(student_before, model_snapshot(student))
    require(optimizer_before == optimizer.state_dict(), "optimizer state")
    require_rng_equal(rng_before, rng_snapshot())
    require(not torch.any(permutation == torch.arange(4)), "fixed point")
    require(torch.equal(torch.sort(permutation).values, torch.arange(4)), "donor not used exactly once")
    require(output[3].total_receivers == 4 and output[3].paste_attempts == 4, "receiver accounting")


def test_probe_math_and_all_pixel_ce():
    gradient = torch.tensor([[[[3.0]], [[4.0]], [[0.0]]]])
    require(common.rgb_l2_saliency(gradient).item() == 5.0, "RGB L2")
    raw = torch.tensor([[[2.0, 4.0], [6.0, 8.0]], [[9.0, 9.0], [9.0, 9.0]]])
    normalized, near_flat = common.normalize_saliency(raw)
    require(torch.equal(normalized[0], torch.tensor([[0.0, 1/3], [2/3, 1.0]])), "min-max")
    require(torch.equal(normalized[1], torch.zeros_like(normalized[1])), "constant normalization")
    probabilities = common.candidate_probabilities(torch.randn(2, 8), torch.tensor([False, True]))
    require(torch.equal(probabilities[1], torch.full((8,), 1/8)), "uniform near-flat")
    teacher, _, intensity_view, pseudo, _ = canonical_inputs(batch=2, height=5, width=7)
    normalized, _, loss, finite = common.compute_self_pseudo_saliency(teacher, intensity_view, pseudo)
    expected = torch.nn.functional.cross_entropy(teacher(intensity_view)[0], pseudo, reduction="mean")
    require(torch.equal(loss, expected), "probe CE is not mean over all pixels")
    require(normalized.shape == pseudo.shape and bool(finite), "saliency result")


def test_golden_common_pipeline():
    teacher, geometry_only, intensity_view, pseudo, confidence = canonical_inputs()
    supplied = torch.linspace(0, 1, pseudo.numel()).reshape_as(confidence)
    near_flat = torch.zeros(pseudo.size(0), dtype=torch.bool)
    original = common.compute_self_pseudo_saliency
    observed = []

    def fixed_saliency(_teacher, probe, target):
        observed.append(probe.detach().clone())
        require(torch.equal(target, pseudo[expected_permutation(4, 9, 0, 23, "cpu")]), "canonical donor target")
        return supplied, near_flat, torch.tensor(1.0), torch.tensor(True)

    common.compute_self_pseudo_saliency = fixed_saliency
    try:
        kwargs = dict(
            teacher=teacher, weak_rgb=geometry_only, strong_rgb=intensity_view,
            hard_pseudo=pseudo, confidence=confidence, base_seed=9, rank=0,
            epoch=2, step=1, absolute_global_iteration=23,
        )
        u1_result = common.apply_u1_saliency_u2u(**kwargs)
        u2_result = common.apply_u1_saliency_u2u(**kwargs, saliency_probe_rgb=intensity_view)
    finally:
        common.compute_self_pseudo_saliency = original
    permutation = expected_permutation(4, 9, 0, 23, "cpu")
    require(torch.equal(observed[0], geometry_only[permutation]), "U1 probe donor")
    require(torch.equal(observed[1], intensity_view[permutation]), "U2 probe donor")
    for left, right in zip(u1_result[:3], u2_result[:3]):
        require(torch.equal(left, right), "golden output mismatch")
    require(u1_result[3] == u2_result[3], "golden diagnostics mismatch")


def test_dispatch_and_config():
    from train_semi import select_unlabeled_mix_branch
    require(select_unlabeled_mix_branch(True, True, 1.0) == ("u1", 1, 1), "U1 dispatch")
    require(select_unlabeled_mix_branch(False, True, 1.0, u2_enabled=True) == ("u2", 1, 1), "U2 dispatch")
    try:
        select_unlabeled_mix_branch(True, True, 1.0, u2_enabled=True)
    except ValueError:
        pass
    else:
        raise AssertionError("U1/U2 mutual exclusion")
    baseline = yaml.safe_load(BASELINE.read_text())
    u1_cfg = yaml.safe_load(U1_CONFIG.read_text())
    u2_cfg = yaml.safe_load(U2_CONFIG.read_text())
    for section in ("dataset", "trainer", "criterion", "net", "checkpoint"):
        require(u2_cfg[section] == baseline[section], f"baseline drift: {section}")
        require(u2_cfg[section] == u1_cfg[section], f"U1 inheritance drift: {section}")
    require(u2_cfg["u2_cross_view_saliency_u2u"] == {
        "enabled": True, "num_candidates": 8, "temperature": 0.2,
        "rng_policy": "u1_rng_policy_v1", "debug_log": True,
    }, "U2 method config")
    for section in ("saliency_cutmix", "fixed_size_csl_destination", "csl", "csl_cutmix", "boundary_mix", "boundary_component", "boundary_compatibility"):
        require(u2_cfg[section]["enabled"] is False, f"incompatible enabled: {section}")
    require(u2_cfg["wandb"]["enable"] == baseline["wandb"]["enable"], "W&B enablement")
    require(u2_cfg["hf"]["enabled"] == baseline["hf"]["enabled"], "HF enablement")
    require(u2_cfg["hf"]["path_in_repo"] not in (baseline["hf"]["path_in_repo"], u1_cfg["hf"]["path_in_repo"]), "HF identity collision")


def test_static_contract():
    source = (ROOT / "util/u1_saliency_u2u.py").read_text()
    train_source = (ROOT / "train_semi.py").read_text()
    require("(input_grad,) = torch.autograd.grad(" in source, "autograd tuple unpack")
    require("create_graph=False" in source and "retain_graph=False" in source, "higher-order graph")
    require("probe_loss.backward" not in source, "probe backward")
    require("donor_probe = probe_rgb[permutation].detach()" in source, "same donor mapping")
    require("mixed_rgb = receiver_strong.clone()" in source, "receiver clone")
    require("mixed_rgb[receiver" in source and "donor_weak[" in source, "weak RGB paste")
    require("hash(" not in source, "Python hash")
    require(source.count("STREAM_") >= 8, "stream constants/usages")
    require("saliency_probe_rgb=image_u_aug if mix_branch == \"u2\" else None" in train_source, "U2 integration")
    require('setdefault("enabled", False)' in train_source, "default false")
    require(source.count("dist.all_reduce") == 2, "collective count")
    require("state_dict" not in source and "get_rng_state" not in source, "production snapshots")


def test_cuda_crop321_u2():
    if not torch.cuda.is_available():
        print("UNVERIFIED test_cuda_crop321_u2: CUDA unavailable")
        return
    device = torch.device("cuda", 0)
    teacher, geometry_only, intensity_view, pseudo, confidence = canonical_inputs(
        batch=2, height=321, width=321, device=device
    )
    before = model_snapshot(teacher)
    rng_before = rng_snapshot()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=geometry_only, strong_rgb=intensity_view,
        hard_pseudo=pseudo, confidence=confidence, base_seed=2, rank=0,
        epoch=0, step=0, absolute_global_iteration=0,
        saliency_probe_rgb=intensity_view,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device)
    require_model_equal(before, model_snapshot(teacher))
    require_rng_equal(rng_before, rng_snapshot())
    require(output[3].paste_attempts == 2, "CUDA paste attempts")
    print(f"MEASURE u2_crop321_helper_runtime_sec={elapsed:.6f} peak_allocated_bytes={peak} batch=2 shape=321x321 device=cuda:0")


def test_ddp_gloo():
    from torch.nn.parallel import DistributedDataParallel
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    geometry_only = torch.arange(2 * 3 * 9 * 11, dtype=torch.float32).reshape(2, 3, 9, 11)
    intensity_view = geometry_only.mul(-0.25).add(1000).clone()
    teacher = DistributedDataParallel(TinyTeacher().eval()).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = torch.full_like(parameter, 3.0)
    with torch.no_grad():
        probabilities = teacher(geometry_only)[0].softmax(1)
        confidence, pseudo = probabilities.max(1)
    before = model_snapshot(teacher)
    common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=geometry_only, strong_rgb=intensity_view,
        hard_pseudo=pseudo, confidence=confidence, base_seed=2,
        rank=dist.get_rank(), epoch=0, step=0, absolute_global_iteration=0,
        saliency_probe_rgb=intensity_view,
    )
    require_model_equal(before, model_snapshot(teacher))
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    if args.ddp:
        test_ddp_gloo()
        print("PASS test_ddp_gloo")
        return
    start = time.perf_counter()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tests = [
        test_probe_identity_and_neutrality,
        test_probe_math_and_all_pixel_ce,
        test_golden_common_pipeline,
        test_dispatch_and_config,
        test_static_contract,
        test_cuda_crop321_u2,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    elapsed = time.perf_counter() - start
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"MEASURE cpu_smoke_runtime_sec={elapsed:.6f} peak_rss_kib={peak_after} peak_delta_kib={max(0, peak_after-peak_before)} batch=4 shape=9x11 device=cpu")
    print("PASS smoke_u2_cross_view_saliency_u2u")


if __name__ == "__main__":
    main()
