#!/usr/bin/env python3
"""Focused, side-effect-free correctness smoke for U3 confidence-filtered saliency."""

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
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import util.u1_saliency_u2u as common  # noqa: E402


U1_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/u1_self_pseudo_saliency_u2u_cutmix/config.yaml"
U2_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/u2_cross_view_saliency_u2u_cutmix/config.yaml"
U3_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/u3_confidence_filtered_cross_view_saliency_u2u_cutmix/config.yaml"


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


def freeze_teacher(device="cpu"):
    teacher = TinyTeacher().to(device).eval()
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
    require(left["modes"] == right["modes"], "recursive modes")
    require(left["requires_grad"] == right["requires_grad"], "requires_grad")
    for field in ("parameters", "buffers"):
        require(all(torch.equal(a, b) for a, b in zip(left[field], right[field])), field)
    for a, b in zip(left["grads"], right["grads"]):
        require((a is None) == (b is None), "grad presence")
        if a is not None:
            require(torch.equal(a, b), "grad value")


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def require_rng_equal(left, right):
    require(left["python"] == right["python"], "Python RNG")
    require(left["numpy"][0] == right["numpy"][0], "NumPy RNG type")
    require(np.array_equal(left["numpy"][1], right["numpy"][1]), "NumPy RNG state")
    require(left["numpy"][2:] == right["numpy"][2:], "NumPy RNG metadata")
    require(torch.equal(left["cpu"], right["cpu"]), "Torch CPU RNG")
    if left["cuda"] is not None:
        require(all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])), "Torch CUDA RNG")


def inputs(batch=4, height=9, width=11, device="cpu"):
    weak = torch.arange(batch * 3 * height * width, dtype=torch.float32, device=device).reshape(batch, 3, height, width)
    strong = weak.mul(-0.25).add(1000).clone()
    teacher = freeze_teacher(device)
    with torch.no_grad():
        probabilities = teacher(weak)[0].softmax(1)
        confidence, pseudo = probabilities.max(1)
    return teacher, weak, strong, pseudo.detach(), confidence.detach()


def permutation(batch, seed, iteration, device="cpu"):
    generator = common.make_torch_generator(seed, 0, iteration, common.STREAM_DERANGEMENT, device)
    return common.sample_derangement(batch, generator, torch.device(device))[0]


def test_config_and_dispatch():
    from train_semi import select_unlabeled_mix_branch
    cfg2 = yaml.safe_load(U2_CONFIG.read_text())
    cfg3 = yaml.safe_load(U3_CONFIG.read_text())
    require(select_unlabeled_mix_branch(False, True, 1.0, u3_enabled=True) == ("u3", 1, 1), "U3 dispatch")
    disabled = select_unlabeled_mix_branch(False, False, 0.0)
    require(disabled == ("legacy", 0, 0), "disabled legacy dispatch")
    for flags in ((True, True, False), (True, False, True), (False, True, True)):
        try:
            select_unlabeled_mix_branch(flags[0], True, 1.0, u2_enabled=flags[1], u3_enabled=flags[2])
        except ValueError:
            pass
        else:
            raise AssertionError("U1/U2/U3 mutual exclusion")
    for section in ("dataset", "trainer", "criterion", "net", "checkpoint"):
        require(cfg3[section] == cfg2[section], f"U2 semantic drift: {section}")
    require(cfg3["dataset"]["train"]["crop"]["size"] == [321, 321], "crop321")
    require(cfg3["dataset"]["ignore_label"] == 255, "runtime ignore label")
    require(cfg3["trainer"]["unsupervised"]["threshold"] == 0.95, "runtime threshold")
    require(cfg3["u3_confidence_filtered_cross_view_saliency_u2u"]["enabled"] is True, "U3 opt-in")
    require(cfg3["wandb"]["enable"] == cfg2["wandb"]["enable"], "W&B enablement")
    require(cfg3["hf"]["enabled"] == cfg2["hf"]["enabled"], "HF enablement")
    require(cfg3["wandb"]["name"] != cfg2["wandb"]["name"], "W&B identity")
    require(cfg3["hf"]["path_in_repo"] != cfg2["hf"]["path_in_repo"], "HF identity")


def test_masked_ce_math():
    torch.manual_seed(17)
    batch, classes, height, width = 3, 4, 2, 3
    probe = torch.randn(batch, 3, height, width, dtype=torch.float64, requires_grad=True)
    layer = nn.Conv2d(3, classes, 1, dtype=torch.float64)
    logits = layer(probe)
    target = torch.tensor([[[255, 1, 2], [3, 0, 1]], [[1, 1, 2], [2, 3, 0]], [[3, 2, 1], [0, 1, 2]]])
    confidence = torch.tensor([[[0.1, .95, .94], [.99, .2, .3]], [[1., .95, .2], [.1, .99, .1]], [[1., 1., 1.], [1., 1., 1.]]], dtype=torch.float64)
    ce = F.cross_entropy(logits, target, reduction="none", ignore_index=255)
    mask = confidence.ge(.95) & target.ne(255)
    counts = mask.flatten(1).sum(1)
    per_image = (ce * mask.to(ce.dtype)).flatten(1).sum(1) / counts.clamp_min(1).to(ce.dtype)
    loss = per_image.sum() / batch
    grad_logits, grad_input = torch.autograd.grad(loss, (logits, probe), retain_graph=True)
    require(ce.shape == target.shape, "unreduced CE shape")
    require(torch.equal(grad_logits.permute(0, 2, 3, 1)[~mask], torch.zeros_like(grad_logits.permute(0, 2, 3, 1)[~mask])), "masked logits gradient")
    require(counts.tolist() == [2, 3, 6], "mixed valid counts")
    require(torch.isfinite(loss) and torch.isfinite(grad_input).all(), "masked CE finite")
    all_valid = torch.ones_like(mask)
    all_loss = ((F.cross_entropy(logits, target.masked_fill(target.eq(255), 0), reduction="none") * all_valid).flatten(1).sum(1) / 6).sum() / batch
    expected = F.cross_entropy(logits, target.masked_fill(target.eq(255), 0), reduction="mean")
    left = torch.autograd.grad(all_loss, probe, retain_graph=True)[0]
    right = torch.autograd.grad(expected, probe)[0]
    require(torch.allclose(all_loss, expected, rtol=1e-15, atol=1e-15), "all-valid loss U2 equivalence")
    require(torch.allclose(left, right, rtol=1e-15, atol=1e-15), "all-valid gradient U2 equivalence")


def test_zero_valid_and_diagnostics():
    teacher, weak, strong, pseudo, confidence = inputs()
    confidence.zero_()
    pseudo[0, 0, 0] = 255
    before = model_snapshot(teacher)
    random.seed(41); np.random.seed(41); torch.manual_seed(41)
    rng_before = rng_snapshot()
    normalized, near_flat, loss, finite = common.compute_self_pseudo_saliency(
        teacher, strong, pseudo, donor_confidence=confidence,
        confidence_threshold=.95, ignore_label=255,
    )
    require(loss.item() == 0.0 and bool(finite), "graph-connected finite zero loss")
    require_rng_equal(rng_before, rng_snapshot())
    require(torch.equal(normalized, torch.zeros_like(normalized)), "zero saliency")
    require(bool(near_flat.all()), "zero saliency near-flat")
    require(torch.equal(common.candidate_probabilities(torch.zeros(4, 8), near_flat), torch.full((4, 8), 1 / 8)), "uniform 1/8")
    rng_before = rng_snapshot()
    result = common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=weak, strong_rgb=strong, hard_pseudo=pseudo,
        confidence=confidence, base_seed=2, rank=0, epoch=0, step=0,
        absolute_global_iteration=17, saliency_probe_rgb=strong,
        confidence_filtered_probe=True, confidence_threshold=.95, ignore_label=255,
    )
    diagnostics = result[3]
    require(diagnostics.u3_zero_valid_donors == 4, "zero-valid donors")
    require(diagnostics.u3_ignored_label_pixels == 1, "ignored pixels")
    require(diagnostics.total_candidate_draws == 32, "eight candidates per donor")
    require(diagnostics.paste_attempts == 4, "one paste attempt per receiver")
    require_model_equal(before, model_snapshot(teacher))
    require_rng_equal(rng_before, rng_snapshot())


def test_tensor_lineage_and_isolation():
    teacher, weak, strong, pseudo, confidence = inputs()
    student = nn.Conv2d(3, 4, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=.01, momentum=.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    teacher_before = model_snapshot(teacher)
    student_before = model_snapshot(student)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    expected = permutation(4, 2, 17)
    weak_before, strong_before = weak.clone(), strong.clone()
    common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=weak, strong_rgb=strong, hard_pseudo=pseudo,
        confidence=confidence, base_seed=2, rank=0, epoch=0, step=0,
        absolute_global_iteration=17, saliency_probe_rgb=strong,
        confidence_filtered_probe=True, confidence_threshold=.95, ignore_label=255,
    )
    require(torch.equal(teacher.seen, strong_before[expected]), "exact U3 probe donor")
    require(not torch.any(expected == torch.arange(4)), "derangement fixed point")
    require(torch.equal(torch.sort(expected).values, torch.arange(4)), "donor uniqueness")
    require(torch.equal(weak, weak_before) and torch.equal(strong, strong_before), "input mutation")
    require_model_equal(teacher_before, model_snapshot(teacher))
    require_model_equal(student_before, model_snapshot(student))
    require(optimizer_before == optimizer.state_dict(), "optimizer isolation")
    require(scheduler_before == scheduler.state_dict(), "scheduler isolation")


def test_common_pipeline_equivalence():
    teacher, weak, strong, pseudo, confidence = inputs()
    supplied = torch.linspace(0, 1, pseudo.numel()).reshape_as(confidence)
    near_flat = torch.zeros(4, dtype=torch.bool)
    original = common.compute_self_pseudo_saliency

    def fixed(_teacher, probe, target, **kwargs):
        return supplied, near_flat, torch.tensor(1.0), torch.tensor(True)

    common.compute_self_pseudo_saliency = fixed
    try:
        base = dict(teacher=teacher, weak_rgb=weak, strong_rgb=strong, hard_pseudo=pseudo,
                    confidence=confidence, base_seed=9, rank=0, epoch=2, step=1,
                    absolute_global_iteration=23, saliency_probe_rgb=strong)
        u2 = common.apply_u1_saliency_u2u(**base)
        u3 = common.apply_u1_saliency_u2u(**base, confidence_filtered_probe=True,
                                         confidence_threshold=.95, ignore_label=255)
    finally:
        common.compute_self_pseudo_saliency = original
    for left, right in zip(u2[:3], u3[:3]):
        require(torch.equal(left, right), "fixed-saliency common output")
    for field in ("total_receivers", "total_candidate_draws", "generated_invalid_candidates",
                  "selected_invalid_candidates", "paste_attempts", "nonempty_paste_attempts",
                  "near_flat_images", "derangement_attempts"):
        require(getattr(u2[3], field) == getattr(u3[3], field), f"common diagnostic {field}")


def test_static_contract():
    helper = (ROOT / "util/u1_saliency_u2u.py").read_text()
    train = (ROOT / "train_semi.py").read_text()
    require("detached_confidence.ge(float(confidence_threshold))" in helper, "inclusive threshold")
    require("safe_denominator = valid_count.clamp_min(1)" in helper, "safe denominator")
    require("reduction=\"none\"" in helper and "ignore_index=int(ignore_label)" in helper, "unreduced ignore CE")
    require("(input_grad,) = torch.autograd.grad(" in helper, "tuple unpack")
    require("create_graph=False" in helper and "retain_graph=False" in helper, "no higher-order graph")
    require("torch.where" not in helper, "unsafe zero division")
    require("probe_loss.backward" not in helper, "probe backward")
    require("hash(" not in helper, "Python hash")
    require(helper.count("dist.all_reduce") == 2, "fixed collective count")
    require("state_dict" not in helper and "get_rng_state" not in helper, "production snapshots")
    require('confidence_threshold=p_threshold if mix_branch == "u3" else None' in train, "runtime threshold forwarding")
    require('ignore_label=ignore if mix_branch == "u3" else None' in train, "runtime ignore forwarding")


def test_cuda_crop321():
    if not torch.cuda.is_available():
        print("UNVERIFIED test_cuda_crop321: RESOURCE UNAVAILABLE — CUDA unavailable")
        return
    device = torch.device("cuda", 0)
    teacher, weak, strong, pseudo, confidence = inputs(2, 321, 321, device)
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    before = rng_snapshot(); start = time.perf_counter()
    result = common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=weak, strong_rgb=strong, hard_pseudo=pseudo,
        confidence=confidence, base_seed=2, rank=0, epoch=0, step=0,
        absolute_global_iteration=0, saliency_probe_rgb=strong,
        confidence_filtered_probe=True, confidence_threshold=.95, ignore_label=255)
    torch.cuda.synchronize(device)
    require_rng_equal(before, rng_snapshot())
    require(result[3].paste_attempts == 2, "CUDA paste attempts")
    print(f"MEASURE u3_crop321_runtime_sec={time.perf_counter()-start:.6f} peak_allocated_bytes={torch.cuda.max_memory_allocated(device)} batch=2 shape=321x321 device=cuda:0")


def test_ddp(backend="gloo"):
    from torch.nn.parallel import DistributedDataParallel
    if not dist.is_initialized():
        dist.init_process_group(backend)
    rank = dist.get_rank()
    device = torch.device("cpu")
    device_ids = None
    teacher_module = TinyTeacher().eval()
    if backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        device_ids = [local_rank]
        teacher_module = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_module).to(device)
    weak = torch.arange(2 * 3 * 9 * 11, dtype=torch.float32, device=device).reshape(2, 3, 9, 11)
    strong = weak.mul(-0.25).add(1000).clone()
    teacher = DistributedDataParallel(teacher_module, device_ids=device_ids).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = torch.full_like(parameter, 3.0)
    with torch.no_grad():
        probabilities = teacher(weak)[0].softmax(1)
        confidence, pseudo = probabilities.max(1)
    before = model_snapshot(teacher)
    result = common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=weak, strong_rgb=strong, hard_pseudo=pseudo,
        confidence=confidence, base_seed=2, rank=rank, epoch=0, step=0,
        absolute_global_iteration=0, saliency_probe_rgb=strong,
        confidence_filtered_probe=True, confidence_threshold=.95, ignore_label=255)
    require_model_equal(before, model_snapshot(teacher))
    require(result[3].paste_attempts == 2, "DDP paste attempts")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--nccl", action="store_true")
    args = parser.parse_args()
    if args.ddp:
        test_ddp("nccl" if args.nccl else "gloo")
        print("PASS test_ddp_nccl_syncbn" if args.nccl else "PASS test_ddp_gloo")
        return
    start = time.perf_counter()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for test in (test_config_and_dispatch, test_masked_ce_math, test_zero_valid_and_diagnostics,
                 test_tensor_lineage_and_isolation, test_common_pipeline_equivalence,
                 test_static_contract, test_cuda_crop321):
        test(); print(f"PASS {test.__name__}")
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"MEASURE cpu_smoke_runtime_sec={time.perf_counter()-start:.6f} peak_rss_kib={peak_after} peak_delta_kib={max(0, peak_after-peak_before)} batch=4 shape=9x11 device=cpu")
    print("PASS smoke_u3_confidence_filtered_cross_view_saliency_u2u")


if __name__ == "__main__":
    main()
