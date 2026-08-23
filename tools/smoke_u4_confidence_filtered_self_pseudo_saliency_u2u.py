#!/usr/bin/env python3
"""Focused side-effect-free smoke for U4 confidence-filtered weak saliency."""

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
U3_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/u3_confidence_filtered_cross_view_saliency_u2u_cutmix/config.yaml"
U4_CONFIG = ROOT / "exps/boundary_mix_v2_v3/voc_semi662/u4_confidence_filtered_self_pseudo_saliency_u2u_cutmix/config.yaml"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.register_buffer("marker", torch.tensor([7.0]))
        self.seen = None

    def forward(self, tensor):
        self.seen = tensor.detach().clone()
        return self.conv(tensor), None


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = torch.full_like(parameter, 3.0)
    return model


def snapshot(model):
    return {
        "parameters": [p.detach().clone() for p in model.parameters()],
        "buffers": [b.detach().clone() for b in model.buffers()],
        "modes": [m.training for m in model.modules()],
        "requires_grad": [p.requires_grad for p in model.parameters()],
        "grads": [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
    }


def require_snapshot_equal(left, right):
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
    teacher = freeze(TinyTeacher().to(device))
    with torch.no_grad():
        probabilities = teacher(weak)[0].softmax(1)
        confidence, pseudo = probabilities.max(1)
    return teacher, weak, strong, pseudo.detach(), confidence.detach()


def expected_permutation(batch, seed, iteration, device="cpu"):
    generator = common.make_torch_generator(seed, 0, iteration, common.STREAM_DERANGEMENT, device)
    return common.sample_derangement(batch, generator, torch.device(device))[0]


def u4_call(teacher, weak, strong, pseudo, confidence, *, seed=2, iteration=17):
    return common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=weak, strong_rgb=strong,
        hard_pseudo=pseudo, confidence=confidence, base_seed=seed,
        rank=0, epoch=0, step=0, absolute_global_iteration=iteration,
        confidence_filtered_probe=True, confidence_threshold=.95,
        ignore_label=255,
    )


def test_config_dispatch_and_identity():
    from train_semi import select_unlabeled_mix_branch
    u1 = yaml.safe_load(U1_CONFIG.read_text())
    u3 = yaml.safe_load(U3_CONFIG.read_text())
    u4 = yaml.safe_load(U4_CONFIG.read_text())
    require(select_unlabeled_mix_branch(False, True, 1.0, u4_enabled=True) == ("u4", 1, 1), "U4 dispatch")
    for flags in ((True, False, False, True), (False, True, False, True), (False, False, True, True)):
        try:
            select_unlabeled_mix_branch(flags[0], True, 1.0, u2_enabled=flags[1], u3_enabled=flags[2], u4_enabled=flags[3])
        except ValueError:
            pass
        else:
            raise AssertionError("four-way mutual exclusion")
    for section in ("dataset", "trainer", "criterion", "net", "checkpoint"):
        require(u4[section] == u1[section], f"U1 semantic drift: {section}")
    require(u4["dataset"]["ignore_label"] == 255, "runtime ignore")
    require(u4["trainer"]["unsupervised"]["threshold"] == .95, "runtime threshold")
    require(u4["u4_confidence_filtered_self_pseudo_saliency_u2u"]["enabled"] is True, "opt-in")
    require(u4["wandb"]["enable"] == u1["wandb"]["enable"], "W&B enablement")
    require(u4["hf"]["enabled"] == u1["hf"]["enabled"], "HF enablement")
    require(u4["wandb"]["name"] not in (u1["wandb"]["name"], u3["wandb"]["name"]), "W&B identity")
    require(u4["hf"]["path_in_repo"] not in (u1["hf"]["path_in_repo"], u3["hf"]["path_in_repo"]), "HF identity")


def test_weak_probe_lineage_and_neutrality():
    teacher, weak, strong, pseudo, confidence = inputs()
    student = nn.Conv2d(3, 4, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=.01, momentum=.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    teacher_before, student_before = snapshot(teacher), snapshot(student)
    optimizer_before, scheduler_before = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
    weak_before, strong_before, pseudo_before, confidence_before = weak.clone(), strong.clone(), pseudo.clone(), confidence.clone()
    permutation = expected_permutation(4, 2, 17)
    random.seed(41); np.random.seed(41); torch.manual_seed(41)
    rng_before = rng_snapshot()
    result = u4_call(teacher, weak, strong, pseudo, confidence)
    require(torch.equal(teacher.seen, weak_before[permutation]), "U4 probe is not exact weak donor")
    require(not torch.equal(teacher.seen, strong_before[permutation]), "augmented donor used as U4 probe")
    require(not torch.any(permutation == torch.arange(4)), "fixed point")
    require(torch.equal(torch.sort(permutation).values, torch.arange(4)), "donor uniqueness")
    require(torch.equal(weak, weak_before) and torch.equal(strong, strong_before), "RGB input mutation")
    require(torch.equal(pseudo, pseudo_before) and torch.equal(confidence, confidence_before), "target mutation")
    require_snapshot_equal(teacher_before, snapshot(teacher))
    require_snapshot_equal(student_before, snapshot(student))
    require(optimizer_before == optimizer.state_dict(), "optimizer isolation")
    require(scheduler_before == scheduler.state_dict(), "scheduler isolation")
    require_rng_equal(rng_before, rng_snapshot())
    require(result[3].u3_total_probe_pixels == pseudo.numel(), "shared filtered-probe diagnostics")


def test_masked_math_and_all_valid_equivalence():
    torch.manual_seed(17)
    batch, classes, height, width = 3, 4, 2, 3
    probe = torch.randn(batch, 3, height, width, requires_grad=True)
    layer = nn.Conv2d(3, classes, 1)
    logits = layer(probe)
    target = torch.tensor([[[255, 1, 2], [3, 0, 1]], [[1, 1, 2], [2, 3, 0]], [[3, 2, 1], [0, 1, 2]]])
    confidence = torch.tensor([[[0.1, .95, .94], [.99, .2, .3]], [[1., .95, .2], [.1, .99, .1]], [[1., 1., 1.], [1., 1., 1.]]])
    ce = F.cross_entropy(logits, target, reduction="none", ignore_index=255)
    mask = confidence.ge(.95) & target.ne(255)
    counts = mask.flatten(1).sum(1)
    loss = ((ce * mask.to(ce.dtype)).flatten(1).sum(1) / counts.clamp_min(1).to(ce.dtype)).sum() / batch
    grad_logits, grad_input = torch.autograd.grad(loss, (logits, probe), retain_graph=True)
    require(counts.tolist() == [2, 3, 6], "mixed valid counts")
    require(torch.equal(grad_logits.permute(0, 2, 3, 1)[~mask], torch.zeros_like(grad_logits.permute(0, 2, 3, 1)[~mask])), "masked output gradient")
    require(torch.isfinite(loss) and torch.isfinite(grad_input).all(), "masked CE finite")
    teacher, weak, _, pseudo, _ = inputs(batch=3, height=5, width=7)
    all_confidence = torch.ones_like(pseudo, dtype=weak.dtype)
    u1 = common.compute_self_pseudo_saliency(teacher, weak, pseudo)
    u4 = common.compute_self_pseudo_saliency(
        teacher, weak, pseudo, donor_confidence=all_confidence,
        confidence_threshold=.95, ignore_label=255,
    )
    require(torch.allclose(u1[2], u4[2], rtol=1e-6, atol=2e-7), "all-valid FP32 loss")
    require(torch.allclose(u1[0], u4[0], rtol=1e-6, atol=2e-7), "all-valid normalized saliency")
    require(torch.equal(u1[1], u4[1]), "all-valid near-flat")


def test_zero_valid_continuation():
    teacher, weak, strong, pseudo, confidence = inputs()
    confidence.zero_(); pseudo[0, 0, 0] = 255
    normalized, near_flat, loss, finite = common.compute_self_pseudo_saliency(
        teacher, weak, pseudo, donor_confidence=confidence,
        confidence_threshold=.95, ignore_label=255,
    )
    require(loss.item() == 0.0 and bool(finite), "finite graph-connected zero")
    require(torch.equal(normalized, torch.zeros_like(normalized)), "zero normalized saliency")
    require(bool(near_flat.all()), "zero near-flat")
    require(torch.equal(common.candidate_probabilities(torch.zeros(4, 8), near_flat), torch.full((4, 8), .125)), "uniform 1/8")
    result = u4_call(teacher, weak, strong, pseudo, confidence)
    require(result[3].u3_zero_valid_donors == 4, "shared zero-valid count")
    require(result[3].u3_ignored_label_pixels == 1, "shared ignored count")
    require(result[3].total_candidate_draws == 32, "eight candidates")
    require(result[3].paste_attempts == 4, "one paste attempt")


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
                    absolute_global_iteration=23)
        u1 = common.apply_u1_saliency_u2u(**base)
        u4 = common.apply_u1_saliency_u2u(
            **base, confidence_filtered_probe=True, confidence_threshold=.95,
            ignore_label=255)
    finally:
        common.compute_self_pseudo_saliency = original
    for left, right in zip(u1[:3], u4[:3]):
        require(torch.equal(left, right), "fixed-saliency common output")
    for field in ("total_receivers", "total_candidate_draws", "generated_invalid_candidates",
                  "selected_invalid_candidates", "paste_attempts", "nonempty_paste_attempts",
                  "near_flat_images", "derangement_attempts"):
        require(getattr(u1[3], field) == getattr(u4[3], field), f"common diagnostic {field}")


def test_static_contract():
    helper = (ROOT / "util/u1_saliency_u2u.py").read_text()
    train = (ROOT / "train_semi.py").read_text()
    require("probe_rgb = weak_rgb if saliency_probe_rgb is None else saliency_probe_rgb" in helper, "weak probe default")
    require("detached_confidence.ge(float(confidence_threshold))" in helper, "inclusive threshold")
    require("safe_denominator = valid_count.clamp_min(1)" in helper, "safe denominator")
    require("(input_grad,) = torch.autograd.grad(" in helper, "autograd tuple")
    require("torch.where" not in helper and "probe_loss.backward" not in helper, "unsafe probe path")
    require("hash(" not in helper and helper.count("dist.all_reduce") == 2, "RNG/collective contract")
    require("state_dict" not in helper and "get_rng_state" not in helper, "production snapshots")
    require('confidence_filtered_probe = True' in train and 'saliency_probe_rgb = image_u_aug' not in train.split('if mix_branch == "u4":', 1)[1].split('apply_u1_saliency_u2u', 1)[0], "U4 weak integration")
    require('key.replace("u3/", "u4/", 1)' in train, "U4 diagnostic namespace")
    gradient = torch.tensor([[[[3.0]], [[4.0]], [[0.0]]]])
    require(common.rgb_l2_saliency(gradient).item() == 5.0, "RGB L2 square root")


def test_cuda_crop321():
    if not torch.cuda.is_available():
        print("UNVERIFIED test_cuda_crop321: RESOURCE UNAVAILABLE — CUDA unavailable")
        return
    device = torch.device("cuda", 0)
    teacher, weak, strong, pseudo, confidence = inputs(2, 321, 321, device)
    before = snapshot(teacher); rng_before = rng_snapshot()
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = u4_call(teacher, weak, strong, pseudo, confidence, iteration=0)
    torch.cuda.synchronize(device)
    require_snapshot_equal(before, snapshot(teacher)); require_rng_equal(rng_before, rng_snapshot())
    require(result[3].paste_attempts == 2, "CUDA paste attempts")
    print(f"MEASURE u4_crop321_runtime_sec={time.perf_counter()-start:.6f} peak_allocated_bytes={torch.cuda.max_memory_allocated(device)} batch=2 shape=321x321 device=cuda:0")


def test_ddp(backend="gloo"):
    from torch.nn.parallel import DistributedDataParallel
    if not dist.is_initialized():
        dist.init_process_group(backend)
    rank = dist.get_rank(); device = torch.device("cpu"); device_ids = None
    module = TinyTeacher().eval()
    if backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank); device_ids = [local_rank]
        module = nn.SyncBatchNorm.convert_sync_batchnorm(module).to(device)
    weak = torch.arange(2 * 3 * 9 * 11, dtype=torch.float32, device=device).reshape(2, 3, 9, 11)
    strong = weak.mul(-.25).add(1000).clone()
    teacher = DistributedDataParallel(module, device_ids=device_ids).eval()
    for p in teacher.parameters(): p.requires_grad_(False); p.grad = torch.full_like(p, 3.0)
    with torch.no_grad(): confidence, pseudo = teacher(weak)[0].softmax(1).max(1)
    before = snapshot(teacher)
    result = common.apply_u1_saliency_u2u(
        teacher=teacher, weak_rgb=weak, strong_rgb=strong, hard_pseudo=pseudo,
        confidence=confidence, base_seed=2, rank=rank, epoch=0, step=0,
        absolute_global_iteration=0, confidence_filtered_probe=True,
        confidence_threshold=.95, ignore_label=255)
    require_snapshot_equal(before, snapshot(teacher)); require(result[3].paste_attempts == 2, "DDP paste")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--ddp", action="store_true"); parser.add_argument("--nccl", action="store_true")
    args = parser.parse_args()
    if args.ddp:
        test_ddp("nccl" if args.nccl else "gloo")
        print("PASS test_ddp_nccl_syncbn" if args.nccl else "PASS test_ddp_gloo")
        return
    start = time.perf_counter(); peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for test in (test_config_dispatch_and_identity, test_weak_probe_lineage_and_neutrality,
                 test_masked_math_and_all_valid_equivalence, test_zero_valid_continuation,
                 test_common_pipeline_equivalence, test_static_contract, test_cuda_crop321):
        test(); print(f"PASS {test.__name__}")
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"MEASURE cpu_smoke_runtime_sec={time.perf_counter()-start:.6f} peak_rss_kib={peak_after} peak_delta_kib={max(0, peak_after-peak_before)} batch=4 shape=9x11 device=cpu")
    print("PASS smoke_u4_confidence_filtered_self_pseudo_saliency_u2u")


if __name__ == "__main__":
    main()
