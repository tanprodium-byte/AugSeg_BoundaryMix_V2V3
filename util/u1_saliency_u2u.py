"""U1 self-pseudo saliency U-to-U CutMix.

The implementation is deliberately self-contained: it preserves the legacy S1
rectangle distribution while keeping every U1 random decision off the global
Python, NumPy, and Torch RNG streams.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


U1_RNG_POLICY_VERSION = "u1_rng_policy_v1"
STREAM_DERANGEMENT = 1
STREAM_CANDIDATES = 2
STREAM_SELECTION = 3
STREAM_DESTINATION = 4
NUM_CANDIDATES = 8
TEMPERATURE = 0.2


class U1SynchronizedError(RuntimeError):
    """A U1 failure that has been synchronized across participating ranks."""


@dataclass
class U1Diagnostics:
    total_receivers: int = 0
    total_candidate_draws: int = 0
    generated_invalid_candidates: int = 0
    receivers_with_invalid_candidate: int = 0
    selected_invalid_candidates: int = 0
    paste_attempts: int = 0
    nonempty_paste_attempts: int = 0
    near_flat_images: int = 0
    derangement_attempts: int = 0

    def as_dict(self) -> Dict[str, int | float | str]:
        paste_rate = self.paste_attempts / max(self.total_receivers, 1)
        mixed_rate = self.nonempty_paste_attempts / max(self.total_receivers, 1)
        return {
            "u1/total_receivers": self.total_receivers,
            "u1/total_candidate_draws": self.total_candidate_draws,
            "u1/generated_invalid_candidates": self.generated_invalid_candidates,
            "u1/receivers_with_invalid_candidate": self.receivers_with_invalid_candidate,
            "u1/selected_invalid_candidates": self.selected_invalid_candidates,
            "u1/paste_attempts": self.paste_attempts,
            "u1/nonempty_paste_attempts": self.nonempty_paste_attempts,
            "u1/paste_attempt_rate": paste_rate,
            "u1/nonempty_mixed_target_rate": mixed_rate,
            "u1/near_flat_images": self.near_flat_images,
            "u1/derangement_attempts": self.derangement_attempts,
            "u1/rng_policy": U1_RNG_POLICY_VERSION,
        }

    def add_(self, other: "U1Diagnostics") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))

    def reset_(self) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, 0)


def _stream_digest(base_seed: int, rank: int, iteration: int, stream_id: int) -> bytes:
    fields = (base_seed, rank, iteration, stream_id)
    if any(int(value) < 0 or int(value) >= (1 << 64) for value in fields):
        raise ValueError("U1 RNG seed fields must be unsigned 64-bit integers")
    serialized = struct.pack("<QQQQ", *(int(value) for value in fields))
    return hashlib.sha256(serialized).digest()


def make_numpy_rng(base_seed: int, rank: int, iteration: int, stream_id: int) -> np.random.RandomState:
    digest = _stream_digest(base_seed, rank, iteration, stream_id)
    words = np.frombuffer(digest, dtype="<u4").copy()
    return np.random.RandomState(words)


def make_torch_generator(
    base_seed: int,
    rank: int,
    iteration: int,
    stream_id: int,
    device: torch.device | str,
) -> torch.Generator:
    digest = _stream_digest(base_seed, rank, iteration, stream_id)
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) & ((1 << 63) - 1)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator


def synchronized_failure_check(
    local_failed: bool,
    category: str,
    *,
    device: torch.device,
    rank: int,
    epoch: int,
    step: int,
    iteration: int,
    detail: str = "",
) -> None:
    """Synchronize a local flag at a fixed call site, then fail on every rank."""
    flag = torch.tensor([int(bool(local_failed))], dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if int(flag.item()):
        context = (
            f"category={category} rank={rank} epoch={epoch} step={step} "
            f"global_iteration={iteration}"
        )
        if detail:
            context += f" detail={detail}"
        raise U1SynchronizedError(context)


def sample_derangement(batch_size: int, generator: torch.Generator, device: torch.device) -> Tuple[torch.Tensor, int]:
    if batch_size < 2:
        raise ValueError("U1 requires local unlabeled batch size B >= 2")
    indices = torch.arange(batch_size, device=device)
    for attempt in range(1, 129):
        permutation = torch.randperm(batch_size, generator=generator, device=device)
        if not torch.any(permutation == indices):
            return permutation, attempt
    raise RuntimeError("U1 derangement exhausted 128 attempts")


def sample_legacy_s1_candidates(
    batch_size: int,
    height: int,
    width: int,
    rng: np.random.RandomState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce S1's Beta(8,2), int, center, clipping, and half-open bounds."""
    candidates = np.empty((batch_size, NUM_CANDIDATES, 4), dtype=np.int64)
    valid = np.empty((batch_size, NUM_CANDIDATES), dtype=np.bool_)
    for candidate_index in range(NUM_CANDIDATES):
        lam = rng.beta(8, 2)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_height = int(height * cut_ratio)
        cut_width = int(width * cut_ratio)
        center_row = rng.randint(low=int(height / 8), high=height, size=[batch_size])
        center_col = rng.randint(low=int(width / 8), high=width, size=[batch_size])
        row1 = np.clip(center_row - cut_height // 2, 0, height)
        col1 = np.clip(center_col - cut_width // 2, 0, width)
        row2 = np.clip(center_row + cut_height // 2, 0, height)
        col2 = np.clip(center_col + cut_width // 2, 0, width)
        candidates[:, candidate_index, 0] = row1
        candidates[:, candidate_index, 1] = col1
        candidates[:, candidate_index, 2] = row2
        candidates[:, candidate_index, 3] = col2
        valid[:, candidate_index] = (row2 > row1) & (col2 > col1)
    return torch.from_numpy(candidates), torch.from_numpy(valid)


def _teacher_logits(teacher: torch.nn.Module, probe_input: torch.Tensor) -> torch.Tensor:
    # Bypass DDP's buffer broadcast while using the exact EMA Teacher module.
    probe_model = teacher.module if hasattr(teacher, "module") else teacher
    output = probe_model(probe_input)
    logits = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(logits, torch.Tensor) or logits.dim() != 4:
        raise ValueError("U1 Teacher probe must return logits [B,C,H,W]")
    return logits


def rgb_l2_saliency(input_grad: torch.Tensor) -> torch.Tensor:
    if input_grad.dim() != 4 or input_grad.size(1) != 3:
        raise ValueError("U1 probe input gradient must have shape [B,3,H,W]")
    return torch.sqrt(
        input_grad[:, 0] ** 2
        + input_grad[:, 1] ** 2
        + input_grad[:, 2] ** 2
    )


def normalize_saliency(s_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if s_raw.dim() != 3 or not s_raw.is_floating_point():
        raise ValueError("raw saliency must be a floating tensor [B,H,W]")
    flat = s_raw.flatten(1)
    s_min = flat.min(dim=1).values
    s_max = flat.max(dim=1).values
    span = s_max - s_min
    normalized = torch.zeros_like(s_raw)
    nonconstant = span != 0
    if nonconstant.any():
        normalized[nonconstant] = (
            s_raw[nonconstant] - s_min[nonconstant, None, None]
        ) / span[nonconstant, None, None]
    epsilon = torch.finfo(s_raw.dtype).eps
    scale = torch.maximum(torch.maximum(s_max.abs(), s_min.abs()), torch.full_like(s_max, epsilon))
    rho = span / scale
    near_flat = rho <= max(32 * epsilon, 1e-6)
    return normalized.detach(), near_flat.detach()


def compute_self_pseudo_saliency(
    teacher: torch.nn.Module,
    donor_weak: torch.Tensor,
    donor_pseudo: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached normalized saliency, near-flat flags, and probe loss."""
    probe_model = teacher.module if hasattr(teacher, "module") else teacher
    if teacher.training or any(module.training for module in teacher.modules()):
        raise RuntimeError("U1 Teacher probe requires the unchanged eval-mode Teacher")
    if any(parameter.requires_grad for parameter in probe_model.parameters()):
        raise RuntimeError("U1 Teacher parameters must remain frozen during the probe")

    probe_input = donor_weak.detach().clone().requires_grad_(True)
    detached_target = donor_pseudo.detach()
    probe_logits = _teacher_logits(teacher, probe_input)
    if probe_logits.shape[0] != detached_target.shape[0] or probe_logits.shape[2:] != detached_target.shape[1:]:
        raise ValueError("CONFLICT — PROBE LOGIT/TARGET SPATIAL RESOLUTION MISMATCH")
    probe_loss = F.cross_entropy(probe_logits, detached_target, reduction="mean")
    (input_grad,) = torch.autograd.grad(
        outputs=probe_loss,
        inputs=(probe_input,),
        create_graph=False,
        retain_graph=False,
    )
    s_raw = rgb_l2_saliency(input_grad)
    normalized, near_flat = normalize_saliency(s_raw)
    return normalized, near_flat, probe_loss.detach(), torch.isfinite(s_raw).all().detach()


def score_candidates(saliency: torch.Tensor, candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, count = candidates.shape[:2]
    scores = saliency.new_zeros((batch, count))
    valid = torch.zeros((batch, count), dtype=torch.bool, device=saliency.device)
    for receiver in range(batch):
        for candidate_index in range(count):
            row1, col1, row2, col2 = (int(v) for v in candidates[receiver, candidate_index].tolist())
            is_valid = row2 > row1 and col2 > col1
            valid[receiver, candidate_index] = is_valid
            if is_valid:
                scores[receiver, candidate_index] = saliency[receiver, row1:row2, col1:col2].mean()
            else:
                scores[receiver, candidate_index] = 0.0
    return scores, valid


def candidate_probabilities(scores: torch.Tensor, near_flat: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(scores / TEMPERATURE, dim=1)
    if near_flat.any():
        probabilities = probabilities.clone()
        probabilities[near_flat] = 1.0 / NUM_CANDIDATES
    return probabilities


def relocate_selected(
    receiver_strong: torch.Tensor,
    donor_weak: torch.Tensor,
    receiver_pseudo: torch.Tensor,
    donor_pseudo: torch.Tensor,
    receiver_confidence: torch.Tensor,
    donor_confidence: torch.Tensor,
    candidates: torch.Tensor,
    selected: torch.Tensor,
    destination_generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mixed_rgb = receiver_strong.clone()
    mixed_pseudo = receiver_pseudo.detach().clone()
    mixed_confidence = receiver_confidence.detach().clone()
    batch, _, height, width = receiver_strong.shape
    nonempty = torch.zeros(batch, dtype=torch.bool, device=receiver_strong.device)
    for receiver in range(batch):
        box = candidates[receiver, int(selected[receiver])]
        row1, col1, row2, col2 = (int(v) for v in box.tolist())
        crop_h, crop_w = row2 - row1, col2 - col1
        # Draw once even for zero-area geometry, preserving the locked contract.
        dst_row = int(torch.randint(0, height - crop_h + 1, (1,), generator=destination_generator, device=receiver_strong.device).item())
        dst_col = int(torch.randint(0, width - crop_w + 1, (1,), generator=destination_generator, device=receiver_strong.device).item())
        dst_row2, dst_col2 = dst_row + crop_h, dst_col + crop_w
        mixed_rgb[receiver, :, dst_row:dst_row2, dst_col:dst_col2] = donor_weak[
            receiver, :, row1:row2, col1:col2
        ]
        mixed_pseudo[receiver, dst_row:dst_row2, dst_col:dst_col2] = donor_pseudo[
            receiver, row1:row2, col1:col2
        ]
        mixed_confidence[receiver, dst_row:dst_row2, dst_col:dst_col2] = donor_confidence[
            receiver, row1:row2, col1:col2
        ]
        nonempty[receiver] = crop_h > 0 and crop_w > 0
    return mixed_rgb, mixed_pseudo, mixed_confidence, nonempty


def apply_u1_saliency_u2u(
    *,
    teacher: torch.nn.Module,
    weak_rgb: torch.Tensor,
    strong_rgb: torch.Tensor,
    hard_pseudo: torch.Tensor,
    confidence: torch.Tensor,
    base_seed: int,
    rank: int,
    epoch: int,
    step: int,
    absolute_global_iteration: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, U1Diagnostics]:
    """Apply one U1 mixing operation to every local receiver."""
    device = weak_rgb.device
    shape_ok = weak_rgb.dim() == 4
    batch = int(weak_rgb.size(0)) if weak_rgb.dim() >= 1 else 0
    expected_spatial = tuple(weak_rgb.shape[2:]) if shape_ok else ()
    local_structural_failure = (
        not shape_ok
        or batch < 2
        or weak_rgb.size(1) != 3
        or strong_rgb.shape != weak_rgb.shape
        or hard_pseudo.shape != (batch, *expected_spatial)
        or confidence.shape != hard_pseudo.shape
        or strong_rgb.device != device
        or hard_pseudo.device != device
        or confidence.device != device
    )
    synchronized_failure_check(
        local_structural_failure,
        "invalid_u1_input",
        device=device,
        rank=rank,
        epoch=epoch,
        step=step,
        iteration=absolute_global_iteration,
        detail=f"weak={tuple(weak_rgb.shape)} strong={tuple(strong_rgb.shape)}",
    )
    _, channels, height, width = weak_rgb.shape

    derangement_generator = make_torch_generator(
        base_seed, rank, absolute_global_iteration, STREAM_DERANGEMENT, device
    )
    permutation = None
    attempts = 0
    try:
        permutation, attempts = sample_derangement(batch, derangement_generator, device)
        derangement_failed = False
    except (ValueError, RuntimeError):
        derangement_failed = True
    synchronized_failure_check(
        derangement_failed,
        "derangement_failure",
        device=device,
        rank=rank,
        epoch=epoch,
        step=step,
        iteration=absolute_global_iteration,
    )
    assert permutation is not None

    donor_weak = weak_rgb[permutation].detach()
    donor_pseudo = hard_pseudo[permutation].detach()
    donor_confidence = confidence[permutation].detach()
    probe_detail = ""
    try:
        normalized_saliency, near_flat, probe_loss, raw_saliency_finite = compute_self_pseudo_saliency(
            teacher, donor_weak, donor_pseudo
        )
        probe_structural_failure = False
    except (RuntimeError, ValueError) as exc:
        probe_structural_failure = True
        probe_detail = str(exc)
        normalized_saliency = near_flat = probe_loss = raw_saliency_finite = None
    synchronized_failure_check(
        probe_structural_failure,
        "teacher_probe_structural_failure",
        device=device,
        rank=rank,
        epoch=epoch,
        step=step,
        iteration=absolute_global_iteration,
        detail=probe_detail,
    )
    assert normalized_saliency is not None
    assert near_flat is not None
    assert probe_loss is not None
    assert raw_saliency_finite is not None
    probe_nonfinite = not bool(torch.isfinite(probe_loss).item())
    raw_or_normalized_nonfinite = (
        not bool(raw_saliency_finite.item())
        or not bool(torch.isfinite(normalized_saliency).all().item())
    )
    synchronized_failure_check(
        probe_nonfinite or raw_or_normalized_nonfinite,
        "nonfinite_probe_or_saliency",
        device=device,
        rank=rank,
        epoch=epoch,
        step=step,
        iteration=absolute_global_iteration,
    )

    geometry_rng = make_numpy_rng(base_seed, rank, absolute_global_iteration, STREAM_CANDIDATES)
    candidates_cpu, _ = sample_legacy_s1_candidates(batch, height, width, geometry_rng)
    candidates = candidates_cpu.to(device=device)
    scores, valid = score_candidates(normalized_saliency, candidates)
    valid_scores_finite = torch.isfinite(scores[valid]).all() if valid.any() else torch.tensor(True, device=device)
    probabilities = candidate_probabilities(scores, near_flat)
    selection_nonfinite = not bool(valid_scores_finite.item()) or not bool(torch.isfinite(probabilities).all().item())
    synchronized_failure_check(
        selection_nonfinite,
        "nonfinite_candidate_selection",
        device=device,
        rank=rank,
        epoch=epoch,
        step=step,
        iteration=absolute_global_iteration,
    )
    selection_generator = make_torch_generator(
        base_seed, rank, absolute_global_iteration, STREAM_SELECTION, device
    )
    selected = torch.multinomial(probabilities, 1, generator=selection_generator).squeeze(1)
    destination_generator = make_torch_generator(
        base_seed, rank, absolute_global_iteration, STREAM_DESTINATION, device
    )
    mixed_rgb, mixed_pseudo, mixed_confidence, nonempty = relocate_selected(
        strong_rgb,
        donor_weak,
        hard_pseudo,
        donor_pseudo,
        confidence,
        donor_confidence,
        candidates,
        selected,
        destination_generator,
    )
    selected_valid = valid[torch.arange(batch, device=device), selected]
    diagnostics = U1Diagnostics(
        total_receivers=batch,
        total_candidate_draws=batch * NUM_CANDIDATES,
        generated_invalid_candidates=int((~valid).sum().item()),
        receivers_with_invalid_candidate=int((~valid).any(dim=1).sum().item()),
        selected_invalid_candidates=int((~selected_valid).sum().item()),
        paste_attempts=batch,
        nonempty_paste_attempts=int(nonempty.sum().item()),
        near_flat_images=int(near_flat.sum().item()),
        derangement_attempts=attempts,
    )
    return mixed_rgb, mixed_pseudo, mixed_confidence, diagnostics


def aggregate_diagnostics(diagnostics: U1Diagnostics, device: torch.device) -> Dict[str, int | float | str]:
    """Aggregate once at a caller-chosen fixed logging boundary; consumes no RNG."""
    names: Iterable[str] = diagnostics.__dataclass_fields__.keys()
    values = torch.tensor([getattr(diagnostics, name) for name in names], dtype=torch.long, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    aggregated = U1Diagnostics(**{name: int(value) for name, value in zip(names, values.tolist())})
    return aggregated.as_dict()
