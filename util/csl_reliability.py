import math

import torch


def _float_stat(value):
    return float(value.detach().item())


def compute_csl_reliability(
    teacher_probs,
    teacher_logits=None,
    features=None,
    cfg=None,
):
    """
    teacher_probs: [B, C, H, W]
    return reliability [B, H, W] in [0,1], detached, plus numeric stats.
    """
    del teacher_logits, features
    cfg = cfg or {}
    mode = cfg.get("reliability_mode", "entropy_margin")
    eps = float(cfg.get("eps", 1e-6))
    if mode != "entropy_margin":
        raise ValueError("CSL reliability backend currently supports entropy_margin only")
    if teacher_probs.dim() != 4:
        raise ValueError("teacher_probs must have shape [B,C,H,W]")

    with torch.no_grad():
        probs = teacher_probs.detach().float().clamp_min(eps)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)
        num_classes = probs.size(1)

        top2 = torch.topk(probs, k=min(2, num_classes), dim=1).values
        confidence = top2[:, 0]
        if num_classes > 1:
            margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
        else:
            margin = torch.ones_like(confidence)

        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
        entropy_norm = entropy / math.log(max(num_classes, 2))
        entropy_norm = entropy_norm.clamp(0.0, 1.0)
        entropy_rel = 1.0 - entropy_norm

        reliability = (confidence * entropy_rel * margin).clamp(0.0, 1.0).detach()
        stats = {
            "csl/mean_reliability": _float_stat(reliability.mean()),
            "csl/reliability_std": _float_stat(reliability.std(unbiased=False)),
            "csl/reliability_min": _float_stat(reliability.min()),
            "csl/reliability_max": _float_stat(reliability.max()),
            "csl/confidence_mean": _float_stat(confidence.mean()),
            "csl/entropy_mean": _float_stat(entropy_norm.mean()),
            "csl/margin_mean": _float_stat(margin.mean()),
        }

    return reliability, stats


def apply_csl_random_reliable_mask(
    reliability,
    mask_prob=0.3,
    training=True,
):
    """
    reliability: [B,H,W]
    return effective_weight [B,H,W], stats.
    """
    if reliability.dim() != 3:
        raise ValueError("reliability must have shape [B,H,W]")

    with torch.no_grad():
        raw = reliability.detach().clamp(0.0, 1.0)
        mask_prob = float(mask_prob)
        if training and mask_prob > 0.0:
            keep_prob = max(0.0, min(1.0, 1.0 - mask_prob))
            rand_keep = torch.empty_like(raw).bernoulli_(keep_prob)
            effective = raw * rand_keep
            masked_ratio = 1.0 - float(rand_keep.detach().mean().item())
        else:
            effective = raw
            masked_ratio = 0.0

        effective = effective.clamp(0.0, 1.0).detach()
        stats = {
            "csl/mask_prob": mask_prob,
            "csl/masked_ratio": masked_ratio,
            "csl/raw_reliability_mean": _float_stat(raw.mean()),
            "csl/effective_weight_mean": _float_stat(effective.mean()),
        }

    return effective, stats
