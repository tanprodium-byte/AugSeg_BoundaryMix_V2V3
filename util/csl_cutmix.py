import math

import numpy as np
import torch

from util.saliency_cutmix import (
    box_mean_saliency,
    sample_box_by_softmax,
    sample_candidate_boxes_with_baseline_sampler,
)


def _stats_from_csl_scores(scores, selected_idx, probs, reliability, boxes, fallback_ratio=0.0):
    batch_index = torch.arange(scores.size(0), device=scores.device)
    selected_scores = scores[batch_index, selected_idx]
    selected_probs = probs[batch_index, selected_idx]
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    entropy = entropy / math.log(max(scores.size(1), 2))
    selected_boxes = boxes[batch_index, selected_idx].unsqueeze(1)
    selected_reliability = box_mean_saliency(reliability, selected_boxes).squeeze(1)

    return {
        "csl_cutmix/score_selected": float(selected_scores.detach().mean().item()),
        "csl_cutmix/score_candidate_mean": float(scores.detach().mean().item()),
        "csl_cutmix/score_candidate_max": float(scores.detach().max(dim=1).values.mean().item()),
        "csl_cutmix/score_candidate_min": float(scores.detach().min(dim=1).values.mean().item()),
        "csl_cutmix/score_candidate_std": float(scores.detach().std(dim=1, unbiased=False).mean().item()),
        "csl_cutmix/prob_selected": float(selected_probs.detach().mean().item()),
        "csl_cutmix/prob_max": float(probs.detach().max(dim=1).values.mean().item()),
        "csl_cutmix/selection_entropy": float(entropy.detach().mean().item()),
        "csl_cutmix/target_reliability_selected": float(selected_reliability.detach().mean().item()),
        "csl_cutmix/fallback_ratio": float(fallback_ratio),
    }


def _fallback_boxes(reliability, base_box_sampler):
    image_size = (reliability.size(0), 1, reliability.size(1), reliability.size(2))
    sampled = base_box_sampler(image_size)
    if isinstance(sampled, torch.Tensor):
        boxes = sampled.to(device=reliability.device, dtype=torch.long)
    else:
        coords = [torch.as_tensor(x, device=reliability.device, dtype=torch.long) for x in sampled]
        boxes = torch.stack(coords, dim=1)
    return boxes


def get_csl_guided_boxes(
    reliability,
    base_box_sampler,
    num_candidates=8,
    temperature=0.2,
    policy="low_reliability",
    *,
    strict=False,
):
    """
    reliability: [B,H,W], CSL reliability on target unlabeled.
    return selected boxes [B,4] and stats.
    """
    if reliability.dim() != 3:
        raise ValueError("reliability must have shape [B,H,W]")
    if policy != "low_reliability":
        raise ValueError("csl_cutmix policy currently supports low_reliability only")

    with torch.no_grad():
        reliability = reliability.detach().clamp(0.0, 1.0)
        try:
            image_size = (reliability.size(0), 1, reliability.size(1), reliability.size(2))
            boxes = sample_candidate_boxes_with_baseline_sampler(
                image_size,
                base_box_sampler,
                num_candidates=int(num_candidates),
                lam_sampler=lambda: np.random.beta(4, 4),
                device=reliability.device,
            )
            score_map = 1.0 - reliability
            scores = box_mean_saliency(score_map, boxes)
            selected_idx, probs = sample_box_by_softmax(scores, temperature=float(temperature))
            batch_index = torch.arange(reliability.size(0), device=reliability.device)
            selected_boxes = boxes[batch_index, selected_idx].detach()
            if strict:
                height, width = reliability.shape[1:]
                valid = (
                    (selected_boxes[:, 0] >= 0)
                    & (selected_boxes[:, 0] < selected_boxes[:, 2])
                    & (selected_boxes[:, 2] <= height)
                    & (selected_boxes[:, 1] >= 0)
                    & (selected_boxes[:, 1] < selected_boxes[:, 3])
                    & (selected_boxes[:, 3] <= width)
                )
                if not bool(valid.all()):
                    bad = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
                    raise ValueError(f"invalid selected CSL boxes for batch indices {bad}")
            stats = _stats_from_csl_scores(scores, selected_idx, probs, reliability, boxes, fallback_ratio=0.0)
            return selected_boxes, stats
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"strict CSL guided box selection failed for batch_size={reliability.size(0)} "
                    f"and spatial_shape={tuple(reliability.shape[1:])}"
                ) from exc
            selected_boxes = _fallback_boxes(reliability, base_box_sampler).detach()
            scores = torch.zeros((reliability.size(0), 1), device=reliability.device)
            probs = torch.ones_like(scores)
            stats = _stats_from_csl_scores(
                scores,
                torch.zeros(reliability.size(0), device=reliability.device, dtype=torch.long),
                probs,
                reliability,
                selected_boxes.unsqueeze(1),
                fallback_ratio=1.0,
            )
            return selected_boxes, stats
