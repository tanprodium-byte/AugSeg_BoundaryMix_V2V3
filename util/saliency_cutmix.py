import math

import torch
import torch.nn.functional as F


def normalize_saliency_per_image(saliency, eps=1e-6):
    if saliency.dim() != 3:
        raise ValueError("saliency must have shape [B,H,W]")
    batch = saliency.shape[0]
    flat = saliency.view(batch, -1)
    s_min = flat.min(dim=1).values.view(batch, 1, 1)
    s_max = flat.max(dim=1).values.view(batch, 1, 1)
    return (saliency - s_min) / (s_max - s_min + eps)


def _first_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "pred", "out"):
            if key in output:
                return output[key]
    return output


def compute_labeled_teacher_saliency(teacher, x_l, y_l, ignore_index=255, eps=1e-6):
    x_src = x_l.detach().clone().requires_grad_(True)
    y_src = y_l.detach()

    teacher_was_training = teacher.training
    param_requires_grad = [p.requires_grad for p in teacher.parameters()]

    try:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        with torch.enable_grad():
            logits = _first_logits(teacher(x_src))
            loss_sal = F.cross_entropy(
                logits,
                y_src,
                ignore_index=ignore_index,
                reduction="mean",
            )
            grad = torch.autograd.grad(
                loss_sal,
                x_src,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

        saliency = grad.norm(p=2, dim=1)
        return normalize_saliency_per_image(saliency, eps=eps).detach()
    finally:
        for p, flag in zip(teacher.parameters(), param_requires_grad):
            p.requires_grad_(flag)
        if teacher_was_training:
            teacher.train()
        else:
            teacher.eval()


def _boxes_from_sampler_output(output, device):
    if isinstance(output, torch.Tensor):
        boxes = output.to(device=device, dtype=torch.long)
        if boxes.dim() != 2 or boxes.size(1) != 4:
            raise ValueError("box tensor must have shape [B,4]")
        return boxes

    if not isinstance(output, (tuple, list)) or len(output) != 4:
        raise ValueError("base_box_sampler must return [B,4] or four coordinate arrays")

    coords = []
    for coord in output:
        coords.append(torch.as_tensor(coord, device=device, dtype=torch.long))
    return torch.stack(coords, dim=1)


def sample_candidate_boxes_with_baseline_sampler(
    image_size,
    base_box_sampler,
    num_candidates=8,
    lam_sampler=None,
    device=None,
):
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")

    boxes = []
    for _ in range(int(num_candidates)):
        lam = lam_sampler() if lam_sampler is not None else None
        try:
            sampled = base_box_sampler(image_size, lam=lam)
        except TypeError:
            sampled = base_box_sampler(image_size)
        boxes.append(_boxes_from_sampler_output(sampled, device=device))
    return torch.stack(boxes, dim=1)


def box_mean_saliency(saliency, boxes):
    if saliency.dim() != 3:
        raise ValueError("saliency must have shape [B,H,W]")
    if boxes.dim() != 3 or boxes.size(-1) != 4:
        raise ValueError("boxes must have shape [B,K,4]")
    if saliency.size(0) != boxes.size(0):
        raise ValueError("saliency and boxes must have the same batch size")

    batch, num_candidates = boxes.shape[:2]
    height, width = saliency.shape[1:]
    scores = saliency.new_zeros((batch, num_candidates))

    for b in range(batch):
        for k in range(num_candidates):
            x1, y1, x2, y2 = boxes[b, k].tolist()
            x1 = max(0, min(int(x1), height))
            x2 = max(0, min(int(x2), height))
            y1 = max(0, min(int(y1), width))
            y2 = max(0, min(int(y2), width))
            if x2 <= x1 or y2 <= y1:
                scores[b, k] = 0.0
            else:
                scores[b, k] = saliency[b, x1:x2, y1:y2].mean()
    return scores


def sample_box_by_softmax(scores, temperature=0.2, eps=1e-6):
    if scores.dim() != 2:
        raise ValueError("scores must have shape [B,K]")
    temperature = max(float(temperature), eps)
    probs = torch.softmax(scores / temperature, dim=1)
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=1, keepdim=True)
    selected_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
    return selected_idx, probs


def _stats_from_scores(scores, selected_idx, probs, fallback_ratio=0.0):
    batch_index = torch.arange(scores.size(0), device=scores.device)
    selected_scores = scores[batch_index, selected_idx]
    selected_probs = probs[batch_index, selected_idx]
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    denom = math.log(max(scores.size(1), 2))
    # Normalized entropy in [0, 1] for comparable logs across different K.
    entropy = entropy / denom

    return {
        "saliency/score_selected": float(selected_scores.detach().mean().item()),
        "saliency/score_candidate_mean": float(scores.detach().mean().item()),
        "saliency/score_candidate_max": float(scores.detach().max(dim=1).values.mean().item()),
        "saliency/score_candidate_min": float(scores.detach().min(dim=1).values.mean().item()),
        "saliency/score_candidate_std": float(scores.detach().std(dim=1, unbiased=False).mean().item()),
        "saliency/prob_selected": float(selected_probs.detach().mean().item()),
        "saliency/prob_max": float(probs.detach().max(dim=1).values.mean().item()),
        "saliency/selection_entropy": float(entropy.detach().mean().item()),
        "saliency/fallback_ratio": float(fallback_ratio),
        "saliency/source_is_labeled_ratio": 1.0,
    }


def get_saliency_guided_boxes(
    teacher,
    source_images,
    source_labels,
    base_box_sampler,
    num_candidates=8,
    temperature=0.2,
    ignore_index=255,
    eps=1e-6,
    lam_sampler=None,
):
    saliency = compute_labeled_teacher_saliency(
        teacher,
        source_images,
        source_labels,
        ignore_index=ignore_index,
        eps=eps,
    )
    boxes = sample_candidate_boxes_with_baseline_sampler(
        source_images.size(),
        base_box_sampler,
        num_candidates=num_candidates,
        lam_sampler=lam_sampler,
        device=source_images.device,
    )
    scores = box_mean_saliency(saliency, boxes)
    selected_idx, probs = sample_box_by_softmax(scores, temperature=temperature, eps=eps)
    batch_index = torch.arange(boxes.size(0), device=boxes.device)
    selected_boxes = boxes[batch_index, selected_idx].detach()
    stats = _stats_from_scores(scores, selected_idx, probs, fallback_ratio=0.0)
    return selected_boxes, stats


def boxes_to_masks(boxes, mask_size, device=None, dtype=torch.float32):
    if boxes.dim() != 2 or boxes.size(1) != 4:
        raise ValueError("boxes must have shape [B,4]")
    batch, height, width = int(mask_size[0]), int(mask_size[-2]), int(mask_size[-1])
    masks = torch.zeros((batch, height, width), device=device or boxes.device, dtype=dtype)
    for b in range(batch):
        x1, y1, x2, y2 = boxes[b].tolist()
        x1 = max(0, min(int(x1), height))
        x2 = max(0, min(int(x2), height))
        y1 = max(0, min(int(y1), width))
        y2 = max(0, min(int(y2), width))
        if x2 > x1 and y2 > y1:
            masks[b, x1:x2, y1:y2] = 1.0
    return masks
