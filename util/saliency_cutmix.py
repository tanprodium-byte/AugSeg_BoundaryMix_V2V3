import math
from collections import deque

import numpy as np
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


def connected_components_from_gt(
    label,
    *,
    ignore_label=255,
    foreground_only=True,
    connectivity=8,
    min_component_area=64,
    max_component_area=20000,
):
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if isinstance(label, torch.Tensor):
        label_np = label.detach().cpu().numpy()
    else:
        label_np = np.asarray(label)
    if label_np.ndim != 2:
        raise ValueError("label must have shape [H,W]")

    height, width = label_np.shape
    visited = np.zeros((height, width), dtype=bool)
    components = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for x in range(height):
        for y in range(width):
            cls = int(label_np[x, y])
            if visited[x, y] or cls == int(ignore_label) or (foreground_only and cls == 0):
                continue

            queue = deque([(x, y)])
            visited[x, y] = True
            pixels = []
            while queue:
                px, py = queue.popleft()
                pixels.append((px, py))
                for dx, dy in offsets:
                    nx, ny = px + dx, py + dy
                    if nx < 0 or nx >= height or ny < 0 or ny >= width or visited[nx, ny]:
                        continue
                    if int(label_np[nx, ny]) != cls:
                        continue
                    visited[nx, ny] = True
                    queue.append((nx, ny))

            area = len(pixels)
            if area < int(min_component_area) or area > int(max_component_area):
                continue
            mask = np.zeros((height, width), dtype=bool)
            xs = []
            ys = []
            for px, py in pixels:
                mask[px, py] = True
                xs.append(px)
                ys.append(py)
            components.append(
                {
                    "class_id": cls,
                    "area": area,
                    "mask": mask,
                    "bbox": (min(xs), min(ys), max(xs) + 1, max(ys) + 1),
                }
            )

    return components


def bbox_from_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = np.asarray(mask, dtype=bool)
    if mask_np.ndim != 2:
        raise ValueError("mask must have shape [H,W]")
    coords = np.argwhere(mask_np)
    if coords.size == 0:
        return None
    x1, y1 = coords.min(axis=0)
    x2, y2 = coords.max(axis=0) + 1
    return int(x1), int(y1), int(x2), int(y2)


def expand_box(box, image_shape, expand_ratio=1.2):
    if len(image_shape) < 2:
        raise ValueError("image_shape must include height and width")
    height, width = int(image_shape[-2]), int(image_shape[-1])
    x1, y1, x2, y2 = [int(v) for v in box]
    box_h = max(1, x2 - x1)
    box_w = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    new_h = box_h * float(expand_ratio)
    new_w = box_w * float(expand_ratio)
    nx1 = max(0, int(math.floor(cx - new_h / 2.0)))
    ny1 = max(0, int(math.floor(cy - new_w / 2.0)))
    nx2 = min(height, int(math.ceil(cx + new_h / 2.0)))
    ny2 = min(width, int(math.ceil(cy + new_w / 2.0)))
    if nx2 <= nx1:
        nx2 = min(height, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(width, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _component_stats(component_records, selected_records, fallback_count, batch_size):
    valid_counts = [len(records) for records in component_records]
    scores = [rec["score"] for records in component_records for rec in records]
    selected_valid = [rec for rec in selected_records if rec is not None]
    box_areas = [
        (rec["box"][2] - rec["box"][0]) * (rec["box"][3] - rec["box"][1])
        for rec in selected_valid
    ]
    entropies = [rec["entropy"] for rec in selected_valid]
    selected_scores = [rec["score"] for rec in selected_valid]
    selected_probs = [rec["prob_selected"] for rec in selected_valid]
    max_probs = [rec["prob_max"] for rec in selected_valid]

    def _mean(values, default=0.0):
        return float(np.mean(values)) if values else float(default)

    return {
        "saliency/score_selected": float(_mean(selected_scores)),
        "saliency/score_candidate_mean": float(_mean(scores)),
        "saliency/score_candidate_max": float(max(scores) if scores else 0.0),
        "saliency/score_candidate_min": float(min(scores) if scores else 0.0),
        "saliency/score_candidate_std": float(np.std(scores) if scores else 0.0),
        "saliency/prob_selected": float(_mean(selected_probs)),
        "saliency/prob_max": float(_mean(max_probs)),
        "saliency/num_components": float(_mean(valid_counts)),
        "saliency/num_valid_components": float(_mean(valid_counts)),
        "saliency/selected_component_class": float(_mean([rec["class_id"] for rec in selected_valid], -1.0)),
        "saliency/selected_component_area": float(_mean([rec["area"] for rec in selected_valid])),
        "saliency/selected_component_score": float(_mean(selected_scores)),
        "saliency/component_score_mean": float(_mean(scores)),
        "saliency/component_score_max": float(max(scores) if scores else 0.0),
        "saliency/component_box_area": float(_mean(box_areas)),
        "saliency/selection_entropy": float(_mean(entropies)),
        "saliency/fallback_ratio": float(fallback_count) / float(max(batch_size, 1)),
        "saliency/source_is_labeled_ratio": 1.0,
    }


def get_saliency_component_guided_boxes(
    teacher,
    source_images,
    source_labels,
    base_box_sampler,
    *,
    temperature=0.2,
    ignore_index=255,
    eps=1e-6,
    lam_sampler=None,
    connectivity=8,
    foreground_only=True,
    min_component_area=64,
    max_component_area=20000,
    box_expand_ratio=1.2,
):
    saliency = compute_labeled_teacher_saliency(
        teacher,
        source_images,
        source_labels,
        ignore_index=ignore_index,
        eps=eps,
    )
    lam = lam_sampler() if lam_sampler is not None else None
    try:
        fallback_sample = base_box_sampler(source_images.size(), lam=lam)
    except TypeError:
        fallback_sample = base_box_sampler(source_images.size())
    fallback_boxes = _boxes_from_sampler_output(fallback_sample, device=source_images.device)
    batch, height, width = source_labels.shape
    selected_boxes = fallback_boxes.clone()
    component_records = []
    selected_records = []
    fallback_count = 0

    for b in range(batch):
        components = connected_components_from_gt(
            source_labels[b],
            ignore_label=ignore_index,
            foreground_only=foreground_only,
            connectivity=connectivity,
            min_component_area=min_component_area,
            max_component_area=max_component_area,
        )
        records = []
        for component in components:
            mask = torch.as_tensor(component["mask"], device=saliency.device, dtype=torch.bool)
            sal_score = float(saliency[b][mask].mean().item()) if mask.any() else 0.0
            denom = max(float(max_component_area - min_component_area), eps)
            area_score = float(np.clip((component["area"] - min_component_area) / denom, 0.0, 1.0))
            score = sal_score * area_score
            records.append({**component, "saliency_score": sal_score, "area_score": area_score, "score": score})

        component_records.append(records)
        if not records:
            fallback_count += 1
            selected_records.append(None)
            continue

        score_tensor = torch.tensor([[rec["score"] for rec in records]], device=source_images.device, dtype=torch.float32)
        selected_idx, probs = sample_box_by_softmax(score_tensor, temperature=temperature, eps=eps)
        idx = int(selected_idx.item())
        selected = records[idx]
        box = expand_box(bbox_from_mask(selected["mask"]), (height, width), expand_ratio=box_expand_ratio)
        selected_boxes[b] = torch.tensor(box, device=source_images.device, dtype=torch.long)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        entropy = entropy / math.log(max(len(records), 2))
        selected_records.append(
            {
                **selected,
                "box": box,
                "entropy": float(entropy.item()),
                "prob_selected": float(probs[0, idx].item()),
                "prob_max": float(probs.max(dim=1).values.item()),
            }
        )

    stats = _component_stats(component_records, selected_records, fallback_count, batch)
    return selected_boxes.detach(), stats


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
