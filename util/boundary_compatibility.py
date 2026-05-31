import math

import torch
import torch.nn.functional as F


def _as_bhw_mask(mask):
    if mask.dim() == 4:
        if mask.size(1) != 1:
            raise ValueError("mix_mask with 4 dims must have shape [B,1,H,W]")
        mask = mask[:, 0]
    if mask.dim() != 3:
        raise ValueError("mix_mask must have shape [B,H,W] or [B,1,H,W]")
    return mask.float()


def _boundary_bands(mask, band_width):
    mask = _as_bhw_mask(mask).unsqueeze(1).clamp(0.0, 1.0)
    kernel_size = 2 * int(band_width) + 1
    pad = int(band_width)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    inner = (mask - eroded.clamp(0.0, 1.0)).clamp(0.0, 1.0)[:, 0] > 0.5
    outer = (dilated.clamp(0.0, 1.0) - mask).clamp(0.0, 1.0)[:, 0] > 0.5
    return inner, outer


def _resize_feature(feature, size):
    if feature.shape[-2:] == size:
        return feature
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=True)


def _one_hot(labels, num_classes, ignore_index):
    safe = labels.detach().clone()
    safe[(safe == int(ignore_index)) | (safe < 0) | (safe >= int(num_classes))] = 0
    one_hot = F.one_hot(safe.long(), num_classes=int(num_classes)).permute(0, 3, 1, 2).float()
    return one_hot


def _gather_bchw(values, batch_idx, y_idx, x_idx):
    return values[batch_idx, :, y_idx, x_idx]


def _gather_bhw(values, batch_idx, y_idx, x_idx):
    return values[batch_idx, y_idx, x_idx]


def _empty_stats(loss, pair_radius, band_width):
    return {
        "num_pairs_per_image": 0.0,
        "same_pairs_ratio": 0.0,
        "diff_pairs_ratio": 0.0,
        "uncertain_pairs_ratio": 0.0,
        "mean_s_sem": float("nan"),
        "mean_s_F": float("nan"),
        "mean_r_ab": float("nan"),
        "mean_JS": float("nan"),
        "L_BCR": float(loss.detach().item()),
        "pair_radius": int(pair_radius),
        "band_width": int(band_width),
    }


def compute_js_boundary_compatibility_loss(
    features,
    mixed_target,
    teacher_probs,
    confidence,
    mix_mask,
    *,
    component_weight_map_or_none=None,
    num_classes,
    ignore_index=255,
    band_width=3,
    pair_mode="radius",
    pair_radius=1,
    topk=5,
    max_pairs_per_image=2048,
    tau_same=0.8,
    tau_diff=0.3,
    margin=0.4,
    use_confidence_gate=True,
    use_component_gate=False,
    detach_teacher_distribution=True,
    detach_gate=True,
    eps=1e-6,
):
    """Compute standalone V3 JS Boundary Compatibility loss.

    M=1 denotes labeled/GT source pixels and M=0 denotes pseudo target pixels.
    In standalone V3, leave component_weight_map_or_none=None and
    use_component_gate=False; this makes q_C equal to 1 for all pairs.
    """
    if pair_mode != "radius":
        raise ValueError("boundary_compatibility currently supports pair_mode='radius' only")
    if int(max_pairs_per_image) <= 0:
        raise ValueError("max_pairs_per_image must be positive")
    if use_component_gate and component_weight_map_or_none is None:
        raise ValueError("use_component_gate=True requires component_weight_map_or_none")

    target = mixed_target.detach()
    mask = _as_bhw_mask(mix_mask).to(device=features.device)
    confidence = confidence.detach().to(device=features.device, dtype=torch.float32)
    if confidence.dim() == 4:
        confidence = confidence[:, 0]
    if target.shape != mask.shape or target.shape != confidence.shape:
        raise ValueError("mixed_target, confidence, and mix_mask must have shape [B,H,W]")

    features = _resize_feature(features, target.shape[-2:])
    features = F.normalize(features, p=2, dim=1, eps=eps)

    one_hot = _one_hot(target.to(device=features.device), num_classes, ignore_index)
    if teacher_probs is None:
        probs = one_hot
    else:
        probs = teacher_probs.to(device=features.device, dtype=torch.float32)
        if probs.shape[-2:] != target.shape[-2:]:
            probs = F.interpolate(probs, size=target.shape[-2:], mode="bilinear", align_corners=True)
        if detach_teacher_distribution:
            probs = probs.detach()

    source_side = mask >= 0.5
    semantic_probs = torch.where(source_side.unsqueeze(1), one_hot, probs)

    if use_confidence_gate:
        gate_map = torch.where(source_side, torch.ones_like(confidence), confidence)
    else:
        gate_map = torch.ones_like(confidence)
    if use_component_gate:
        component_gate = component_weight_map_or_none.to(device=features.device, dtype=torch.float32)
        if component_gate.shape != gate_map.shape:
            raise ValueError("component_weight_map_or_none must have shape [B,H,W]")
        gate_map = gate_map * component_gate
    if detach_gate:
        gate_map = gate_map.detach()

    valid = target.to(device=features.device).ne(int(ignore_index))
    inner, outer = _boundary_bands(mask, band_width)
    inner = inner & valid & source_side
    outer = outer & valid & (~source_side)

    losses = []
    s_sem_all = []
    s_f_all = []
    r_all = []
    js_all = []
    same_count = 0
    diff_count = 0
    uncertain_count = 0
    total_pairs = 0
    per_image_counts = []
    radius = int(pair_radius)
    max_pairs = int(max_pairs_per_image)

    for b in range(target.shape[0]):
        inner_yx = inner[b].nonzero(as_tuple=False)
        outer_mask = outer[b]
        sample_pairs = []
        for yx in inner_yx:
            y = int(yx[0].item())
            x = int(yx[1].item())
            y1 = max(0, y - radius)
            y2 = min(target.shape[-2] - 1, y + radius)
            x1 = max(0, x - radius)
            x2 = min(target.shape[-1] - 1, x + radius)
            local = outer_mask[y1 : y2 + 1, x1 : x2 + 1].nonzero(as_tuple=False)
            for local_yx in local:
                yy = int(local_yx[0].item()) + y1
                xx = int(local_yx[1].item()) + x1
                if (yy - y) * (yy - y) + (xx - x) * (xx - x) > radius * radius:
                    continue
                sample_pairs.append((y, x, yy, xx))

        if len(sample_pairs) > max_pairs:
            keep = torch.linspace(0, len(sample_pairs) - 1, steps=max_pairs, device=features.device).long().cpu().tolist()
            sample_pairs = [sample_pairs[i] for i in keep]
        per_image_counts.append(len(sample_pairs))
        if not sample_pairs:
            continue

        pairs = torch.tensor(sample_pairs, device=features.device, dtype=torch.long)
        batch_idx = torch.full((pairs.size(0),), b, device=features.device, dtype=torch.long)
        ay, ax, by, bx = pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3]

        p_a = _gather_bchw(semantic_probs, batch_idx, ay, ax).clamp_min(float(eps))
        p_b = _gather_bchw(semantic_probs, batch_idx, by, bx).clamp_min(float(eps))
        p_a = p_a / p_a.sum(dim=1, keepdim=True).clamp_min(float(eps))
        p_b = p_b / p_b.sum(dim=1, keepdim=True).clamp_min(float(eps))
        m = (0.5 * (p_a + p_b)).clamp_min(float(eps))
        js = 0.5 * (p_a * (p_a / m).log()).sum(dim=1) + 0.5 * (p_b * (p_b / m).log()).sum(dim=1)
        js_norm = (js / math.log(2.0)).clamp(0.0, 1.0)
        s_sem = 1.0 - js_norm

        f_a = _gather_bchw(features, batch_idx, ay, ax)
        f_b = _gather_bchw(features, batch_idx, by, bx)
        s_f = ((f_a * f_b).sum(dim=1).clamp(-1.0, 1.0) + 1.0) * 0.5
        r_ab = _gather_bhw(gate_map, batch_idx, ay, ax) * _gather_bhw(gate_map, batch_idx, by, bx)
        if detach_gate:
            r_ab = r_ab.detach()

        same = s_sem > float(tau_same)
        diff = s_sem < float(tau_diff)
        active = same | diff
        uncertain = ~active
        same_count += int(same.sum().detach().item())
        diff_count += int(diff.sum().detach().item())
        uncertain_count += int(uncertain.sum().detach().item())
        total_pairs += int(pairs.size(0))

        pair_loss = torch.zeros_like(s_f)
        pair_loss[same] = (1.0 - s_f[same]).pow(2)
        pair_loss[diff] = F.relu(s_f[diff] - float(margin)).pow(2)
        if active.any():
            losses.append((pair_loss[active] * r_ab[active]).sum())
            r_all.append(r_ab[active])
        s_sem_all.append(s_sem.detach())
        s_f_all.append(s_f.detach())
        js_all.append(js_norm.detach())

    zero = features.sum() * 0.0
    if not losses:
        if total_pairs > 0 and s_sem_all:
            s_sem_cat = torch.cat(s_sem_all)
            s_f_cat = torch.cat(s_f_all)
            js_cat = torch.cat(js_all)
            stats = {
                "num_pairs_per_image": float(sum(per_image_counts) / max(len(per_image_counts), 1)),
                "same_pairs_ratio": float(same_count / max(total_pairs, 1)),
                "diff_pairs_ratio": float(diff_count / max(total_pairs, 1)),
                "uncertain_pairs_ratio": float(uncertain_count / max(total_pairs, 1)),
                "mean_s_sem": float(s_sem_cat.mean().item()),
                "mean_s_F": float(s_f_cat.mean().item()),
                "mean_r_ab": float("nan"),
                "mean_JS": float(js_cat.mean().item()),
                "L_BCR": float(zero.detach().item()),
                "pair_radius": int(pair_radius),
                "band_width": int(band_width),
            }
            return zero, stats
        return zero, _empty_stats(zero, pair_radius, band_width)

    numerator = torch.stack(losses).sum()
    denominator = torch.cat(r_all).sum().clamp_min(float(eps))
    loss = numerator / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("boundary_compatibility produced non-finite loss")

    s_sem_cat = torch.cat(s_sem_all)
    s_f_cat = torch.cat(s_f_all)
    js_cat = torch.cat(js_all)
    r_cat = torch.cat(r_all) if r_all else torch.empty(0, device=features.device)
    stats = {
        "num_pairs_per_image": float(sum(per_image_counts) / max(len(per_image_counts), 1)),
        "same_pairs_ratio": float(same_count / max(total_pairs, 1)),
        "diff_pairs_ratio": float(diff_count / max(total_pairs, 1)),
        "uncertain_pairs_ratio": float(uncertain_count / max(total_pairs, 1)),
        "mean_s_sem": float(s_sem_cat.mean().item()),
        "mean_s_F": float(s_f_cat.mean().item()),
        "mean_r_ab": float(r_cat.mean().item()) if r_cat.numel() > 0 else float("nan"),
        "mean_JS": float(js_cat.mean().item()),
        "L_BCR": float(loss.detach().item()),
        "pair_radius": int(pair_radius),
        "band_width": int(band_width),
    }
    return loss, stats
