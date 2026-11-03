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
    return F.one_hot(safe.long(), num_classes=int(num_classes)).permute(0, 3, 1, 2).float()


def _gather_bchw(values, batch_idx, y_idx, x_idx):
    return values[batch_idx, :, y_idx, x_idx]


def _gather_bhw(values, batch_idx, y_idx, x_idx):
    return values[batch_idx, y_idx, x_idx]


def _mean_or_nan(values):
    if values is None or values.numel() == 0:
        return float("nan")
    return float(values.detach().mean().item())


def _sum_or_zero(values):
    if not values:
        return None
    return torch.cat(values)


def _cfg_value(cfg, key, default):
    if cfg is None:
        return default
    return cfg.get(key, default)


def _as_confidence_map(confidence_or_logits, target_shape, device, eps):
    value = confidence_or_logits.detach().to(device=device, dtype=torch.float32)
    if value.dim() == 3:
        confidence = value
    elif value.dim() == 4:
        if value.size(1) == 1:
            confidence = value[:, 0]
        elif (
            float(value.detach().amin().item()) >= 0.0
            and float(value.detach().amax().item()) <= 1.0 + float(eps)
            and torch.allclose(
                value.detach().sum(dim=1),
                torch.ones_like(value[:, 0]),
                atol=1e-3,
                rtol=1e-3,
            )
        ):
            confidence = value.clamp(0.0, 1.0).max(dim=1).values
        else:
            confidence = F.softmax(value, dim=1).max(dim=1).values
    else:
        raise ValueError("confidence_or_logits must have shape [B,H,W], [B,1,H,W], or [B,C,H,W]")
    if confidence.shape != target_shape:
        raise ValueError("confidence_or_logits must align to mixed_target spatial shape")
    return confidence


def _base_stats(zero, pair_radius, band_width):
    return {
        "bcr/num_pairs_candidate": 0,
        "bcr/num_pairs_sampled": 0,
        "bcr/num_pairs_active": 0,
        "bcr/num_pairs_same": 0,
        "bcr/num_pairs_diff": 0,
        "bcr/num_pairs_uncertain": 0,
        "bcr/mean_s_sem": float("nan"),
        "bcr/mean_s_sem_same": float("nan"),
        "bcr/mean_s_sem_diff": float("nan"),
        "bcr/mean_s_S": float("nan"),
        "bcr/mean_s_S_same": float("nan"),
        "bcr/mean_s_S_diff": float("nan"),
        "bcr/mean_r_ab": float("nan"),
        "bcr/loss_bcr": float(zero.detach().item()),
        "bcr/loss_same": 0.0,
        "bcr/loss_diff": 0.0,
        "bcr/use_component_gate": 0.0,
        "bcr/component_gate_mode": "none",
        "bcr/mean_q_pair_a": float("nan"),
        "bcr/mean_q_pair_b": float("nan"),
        "bcr/mean_q_pair_product": float("nan"),
        "bcr/mean_r_before_component_gate": float("nan"),
        "bcr/mean_r_after_component_gate": float("nan"),
        "pair_radius": int(pair_radius),
        "band_width": int(band_width),
        "num_pairs_per_image": 0.0,
        "same_pairs_ratio": 0.0,
        "diff_pairs_ratio": 0.0,
        "uncertain_pairs_ratio": 0.0,
        "mean_s_sem": float("nan"),
        "mean_s_F": float("nan"),
        "mean_r_ab": float("nan"),
        "mean_JS": float("nan"),
        "L_BCR": float(zero.detach().item()),
    }


def compute_js_boundary_compatibility_loss(
    features,
    mixed_target,
    teacher_probs,
    confidence,
    mix_mask,
    component_weight_map_or_none=None,
    teacher_features=None,
    cfg=None,
    *,
    num_classes=None,
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
    component_gate_mode="direct",
    component_gate_alpha=0.5,
    component_gate_threshold=0.1,
    same_loss_mode="hard_one",
    relation_mode="base_margin",
    use_teacher_features=False,
    teacher_feature_detach=True,
    affinity_target="hard",
    affinity_temperature=0.2,
    detach_teacher_distribution=True,
    detach_gate=True,
    eps=1e-6,
):
    """Compute raw Boundary Compatibility Regularization loss.

    The fourth argument may be either a detached confidence/max-probability map
    shaped [B,H,W] / [B,1,H,W], or teacher logits/probabilities shaped
    [B,C,H,W]. In the 4D class-channel case the confidence gate is the max
    probability, computed with softmax for raw logits.

    The returned loss is not multiplied by lambda_bcr. The training loop owns
    lambda scaling so disabled or lambda_bcr=0 configs stay baseline-neutral.
    """
    if cfg is not None:
        band_width = _cfg_value(cfg, "band_width", band_width)
        pair_mode = _cfg_value(cfg, "pair_mode", pair_mode)
        pair_radius = _cfg_value(cfg, "pair_radius", pair_radius)
        topk = _cfg_value(cfg, "topk", topk)
        max_pairs_per_image = _cfg_value(cfg, "max_pairs_per_image", max_pairs_per_image)
        tau_same = _cfg_value(cfg, "tau_same", tau_same)
        tau_diff = _cfg_value(cfg, "tau_diff", tau_diff)
        margin = _cfg_value(cfg, "margin", margin)
        use_confidence_gate = _cfg_value(cfg, "use_confidence_gate", use_confidence_gate)
        use_component_gate = _cfg_value(cfg, "use_component_gate", use_component_gate)
        component_gate_mode = _cfg_value(cfg, "component_gate_mode", component_gate_mode)
        component_gate_alpha = _cfg_value(cfg, "component_gate_alpha", component_gate_alpha)
        component_gate_threshold = _cfg_value(cfg, "component_gate_threshold", component_gate_threshold)
        same_loss_mode = _cfg_value(cfg, "same_loss_mode", same_loss_mode)
        relation_mode = _cfg_value(cfg, "relation_mode", relation_mode)
        use_teacher_features = _cfg_value(cfg, "use_teacher_features", use_teacher_features)
        teacher_feature_detach = _cfg_value(cfg, "teacher_feature_detach", teacher_feature_detach)
        affinity_target = _cfg_value(cfg, "affinity_target", affinity_target)
        affinity_temperature = _cfg_value(cfg, "affinity_temperature", affinity_temperature)
        detach_teacher_distribution = _cfg_value(cfg, "detach_teacher_distribution", detach_teacher_distribution)
        detach_gate = _cfg_value(cfg, "detach_gate", detach_gate)
        eps = _cfg_value(cfg, "eps", eps)

    if pair_mode != "radius":
        raise ValueError("boundary_compatibility currently supports pair_mode='radius' only")
    if int(max_pairs_per_image) <= 0:
        raise ValueError("max_pairs_per_image must be positive")
    if num_classes is None:
        raise ValueError("num_classes is required")

    relation_mode = str(relation_mode)
    same_loss_mode = str(same_loss_mode)
    component_gate_mode = str(component_gate_mode)
    affinity_target = str(affinity_target)
    teacher_required = relation_mode in ("teacher_feature_gate", "teacher_relation_consistency") or bool(use_teacher_features)
    if teacher_required and teacher_features is None:
        raise ValueError("%s requires teacher_features" % relation_mode)
    if use_component_gate and component_gate_mode not in ("direct", "soft", "filter", "none"):
        raise ValueError("component_gate_mode must be one of direct, soft, filter, none")
    if use_component_gate and component_gate_mode != "none" and component_weight_map_or_none is None:
        raise ValueError("use_component_gate=True requires component_weight_map_or_none")

    target = mixed_target.detach().to(device=features.device)
    mask = _as_bhw_mask(mix_mask.detach()).to(device=features.device)
    confidence = _as_confidence_map(confidence, target.shape, features.device, eps)
    if target.shape != mask.shape:
        raise ValueError("mixed_target and mix_mask must have shape [B,H,W]")

    features = _resize_feature(features, target.shape[-2:])
    features_norm = F.normalize(features, p=2, dim=1, eps=float(eps))

    teacher_norm = None
    if teacher_features is not None:
        teacher_features = _resize_feature(teacher_features, target.shape[-2:])
        if teacher_feature_detach:
            teacher_features = teacher_features.detach()
        teacher_norm = F.normalize(teacher_features, p=2, dim=1, eps=float(eps))

    one_hot = _one_hot(target, num_classes, ignore_index).to(device=features.device)
    if teacher_probs is None:
        probs = one_hot
    else:
        probs = teacher_probs.detach().to(device=features.device, dtype=torch.float32)
        if probs.shape[-2:] != target.shape[-2:]:
            probs = F.interpolate(probs, size=target.shape[-2:], mode="bilinear", align_corners=True)
        if detach_teacher_distribution:
            probs = probs.detach()

    source_side = mask >= 0.5
    semantic_probs = torch.where(source_side.unsqueeze(1), one_hot, probs).detach()

    if use_confidence_gate:
        gate_map = torch.where(source_side, torch.ones_like(confidence), confidence)
    else:
        gate_map = torch.ones_like(confidence)
    if detach_gate:
        gate_map = gate_map.detach()

    component_gate = None
    if use_component_gate and component_gate_mode != "none":
        component_gate = component_weight_map_or_none.detach().to(device=features.device, dtype=torch.float32)
        if component_gate.shape != gate_map.shape:
            raise ValueError("component_weight_map_or_none must have shape [B,H,W]")

    valid = target.ne(int(ignore_index))
    inner, outer = _boundary_bands(mask, band_width)
    inner = inner & valid & source_side
    outer = outer & valid & (~source_side)

    losses = []
    same_losses = []
    diff_losses = []
    affinity_losses = []
    r_active_all = []
    r_before_all = []
    r_after_all = []
    q_a_all = []
    q_b_all = []
    s_sem_all = []
    s_s_all = []
    s_t_all = []
    a_all = []
    a_same_all = []
    a_diff_all = []
    y_all = []
    same_s_sem_all = []
    diff_s_sem_all = []
    same_s_s_all = []
    diff_s_s_all = []
    same_s_t_all = []
    diff_s_t_all = []
    same_gate_t_all = []
    diff_gate_t_all = []
    soft_same_abs_all = []
    teacher_rel_abs_all = []

    num_candidate = 0
    num_sampled = 0
    num_same = 0
    num_diff = 0
    num_uncertain = 0
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

        num_candidate += len(sample_pairs)
        if len(sample_pairs) > max_pairs:
            keep = torch.linspace(0, len(sample_pairs) - 1, steps=max_pairs, device=features.device).long().cpu().tolist()
            sample_pairs = [sample_pairs[i] for i in keep]
        per_image_counts.append(len(sample_pairs))
        num_sampled += len(sample_pairs)
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
        s_sem = (1.0 - js_norm).detach()

        f_a = _gather_bchw(features_norm, batch_idx, ay, ax)
        f_b = _gather_bchw(features_norm, batch_idx, by, bx)
        cos_raw = (f_a * f_b).sum(dim=1).clamp(-1.0, 1.0)
        s_s = (cos_raw + 1.0) * 0.5

        s_t = None
        if teacher_norm is not None:
            t_a = _gather_bchw(teacher_norm, batch_idx, ay, ax)
            t_b = _gather_bchw(teacher_norm, batch_idx, by, bx)
            s_t = (((t_a * t_b).sum(dim=1).clamp(-1.0, 1.0) + 1.0) * 0.5).detach()

        r_ab = _gather_bhw(gate_map, batch_idx, ay, ax) * _gather_bhw(gate_map, batch_idx, by, bx)
        if detach_gate:
            r_ab = r_ab.detach()
        r_before = r_ab.detach()
        r_after = r_ab

        same = s_sem > float(tau_same)
        diff = s_sem < float(tau_diff)
        active = same | diff

        if use_component_gate and component_gate is not None and component_gate_mode != "none":
            q_a = _gather_bhw(component_gate, batch_idx, ay, ax).detach()
            q_b = _gather_bhw(component_gate, batch_idx, by, bx).detach()
            q_a_all.append(q_a)
            q_b_all.append(q_b)
            if component_gate_mode == "direct":
                r_after = r_after * q_a * q_b
            elif component_gate_mode == "soft":
                alpha = float(component_gate_alpha)
                q_tilde_a = alpha + (1.0 - alpha) * q_a
                q_tilde_b = alpha + (1.0 - alpha) * q_b
                r_after = r_after * q_tilde_a * q_tilde_b
            elif component_gate_mode == "filter":
                active = active & (q_a >= float(component_gate_threshold)) & (q_b >= float(component_gate_threshold))
        if detach_gate:
            r_after = r_after.detach()

        uncertain = ~(same | diff)
        num_same += int(same.sum().detach().item())
        num_diff += int(diff.sum().detach().item())
        num_uncertain += int(uncertain.sum().detach().item())

        s_sem_all.append(s_sem)
        s_s_all.append(s_s.detach())
        r_before_all.append(r_before.detach())
        r_after_all.append(r_after.detach())
        if same.any():
            same_s_sem_all.append(s_sem[same])
            same_s_s_all.append(s_s[same].detach())
        if diff.any():
            diff_s_sem_all.append(s_sem[diff])
            diff_s_s_all.append(s_s[diff].detach())
        if s_t is not None:
            s_t_all.append(s_t)
            if same.any():
                same_s_t_all.append(s_t[same])
            if diff.any():
                diff_s_t_all.append(s_t[diff])

        if not active.any():
            continue

        if relation_mode == "base_margin":
            pair_loss = torch.zeros_like(s_s)
            if same_loss_mode == "hard_one":
                pair_loss[same] = (1.0 - s_s[same]).pow(2)
            elif same_loss_mode == "soft_semantic":
                same_loss = (s_sem[same] - s_s[same]).pow(2)
                pair_loss[same] = same_loss
                if same.any():
                    soft_same_abs_all.append((s_s[same].detach() - s_sem[same]).abs())
            else:
                raise ValueError("same_loss_mode must be hard_one or soft_semantic")
            pair_loss[diff] = F.relu(s_s[diff] - float(margin)).pow(2)
            losses.append((pair_loss[active] * r_after[active]).sum())
            r_active_all.append(r_after[active])
            if same.any():
                same_losses.append(pair_loss[same].detach())
            if diff.any():
                diff_losses.append(pair_loss[diff].detach())

        elif relation_mode == "teacher_feature_gate":
            if s_t is None:
                raise ValueError("teacher_feature_gate requires teacher_features")
            pair_loss = torch.zeros_like(s_s)
            pair_loss[same] = (1.0 - s_s[same]).pow(2)
            pair_loss[diff] = F.relu(s_s[diff] - float(margin)).pow(2)
            gate = torch.zeros_like(s_s)
            gate[same] = r_after[same] * s_t[same]
            gate[diff] = r_after[diff] * (1.0 - s_t[diff])
            losses.append((pair_loss[active] * gate[active]).sum())
            r_active_all.append(gate[active])
            if same.any():
                same_losses.append(pair_loss[same].detach())
                same_gate_t_all.append(gate[same].detach())
            if diff.any():
                diff_losses.append(pair_loss[diff].detach())
                diff_gate_t_all.append(gate[diff].detach())

        elif relation_mode == "teacher_relation_consistency":
            if s_t is None:
                raise ValueError("teacher_relation_consistency requires teacher_features")
            loss_rel = (s_s[active] - s_t[active]).pow(2)
            losses.append((loss_rel * r_after[active]).sum())
            r_active_all.append(r_after[active])
            teacher_rel_abs_all.append((s_s[active].detach() - s_t[active]).abs())

        elif relation_mode == "affinity_bce":
            temperature = max(float(affinity_temperature), float(eps))
            affinity = torch.sigmoid(cos_raw / temperature).clamp(float(eps), 1.0 - float(eps))
            if affinity_target == "hard":
                y_rel = torch.zeros_like(affinity)
                y_rel[same] = 1.0
                y_rel[diff] = 0.0
            elif affinity_target == "soft":
                y_rel = s_sem.detach()
            else:
                raise ValueError("affinity_target must be hard or soft")
            bce = F.binary_cross_entropy(affinity, y_rel, reduction="none")
            losses.append((bce[active] * r_after[active]).sum())
            r_active_all.append(r_after[active])
            affinity_losses.append(bce[active].detach())
            a_all.append(affinity[active].detach())
            if same.any():
                a_same_all.append(affinity[same].detach())
            if diff.any():
                a_diff_all.append(affinity[diff].detach())
            y_all.append(y_rel[active].detach())
        else:
            raise ValueError("unsupported relation_mode: %s" % relation_mode)

    zero = features.sum() * 0.0
    stats = _base_stats(zero, pair_radius, band_width)
    stats["bcr/num_pairs_candidate"] = int(num_candidate)
    stats["bcr/num_pairs_sampled"] = int(num_sampled)
    stats["bcr/num_pairs_same"] = int(num_same)
    stats["bcr/num_pairs_diff"] = int(num_diff)
    stats["bcr/num_pairs_uncertain"] = int(num_uncertain)
    stats["bcr/use_component_gate"] = float(bool(use_component_gate and component_gate_mode != "none"))
    stats["bcr/component_gate_mode"] = component_gate_mode if use_component_gate else "none"
    stats["pair_radius"] = int(pair_radius)
    stats["band_width"] = int(band_width)
    stats["num_pairs_per_image"] = float(sum(per_image_counts) / max(len(per_image_counts), 1))
    stats["same_pairs_ratio"] = float(num_same / max(num_sampled, 1))
    stats["diff_pairs_ratio"] = float(num_diff / max(num_sampled, 1))
    stats["uncertain_pairs_ratio"] = float(num_uncertain / max(num_sampled, 1))

    s_sem_cat = _sum_or_zero(s_sem_all)
    s_s_cat = _sum_or_zero(s_s_all)
    r_before_cat = _sum_or_zero(r_before_all)
    r_after_cat = _sum_or_zero(r_after_all)
    q_a_cat = _sum_or_zero(q_a_all)
    q_b_cat = _sum_or_zero(q_b_all)
    same_s_sem_cat = _sum_or_zero(same_s_sem_all)
    diff_s_sem_cat = _sum_or_zero(diff_s_sem_all)
    same_s_s_cat = _sum_or_zero(same_s_s_all)
    diff_s_s_cat = _sum_or_zero(diff_s_s_all)

    stats["bcr/mean_s_sem"] = _mean_or_nan(s_sem_cat)
    stats["bcr/mean_s_sem_same"] = _mean_or_nan(same_s_sem_cat)
    stats["bcr/mean_s_sem_diff"] = _mean_or_nan(diff_s_sem_cat)
    stats["bcr/mean_s_S"] = _mean_or_nan(s_s_cat)
    stats["bcr/mean_s_S_same"] = _mean_or_nan(same_s_s_cat)
    stats["bcr/mean_s_S_diff"] = _mean_or_nan(diff_s_s_cat)
    stats["bcr/mean_r_before_component_gate"] = _mean_or_nan(r_before_cat)
    stats["bcr/mean_r_after_component_gate"] = _mean_or_nan(r_after_cat)
    stats["mean_s_sem"] = stats["bcr/mean_s_sem"]
    stats["mean_s_F"] = stats["bcr/mean_s_S"]
    stats["mean_JS"] = float(1.0 - stats["bcr/mean_s_sem"]) if not math.isnan(stats["bcr/mean_s_sem"]) else float("nan")

    if q_a_cat is not None and q_b_cat is not None:
        stats["bcr/mean_q_pair_a"] = _mean_or_nan(q_a_cat)
        stats["bcr/mean_q_pair_b"] = _mean_or_nan(q_b_cat)
        stats["bcr/mean_q_pair_product"] = _mean_or_nan(q_a_cat * q_b_cat)

    if not losses:
        stats["bcr/num_pairs_active"] = 0
        stats["bcr/mean_r_ab"] = float("nan")
        stats["mean_r_ab"] = float("nan")
        return zero, stats

    r_active_cat = torch.cat(r_active_all)
    numerator = torch.stack(losses).sum()
    denominator = r_active_cat.sum().clamp_min(float(eps))
    loss = numerator / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("boundary_compatibility produced non-finite loss")

    stats["bcr/num_pairs_active"] = int(r_active_cat.numel())
    stats["bcr/mean_r_ab"] = _mean_or_nan(r_active_cat)
    stats["bcr/loss_bcr"] = float(loss.detach().item())
    stats["mean_r_ab"] = stats["bcr/mean_r_ab"]
    stats["L_BCR"] = stats["bcr/loss_bcr"]

    same_loss_cat = _sum_or_zero(same_losses)
    diff_loss_cat = _sum_or_zero(diff_losses)
    stats["bcr/loss_same"] = _mean_or_nan(same_loss_cat)
    stats["bcr/loss_diff"] = _mean_or_nan(diff_loss_cat)

    soft_same_abs_cat = _sum_or_zero(soft_same_abs_all)
    if soft_same_abs_cat is not None:
        stats["bcr/mean_abs_sS_minus_sSem_same"] = _mean_or_nan(soft_same_abs_cat)

    s_t_cat = _sum_or_zero(s_t_all)
    same_s_t_cat = _sum_or_zero(same_s_t_all)
    diff_s_t_cat = _sum_or_zero(diff_s_t_all)
    if s_t_cat is not None:
        stats["bcr/mean_s_T"] = _mean_or_nan(s_t_cat)
        stats["bcr/mean_s_T_same"] = _mean_or_nan(same_s_t_cat)
        stats["bcr/mean_s_T_diff"] = _mean_or_nan(diff_s_t_cat)
        stats["bcr/mean_teacher_feature_gate_same"] = _mean_or_nan(_sum_or_zero(same_gate_t_all))
        stats["bcr/mean_teacher_feature_gate_diff"] = _mean_or_nan(_sum_or_zero(diff_gate_t_all))

    teacher_rel_abs_cat = _sum_or_zero(teacher_rel_abs_all)
    if teacher_rel_abs_cat is not None:
        stats["bcr/mean_abs_sS_minus_sT_active"] = _mean_or_nan(teacher_rel_abs_cat)

    affinity_cat = _sum_or_zero(a_all)
    y_cat = _sum_or_zero(y_all)
    affinity_loss_cat = _sum_or_zero(affinity_losses)
    if affinity_cat is not None:
        stats["bcr/affinity_temperature"] = float(affinity_temperature)
        stats["bcr/mean_A_affinity"] = _mean_or_nan(affinity_cat)
        stats["bcr/mean_A_same"] = _mean_or_nan(_sum_or_zero(a_same_all))
        stats["bcr/mean_A_diff"] = _mean_or_nan(_sum_or_zero(a_diff_all))
        stats["bcr/mean_y_rel"] = _mean_or_nan(y_cat)
        stats["bcr/loss_affinity"] = _mean_or_nan(affinity_loss_cat)

    return loss, stats
