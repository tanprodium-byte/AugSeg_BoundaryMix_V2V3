import numpy as np
import torch
import scipy.ndimage as nd


def _stat(values):
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _connectivity_structure(connectivity):
    if int(connectivity) != 8:
        raise ValueError("boundary_component currently supports connectivity=8 only")
    return np.ones((3, 3), dtype=np.int8)


@torch.no_grad()
def compute_component_weights(
    target,
    confidence,
    mix_mask,
    *,
    ignore_index=255,
    num_classes=None,
    connectivity=8,
    apply_to="target_only",
    foreground_only=True,
    tau_visible_low=0.2,
    tau_visible_high=0.6,
    area_min=64,
    area_max=512,
    use_mean_confidence_in_q=True,
    base_pixel_weight="one",
    weight_mode="soft",
    force_q_one=False,
    eps=1e-6,
):
    """Build detached V2 component weights for mixed CE.

    M=1 denotes labeled-source pixels. For target-only V2, affected connected
    components are found on the target pseudo-label map and only their visible
    target-side pixels, C intersect (1-M), receive the soft q_C multiplier.
    """
    if apply_to != "target_only":
        raise ValueError("boundary_component currently supports apply_to='target_only' only")
    if weight_mode != "soft":
        raise ValueError("boundary_component currently supports weight_mode='soft' only")
    if base_pixel_weight not in ("one", "confidence"):
        raise ValueError("base_pixel_weight must be 'one' or 'confidence'")

    labels = target.detach()
    conf = confidence.detach().to(device=labels.device, dtype=torch.float32)
    mask = mix_mask.detach().to(device=labels.device, dtype=torch.float32)
    if mask.dim() == 4:
        if mask.size(1) != 1:
            raise ValueError("mix_mask with 4 dims must have shape [B,1,H,W]")
        mask = mask[:, 0]
    if labels.shape != conf.shape or labels.shape != mask.shape:
        raise ValueError("target, confidence, and mix_mask must have shape [B,H,W]")

    if base_pixel_weight == "confidence":
        weight = conf.clone()
    else:
        weight = torch.ones_like(conf)

    source_side = mask >= 0.5
    target_side = ~source_side
    weight[source_side] = 1.0

    tau_span = max(float(tau_visible_high) - float(tau_visible_low), eps)
    area_span = max(float(area_max) - float(area_min), eps)
    structure = _connectivity_structure(connectivity)

    labels_np = labels.cpu().numpy()
    conf_np = conf.cpu().numpy()
    source_np = source_side.cpu().numpy()
    target_np = target_side.cpu().numpy()

    affected_component_count = 0
    affected_pixel_count = 0
    visible_ratios = []
    visible_areas = []
    mean_confidences = []
    q_values = []
    per_class_count = {}
    per_class_q_sum = {}

    for b in range(labels_np.shape[0]):
        sample_labels = labels_np[b]
        sample_conf = conf_np[b]
        sample_source = source_np[b]
        sample_target = target_np[b]

        classes = np.unique(sample_labels)
        for cls in classes:
            cls_int = int(cls)
            if cls_int == int(ignore_index):
                continue
            if foreground_only and cls_int == 0:
                continue
            if num_classes is not None and (cls_int < 0 or cls_int >= int(num_classes)):
                continue

            class_mask = sample_labels == cls_int
            cc, num_cc = nd.label(class_mask, structure=structure)
            for comp_id in range(1, num_cc + 1):
                comp = cc == comp_id
                if not comp.any():
                    continue

                comp_source = comp & sample_source
                comp_target = comp & sample_target
                if not comp_source.any() or not comp_target.any():
                    continue

                comp_area = int(comp.sum())
                vis_area = int(comp_target.sum())
                visible_ratio = float(vis_area) / float(max(comp_area, 1))
                mean_conf = float(sample_conf[comp_target].mean()) if vis_area > 0 else 0.0

                if force_q_one:
                    q_c = 1.0
                else:
                    visible_score = (visible_ratio - float(tau_visible_low)) / tau_span
                    visible_score = float(np.clip(visible_score, 0.0, 1.0))
                    area_score = (float(vis_area) - float(area_min)) / area_span
                    area_score = float(np.clip(area_score, 0.0, 1.0))
                    conf_score = mean_conf if use_mean_confidence_in_q else 1.0
                    q_c = float(np.clip(conf_score * visible_score * area_score, 0.0, 1.0))

                affected_component_count += 1
                affected_pixel_count += vis_area
                visible_ratios.append(visible_ratio)
                visible_areas.append(float(vis_area))
                mean_confidences.append(mean_conf)
                q_values.append(q_c)
                per_class_count[cls_int] = per_class_count.get(cls_int, 0) + 1
                per_class_q_sum[cls_int] = per_class_q_sum.get(cls_int, 0.0) + q_c

                comp_target_t = torch.from_numpy(comp_target).to(device=weight.device, dtype=torch.bool)
                weight[b] = torch.where(comp_target_t, weight[b] * weight.new_tensor(q_c), weight[b])

    visible_ratio_stats = _stat(visible_ratios)
    visible_area_stats = _stat(visible_areas)
    q_stats = _stat(q_values)
    per_class_mean_q = {
        int(cls): float(per_class_q_sum[cls] / max(per_class_count[cls], 1))
        for cls in per_class_count
    }

    stats = {
        "num_affected_components": int(affected_component_count),
        "num_affected_pixels": int(affected_pixel_count),
        "visible_ratio_mean": visible_ratio_stats["mean"],
        "visible_ratio_std": visible_ratio_stats["std"],
        "visible_ratio_min": visible_ratio_stats["min"],
        "visible_ratio_max": visible_ratio_stats["max"],
        "visible_area_mean": visible_area_stats["mean"],
        "visible_area_std": visible_area_stats["std"],
        "visible_area_min": visible_area_stats["min"],
        "visible_area_max": visible_area_stats["max"],
        "mean_confidence_R_C": float(np.mean(mean_confidences)) if mean_confidences else float("nan"),
        "q_C_mean": q_stats["mean"],
        "q_C_std": q_stats["std"],
        "q_C_min": q_stats["min"],
        "q_C_max": q_stats["max"],
        "per_class_affected_component_count": {int(k): int(v) for k, v in per_class_count.items()},
        "per_class_mean_q_C": per_class_mean_q,
    }

    if not torch.isfinite(weight).all():
        raise FloatingPointError("boundary_component produced non-finite weights")

    return weight.detach(), stats
