import numpy as np
import torch
import torch.nn.functional as F


def _as_bchw_mask(mask):
    if mask.dim() == 3:
        return mask.unsqueeze(1).float()
    if mask.dim() == 4 and mask.size(1) == 1:
        return mask.float()
    raise ValueError("mask must have shape [B,H,W] or [B,1,H,W]")


def erode_mask(mask, kernel_size=5):
    mask = _as_bchw_mask(mask)
    pad = kernel_size // 2
    return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)


def dilate_mask(mask, kernel_size=5):
    mask = _as_bchw_mask(mask)
    pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)


def get_inner_outer_boundary(mask, kernel_size=5):
    mask = _as_bchw_mask(mask).clamp(0.0, 1.0)
    eroded = erode_mask(mask, kernel_size=kernel_size).clamp(0.0, 1.0)
    dilated = dilate_mask(mask, kernel_size=kernel_size).clamp(0.0, 1.0)
    inner = (mask - eroded).clamp(0.0, 1.0)
    outer = (dilated - mask).clamp(0.0, 1.0)
    return inner, outer


def boundary_confidence_weight_map(
    mix_mask,
    confidence,
    *,
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
    eps=1e-6,
):
    mask = _as_bchw_mask(mix_mask).clamp(0.0, 1.0)
    inner, outer = get_inner_outer_boundary(mask, kernel_size=kernel_size)

    if use_confidence:
        risk = _as_bchw_mask(confidence).to(device=mask.device, dtype=mask.dtype)
    else:
        risk = torch.ones_like(mask)

    source_interior = (mask - inner).clamp(0.0, 1.0)
    target = 1.0 - mask
    target_exterior = (target - outer).clamp(0.0, 1.0)

    weight = (
        source_interior
        + float(gamma_in) * inner
        + risk * target_exterior
        + float(gamma_out) * risk * outer
    )

    if normalize_weight:
        valid = weight > 0
        mean_weight = weight[valid].mean().clamp_min(eps) if valid.any() else weight.new_tensor(1.0)
        weight = weight / mean_weight

    return weight.squeeze(1)


def _masked_mean(values, mask):
    mask = mask.to(dtype=torch.bool)
    if not mask.any():
        return float("nan")
    return float(values[mask].detach().mean().item())


def boundary_mix_debug_tensors(
    mix_mask,
    confidence,
    *,
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
):
    mask = _as_bchw_mask(mix_mask).clamp(0.0, 1.0)
    confidence = _as_bchw_mask(confidence).to(device=mask.device, dtype=mask.dtype)
    inner, outer = get_inner_outer_boundary(mask, kernel_size=kernel_size)
    weight = boundary_confidence_weight_map(
        mask,
        confidence,
        kernel_size=kernel_size,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
        use_confidence=use_confidence,
        normalize_weight=normalize_weight,
    ).unsqueeze(1)
    target_exterior = ((1.0 - mask) - outer).clamp(0.0, 1.0)

    return {
        "mask": mask,
        "inner": inner,
        "outer": outer,
        "confidence": confidence,
        "weight": weight,
        "target_exterior": target_exterior,
    }


def boundary_mix_debug_stats(
    mix_mask,
    confidence,
    *,
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
):
    tensors = boundary_mix_debug_tensors(
        mix_mask,
        confidence,
        kernel_size=kernel_size,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
        use_confidence=use_confidence,
        normalize_weight=normalize_weight,
    )
    weight = tensors["weight"]
    inner = tensors["inner"]
    outer = tensors["outer"]
    confidence = tensors["confidence"]
    target_exterior = tensors["target_exterior"]

    return {
        "weight_mean": float(weight.detach().mean().item()),
        "weight_min": float(weight.detach().min().item()),
        "weight_max": float(weight.detach().max().item()),
        "b_in_pixels": int(inner.detach().sum().item()),
        "b_out_pixels": int(outer.detach().sum().item()),
        "target_exterior_conf_mean": _masked_mean(confidence, target_exterior > 0),
        "target_outer_boundary_conf_mean": _masked_mean(confidence, outer > 0),
    }


def weighted_cross_entropy_loss(
    predict,
    target,
    weight,
    *,
    ignore_index=255,
    eps=1e-6,
):
    ce = F.cross_entropy(predict, target, ignore_index=ignore_index, reduction="none")
    valid = (target != ignore_index).to(dtype=ce.dtype)
    weight = weight.to(device=ce.device, dtype=ce.dtype) * valid
    return (ce * weight).sum() / (weight.sum() + eps)


def compute_c4_direct_mix_stats(
    mix_source_mask: torch.Tensor,
    mixed_target: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> dict[str, float]:
    """Compute C4 instrumentation from provenance without mutation or RNG."""
    pasted_mask = mix_source_mask.detach().bool()
    detached_target = mixed_target.detach()
    batch, height, width = pasted_mask.shape

    gate_attempted_count = batch
    mixed_sample_mask = pasted_mask.flatten(1).any(dim=1)
    gate_pass_count = mixed_sample_mask.sum()
    pasted_pixel_count = pasted_mask.sum()

    boxes = target_boxes.detach()
    box_height = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
    box_width = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    box_area = box_height * box_width

    valid_pasted_mask = pasted_mask & detached_target.ne(ignore_index)
    ignore_pasted_mask = pasted_mask & detached_target.eq(ignore_index)
    valid_count = valid_pasted_mask.sum()
    ignore_count = ignore_pasted_mask.sum()

    gate_attempted = float(gate_attempted_count)
    gate_pass = float(gate_pass_count.item())
    pasted_pixels = float(pasted_pixel_count.item())
    if pasted_pixels > 0:
        valid_ratio = float(valid_count.item()) / pasted_pixels
        ignore_ratio = float(ignore_count.item()) / pasted_pixels
    else:
        valid_ratio = 0.0
        ignore_ratio = 0.0

    selected_box_area_mean = float(box_area.float().mean().item())
    return {
        "c4/gate_attempted_count": gate_attempted,
        "c4/gate_pass_count": gate_pass,
        "c4/gate_pass_ratio": gate_pass / gate_attempted,
        "c4/mixed_sample_count": gate_pass,
        "c4/mixed_sample_ratio": gate_pass / gate_attempted,
        "c4/pasted_pixel_count": pasted_pixels,
        "c4/pasted_pixel_ratio": pasted_pixels / float(batch * height * width),
        "c4/selected_box_area_mean": selected_box_area_mean,
        "c4/selected_box_area_ratio_mean": selected_box_area_mean / float(height * width),
        "c4/valid_labeled_pasted_pixel_count": float(valid_count.item()),
        "c4/valid_labeled_pixel_ratio": valid_ratio,
        "c4/ignore_labeled_pasted_pixel_count": float(ignore_count.item()),
        "c4/ignore_labeled_pixel_ratio": ignore_ratio,
    }


def cut_mix_label_adaptive_c4_direct_labeled(
    unlabeled_image,
    unlabeled_mask,
    unlabeled_logits,
    labeled_image,
    labeled_mask,
    lst_reliabilities,
    target_boxes,
    *,
    unlabeled_probs=None,
    unlabeled_weight=None,
    ignore_index=255,
):
    """Directly paste labeled crops into their final C4 unlabeled targets."""
    batch = unlabeled_image.size(0)
    if labeled_image.size(0) != batch or labeled_mask.size(0) != batch:
        raise ValueError(
            "C4 requires equal labeled and unlabeled batch sizes: "
            f"unlabeled={batch}, labeled_image={labeled_image.size(0)}, labeled_mask={labeled_mask.size(0)}"
        )
    if len(lst_reliabilities) != batch:
        raise ValueError(f"C4 sample reliability must have length {batch}, got {len(lst_reliabilities)}")
    if unlabeled_image.dim() != 4 or labeled_image.dim() != 4:
        raise ValueError("C4 labeled and unlabeled images must have shape [B,C,H,W]")
    if unlabeled_mask.dim() != 3 or labeled_mask.dim() != 3 or unlabeled_logits.dim() != 3:
        raise ValueError("C4 masks and confidence metadata must have shape [B,H,W]")
    if unlabeled_image.shape[1:] != labeled_image.shape[1:]:
        raise ValueError(
            "C4 labeled/unlabeled image shapes must match after the batch dimension: "
            f"{tuple(labeled_image.shape[1:])} != {tuple(unlabeled_image.shape[1:])}"
        )
    if labeled_image.shape[2:] != labeled_mask.shape[1:]:
        raise ValueError(
            "C4 labeled image/GT spatial mismatch: "
            f"{tuple(labeled_image.shape[2:])} != {tuple(labeled_mask.shape[1:])}"
        )
    if unlabeled_image.shape[2:] != unlabeled_mask.shape[1:] or unlabeled_mask.shape != unlabeled_logits.shape:
        raise ValueError("C4 unlabeled image, pseudo-label, and confidence metadata spatial shapes must match")
    if unlabeled_weight is None:
        raise ValueError("C4 requires the official CSL pixel-weight map")
    if unlabeled_weight.shape != unlabeled_mask.shape:
        raise ValueError("C4 CSL pixel weight must have shape [B,H,W]")
    if unlabeled_probs is not None and (
        unlabeled_probs.dim() != 4
        or unlabeled_probs.size(0) != batch
        or unlabeled_probs.shape[2:] != unlabeled_mask.shape[1:]
    ):
        raise ValueError("C4 teacher probabilities must have shape [B,C,H,W]")

    target_boxes = torch.as_tensor(target_boxes, device=unlabeled_image.device, dtype=torch.long)
    if target_boxes.shape != (batch, 4):
        raise ValueError(f"C4 target_boxes must have shape [{batch},4], got {tuple(target_boxes.shape)}")

    mixed_image = unlabeled_image.clone()
    mixed_target = unlabeled_mask.clone()
    mixed_logits = unlabeled_logits.clone()
    mixed_probs = unlabeled_probs.clone() if unlabeled_probs is not None else None
    mixed_weight = unlabeled_weight.clone()
    mix_source_mask = torch.zeros_like(unlabeled_mask, dtype=torch.float32)

    # Exactly one permutation supplies both labeled RGB and its aligned GT.
    donor_perm = torch.randperm(batch, device=labeled_image.device)
    target_height, target_width = unlabeled_mask.shape[1:]
    source_height, source_width = labeled_mask.shape[1:]

    for i in range(batch):
        reliability_i = float(lst_reliabilities[i])
        if np.random.random() <= reliability_i:
            continue

        row1, col1, row2, col2 = (
            int(value) for value in target_boxes[i].tolist()
        )

        if not (
            0 <= row1 < row2 <= target_height
            and 0 <= col1 < col2 <= target_width
        ):
            raise ValueError(
                f"C4 invalid target box for target {i}: "
                f"[row1={row1}, col1={col1}, row2={row2}, col2={col2}] "
                f"outside height={target_height}, width={target_width}"
            )

        if row2 > source_height or col2 > source_width:
            raise ValueError(
                f"C4 target box for target {i} exceeds labeled source geometry "
                f"height={source_height}, width={source_width}"
            )

        donor_i = int(donor_perm[i].item())

        source_image = labeled_image[
            donor_i, :, row1:row2, col1:col2
        ]
        source_gt = labeled_mask[
            donor_i, row1:row2, col1:col2
        ]

        destination_image = mixed_image[
            i, :, row1:row2, col1:col2
        ]
        destination_target = mixed_target[
            i, row1:row2, col1:col2
        ]
        if source_image.shape != destination_image.shape:
            raise ValueError(
                f"C4 image crop shape mismatch for target {i}, donor {donor_i}: "
                f"source={tuple(source_image.shape)}, destination={tuple(destination_image.shape)}"
            )
        if source_gt.shape != destination_target.shape:
            raise ValueError(
                f"C4 GT crop shape mismatch for target {i}, donor {donor_i}: "
                f"source={tuple(source_gt.shape)}, destination={tuple(destination_target.shape)}"
            )

        mixed_image[
            i, :, row1:row2, col1:col2
        ] = source_image

        mixed_target[
            i, row1:row2, col1:col2
        ] = source_gt

        mixed_logits[
            i, row1:row2, col1:col2
        ] = 1.0

        if mixed_probs is not None:
            mixed_probs[
                i, :, row1:row2, col1:col2
            ] = 0.0

        valid_labeled = source_gt.ne(ignore_index)

        mixed_weight[
            i, row1:row2, col1:col2
        ] = valid_labeled.to(dtype=mixed_weight.dtype)

        mix_source_mask[
            i, row1:row2, col1:col2
        ] = 1.0

    if mixed_probs is not None:
        return mixed_image, mixed_target, mixed_logits, mix_source_mask, mixed_probs, mixed_weight
    return mixed_image, mixed_target, mixed_logits, mix_source_mask, mixed_weight


def cut_mix_label_adaptive_with_mask(
    unlabeled_image,
    unlabeled_mask,
    unlabeled_logits,
    labeled_image,
    labeled_mask,
    lst_confidences,
    return_target_metadata=False,
    unlabeled_probs=None,
    unlabeled_weight=None,
    labeled_boxes=None,
    labeled_masks=None,
    direct_labeled_mix=False,
    direct_paste_policy="same_coordinate",
    direct_confidence_gate=False,
    target_boxes=None,
):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    mix_unlabeled_probs = unlabeled_probs.clone() if unlabeled_probs is not None else None
    mix_unlabeled_weight = unlabeled_weight.clone() if unlabeled_weight is not None else None
    target_component_mask = unlabeled_mask.clone()
    target_component_logits = unlabeled_logits.clone()
    mix_target_component_mask = target_component_mask.clone()
    mix_target_component_logits = target_component_logits.clone()
    mix_source_mask = torch.zeros_like(unlabeled_mask, dtype=torch.float32)
    source_mask = torch.zeros_like(unlabeled_mask, dtype=torch.float32)
    labeled_logits = torch.ones_like(labeled_mask)

    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    if labeled_boxes is not None and labeled_masks is not None:
        raise ValueError("labeled_boxes and labeled_masks cannot both be provided")
    if direct_paste_policy not in ("same_coordinate", "random_target"):
        raise ValueError(f"Unsupported direct_paste_policy: {direct_paste_policy}")

    def _return_with_optional_metadata():
        if return_target_metadata:
            if mix_unlabeled_probs is not None and mix_unlabeled_weight is not None:
                return (
                    mix_unlabeled_image,
                    mix_unlabeled_target,
                    mix_unlabeled_logits,
                    mix_source_mask,
                    target_component_mask,
                    target_component_logits,
                    mix_unlabeled_probs,
                    mix_unlabeled_weight,
                )
            if mix_unlabeled_probs is not None:
                return (
                    mix_unlabeled_image,
                    mix_unlabeled_target,
                    mix_unlabeled_logits,
                    mix_source_mask,
                    target_component_mask,
                    target_component_logits,
                    mix_unlabeled_probs,
                )
            if mix_unlabeled_weight is not None:
                return (
                    mix_unlabeled_image,
                    mix_unlabeled_target,
                    mix_unlabeled_logits,
                    mix_source_mask,
                    target_component_mask,
                    target_component_logits,
                    mix_unlabeled_weight,
                )
            return (
                mix_unlabeled_image,
                mix_unlabeled_target,
                mix_unlabeled_logits,
                mix_source_mask,
                target_component_mask,
                target_component_logits,
            )

        if mix_unlabeled_probs is not None and mix_unlabeled_weight is not None:
            return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_source_mask, mix_unlabeled_probs, mix_unlabeled_weight
        if mix_unlabeled_probs is not None:
            return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_source_mask, mix_unlabeled_probs
        if mix_unlabeled_weight is not None:
            return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_source_mask, mix_unlabeled_weight
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_source_mask

    if direct_labeled_mix and labeled_boxes is not None:
        labeled_boxes = torch.as_tensor(labeled_boxes, device=unlabeled_image.device, dtype=torch.long)
        if labeled_boxes.shape != (unlabeled_image.size(0), 4):
            raise ValueError("labeled_boxes must have shape [B,4]")
        batch, height, width = unlabeled_mask.shape

        for i in range(batch):
            if direct_confidence_gate and np.random.random() <= lst_confidences[i]:
                continue
            src = int(u_rand_index[i].item())
            src_h1, src_w1, src_h2, src_w2 = labeled_boxes[src].tolist()
            src_h1 = max(0, min(int(src_h1), height))
            src_h2 = max(0, min(int(src_h2), height))
            src_w1 = max(0, min(int(src_w1), width))
            src_w2 = max(0, min(int(src_w2), width))
            if src_h2 <= src_h1 or src_w2 <= src_w1:
                continue

            crop_h = src_h2 - src_h1
            crop_w = src_w2 - src_w1
            if direct_paste_policy == "random_target":
                max_h = height - crop_h
                max_w = width - crop_w
                dst_h1 = int(torch.randint(0, max_h + 1, (1,), device=unlabeled_image.device).item())
                dst_w1 = int(torch.randint(0, max_w + 1, (1,), device=unlabeled_image.device).item())
                dst_h2 = dst_h1 + crop_h
                dst_w2 = dst_w1 + crop_w
            else:
                dst_h1, dst_w1, dst_h2, dst_w2 = src_h1, src_w1, src_h2, src_w2

            mix_unlabeled_image[i, :, dst_h1:dst_h2, dst_w1:dst_w2] = labeled_image[
                src, :, src_h1:src_h2, src_w1:src_w2
            ]
            mix_unlabeled_target[i, dst_h1:dst_h2, dst_w1:dst_w2] = labeled_mask[
                src, src_h1:src_h2, src_w1:src_w2
            ]
            mix_unlabeled_logits[i, dst_h1:dst_h2, dst_w1:dst_w2] = labeled_logits[
                src, src_h1:src_h2, src_w1:src_w2
            ].to(
                dtype=mix_unlabeled_logits.dtype
            )
            if mix_unlabeled_probs is not None:
                mix_unlabeled_probs[i, :, dst_h1:dst_h2, dst_w1:dst_w2] = 0.0
            if mix_unlabeled_weight is not None:
                mix_unlabeled_weight[i, dst_h1:dst_h2, dst_w1:dst_w2] = 1.0
            mix_source_mask[i, dst_h1:dst_h2, dst_w1:dst_w2] = 1.0

        return _return_with_optional_metadata()

    if direct_labeled_mix and labeled_masks is not None:
        labeled_masks = torch.as_tensor(labeled_masks, device=unlabeled_image.device, dtype=torch.bool)
        if labeled_masks.shape != unlabeled_mask.shape:
            raise ValueError("labeled_masks must have shape [B,H,W]")
        shuffled_masks = labeled_masks[u_rand_index]

        for i in range(unlabeled_mask.size(0)):
            src = int(u_rand_index[i].item())
            mask = shuffled_masks[i]
            if not mask.any():
                continue
            mix_unlabeled_image[i, :, mask] = labeled_image[src, :, mask]
            mix_unlabeled_target[i, mask] = labeled_mask[src, mask]
            mix_unlabeled_logits[i, mask] = labeled_logits[src, mask].to(dtype=mix_unlabeled_logits.dtype)
            if mix_unlabeled_probs is not None:
                mix_unlabeled_probs[i, :, mask] = 0.0
            if mix_unlabeled_weight is not None:
                mix_unlabeled_weight[i, mask] = 1.0
            mix_source_mask[i, mask] = 1.0

        return _return_with_optional_metadata()

    if labeled_boxes is None:
        l_bbx1, l_bby1, l_bbx2, l_bby2 = _rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))
    else:
        labeled_boxes = torch.as_tensor(labeled_boxes, device=unlabeled_image.device, dtype=torch.long)
        if labeled_boxes.shape != (unlabeled_image.size(0), 4):
            raise ValueError("labeled_boxes must have shape [B,4]")
        labeled_boxes = labeled_boxes[u_rand_index]
        l_bbx1 = labeled_boxes[:, 0].detach().cpu().numpy()
        l_bby1 = labeled_boxes[:, 1].detach().cpu().numpy()
        l_bbx2 = labeled_boxes[:, 2].detach().cpu().numpy()
        l_bby2 = labeled_boxes[:, 3].detach().cpu().numpy()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = _rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            mix_unlabeled_image[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = (
                labeled_image[u_rand_index[i], :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            )

            mix_unlabeled_target[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = (
                labeled_mask[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            )

            mix_unlabeled_logits[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = (
                labeled_logits[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]].to(
                    dtype=mix_unlabeled_logits.dtype
                )
            )
            if mix_unlabeled_probs is not None:
                mix_unlabeled_probs[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = 0.0
            if mix_unlabeled_weight is not None:
                mix_unlabeled_weight[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = 1.0

            mix_source_mask[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = 1.0

    if target_boxes is not None:
        target_boxes = torch.as_tensor(target_boxes, device=unlabeled_image.device, dtype=torch.long)
        if target_boxes.shape != (unlabeled_image.size(0), 4):
            raise ValueError("target_boxes must have shape [B,4]")
        u_bbx1 = target_boxes[:, 0].detach().cpu().numpy()
        u_bby1 = target_boxes[:, 1].detach().cpu().numpy()
        u_bbx2 = target_boxes[:, 2].detach().cpu().numpy()
        u_bby2 = target_boxes[:, 3].detach().cpu().numpy()

    for i in range(0, unlabeled_image.shape[0]):
        unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

        unlabeled_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_unlabeled_target[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

        unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )
        if unlabeled_probs is not None:
            unlabeled_probs[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
                mix_unlabeled_probs[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            )
        if unlabeled_weight is not None:
            unlabeled_weight[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
                mix_unlabeled_weight[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            )

        target_component_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_target_component_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

        target_component_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_target_component_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

        source_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = (
            mix_source_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        )

    if return_target_metadata:
        if unlabeled_probs is not None and unlabeled_weight is not None:
            return (
                unlabeled_image,
                unlabeled_mask,
                unlabeled_logits,
                source_mask,
                target_component_mask,
                target_component_logits,
                unlabeled_probs,
                unlabeled_weight,
            )
        if unlabeled_probs is not None:
            return (
                unlabeled_image,
                unlabeled_mask,
                unlabeled_logits,
                source_mask,
                target_component_mask,
                target_component_logits,
                unlabeled_probs,
            )
        if unlabeled_weight is not None:
            return (
                unlabeled_image,
                unlabeled_mask,
                unlabeled_logits,
                source_mask,
                target_component_mask,
                target_component_logits,
                unlabeled_weight,
            )
        return unlabeled_image, unlabeled_mask, unlabeled_logits, source_mask, target_component_mask, target_component_logits

    if unlabeled_probs is not None and unlabeled_weight is not None:
        return unlabeled_image, unlabeled_mask, unlabeled_logits, source_mask, unlabeled_probs, unlabeled_weight
    if unlabeled_probs is not None:
        return unlabeled_image, unlabeled_mask, unlabeled_logits, source_mask, unlabeled_probs
    if unlabeled_weight is not None:
        return unlabeled_image, unlabeled_mask, unlabeled_logits, source_mask, unlabeled_weight
    return unlabeled_image, unlabeled_mask, unlabeled_logits, source_mask


def thresholded_boundary_mix_loss(
    predict,
    target,
    confidence,
    mix_mask,
    *,
    thresh=0.95,
    ignore_index=255,
    kernel_size=5,
    gamma_in=0.7,
    gamma_out=0.3,
    use_confidence=True,
    normalize_weight=True,
):
    valid = confidence.ge(thresh).bool() & target.ne(ignore_index).bool()
    target = target.clone()
    target[~valid] = ignore_index
    weight = boundary_confidence_weight_map(
        mix_mask,
        confidence,
        kernel_size=kernel_size,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
        use_confidence=use_confidence,
        normalize_weight=normalize_weight,
    )
    loss = weighted_cross_entropy_loss(
        predict,
        target,
        weight,
        ignore_index=ignore_index,
    )
    return loss, valid.float().mean()


def _rand_bbox(size, lam=None):
    if len(size) == 4:
        height = size[2]
        width = size[3]
    elif len(size) == 3:
        height = size[1]
        width = size[2]
    else:
        raise ValueError("size must have 3 or 4 dimensions")

    batch = size[0]

    cut_ratio = np.sqrt(1.0 - lam)
    cut_height = int(height * cut_ratio)
    cut_width = int(width * cut_ratio)

    center_row = np.random.randint(
        size=[batch],
        low=int(height / 8),
        high=height,
    )
    center_col = np.random.randint(
        size=[batch],
        low=int(width / 8),
        high=width,
    )

    row1 = np.clip(center_row - cut_height // 2, 0, height)
    col1 = np.clip(center_col - cut_width // 2, 0, width)
    row2 = np.clip(center_row + cut_height // 2, 0, height)
    col2 = np.clip(center_col + cut_width // 2, 0, width)

    return row1, col1, row2, col2
