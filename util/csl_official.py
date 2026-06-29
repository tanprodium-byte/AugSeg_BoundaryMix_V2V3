import math

import torch
import torch.nn.functional as F


def _float_stat(value):
    return float(value.detach().float().item())


@torch.no_grad()
def get_max_confidence_and_residual_variance(predictions, valid_mask):
    """
    Official CSL PCOS-style features from:
    https://github.com/PanLiuCSU/CSL/blob/main/util/PCOS.py

    predictions: [B,C,H,W] teacher probabilities.
    valid_mask: [B,H,W] bool/0-1 mask; invalid pixels become NaN features.
    """
    if predictions.dim() != 4:
        raise ValueError("predictions must have shape [B,C,H,W]")
    if valid_mask.dim() != 3:
        raise ValueError("valid_mask must have shape [B,H,W]")
    if predictions.shape[0] != valid_mask.shape[0] or predictions.shape[2:] != valid_mask.shape[1:]:
        raise ValueError("predictions and valid_mask spatial shapes must match")

    predictions = predictions.detach().float()
    valid_mask = valid_mask.to(device=predictions.device, dtype=torch.bool)
    valid_expanded = valid_mask.unsqueeze(1).expand_as(predictions)
    nan = torch.tensor(float("nan"), device=predictions.device, dtype=predictions.dtype)
    masked_predictions = torch.where(valid_expanded, predictions, nan)

    safe_predictions = torch.where(valid_expanded, predictions, torch.zeros_like(predictions))
    max_confidence, max_indices = torch.max(safe_predictions, dim=1)
    max_confidence = torch.where(valid_mask, max_confidence, torch.full_like(max_confidence, float("nan")))

    one_hot_max = F.one_hot(max_indices, num_classes=predictions.shape[1]).permute(0, 3, 1, 2)
    remaining_predictions = masked_predictions * (1 - one_hot_max)
    num_remaining_classes = max(predictions.shape[1] - 1, 1)
    mean_remaining_predictions = torch.sum(remaining_predictions, dim=1) / num_remaining_classes
    remaining_predictions_diff = remaining_predictions - mean_remaining_predictions.unsqueeze(1)
    residual_variance = torch.sum(remaining_predictions_diff ** 2, dim=1) / num_remaining_classes
    residual_variance = torch.where(valid_mask, residual_variance, torch.full_like(residual_variance, float("nan")))
    return max_confidence, residual_variance


@torch.no_grad()
def batch_class_stats(max_conf, res_var):
    means = []
    vars_ = []
    for index in range(max_conf.shape[0]):
        features = torch.stack([max_conf[index], res_var[index]], dim=-1).view(-1, 2)
        valid_mask = ~torch.isnan(features).any(dim=-1)
        valid_features = features[valid_mask]

        if valid_features.size(0) == 0:
            means.append(torch.tensor((1.0, 0.0), device=max_conf.device, dtype=max_conf.dtype))
            vars_.append(torch.tensor((1.0, 1.0), device=max_conf.device, dtype=max_conf.dtype))
            continue
        class_assignments = _class_assignment(valid_features, 2)
        class_centers = _compute_class_centers(valid_features, class_assignments, 2)
        max_mean_idx = torch.argmax(class_centers[0][:, 0])
        means.append(class_centers[0][max_mean_idx])
        vars_.append(class_centers[1][max_mean_idx])
    return torch.stack(means), torch.stack(vars_)


@torch.no_grad()
def _compute_eigenvectors_with_svd(X, num_classes):
    _, S, Vt = torch.linalg.svd(X.T, full_matrices=False)
    eigvals = S ** 2
    idx = torch.argsort(-eigvals)
    return Vt.T[:, idx[:num_classes]]


@torch.no_grad()
def _class_assignment(input, num_classes):
    eigenvectors = _compute_eigenvectors_with_svd(input, num_classes)
    return torch.argmax(torch.abs(eigenvectors), dim=1)


@torch.no_grad()
def _compute_class_centers(features, class_assignments, num_classes):
    means = []
    vars_ = []
    for class_id in range(num_classes):
        points_in_class = features[class_assignments == class_id]
        num_points = points_in_class.size(0)
        if num_points == 0:
            mean = torch.zeros(features.size(1), device=features.device, dtype=features.dtype)
            var = torch.zeros(features.size(1), device=features.device, dtype=features.dtype)
        elif num_points == 1:
            mean = points_in_class.squeeze(0)
            var = torch.zeros(features.size(1), device=features.device, dtype=features.dtype)
        else:
            mean = points_in_class.mean(dim=0)
            var = points_in_class.var(dim=0, unbiased=True)
        means.append(mean)
        vars_.append(var)
    return torch.stack(means), torch.stack(vars_)


def _valid_mask_from_ignore(ignore_mask, shape, device):
    if ignore_mask is None:
        return torch.ones(shape, device=device, dtype=torch.bool)
    ignore_mask = ignore_mask.to(device=device)
    if ignore_mask.dtype == torch.bool:
        return ignore_mask
    return ignore_mask.ne(255)


@torch.no_grad()
def compute_csl_official_selection(
    teacher_probs,
    ignore_mask=None,
    alpha=8.0,
    eps=1e-8,
    reliable_mode="weight_eq_one",
):
    """
    Official CSL-like PCOS selection/weight wrapper.

    Weight formula follows SemiModule.get_weight in:
    https://github.com/PanLiuCSU/CSL/blob/main/train/semi_supervised_train.py
    """
    if teacher_probs.dim() != 4:
        raise ValueError("teacher_probs must have shape [B,C,H,W]")
    probs = teacher_probs.detach().float().clamp_min(float(eps))
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(float(eps))
    batch, _, height, width = probs.shape
    valid_mask = _valid_mask_from_ignore(ignore_mask, (batch, height, width), probs.device)

    max_confidence, residual_variance = get_max_confidence_and_residual_variance(probs, valid_mask)
    means, vars_ = batch_class_stats(max_confidence, residual_variance)
    conf_mean = means[:, 0].view(-1, 1, 1)
    res_mean = means[:, 1].view(-1, 1, 1)
    conf_var = vars_[:, 0].view(-1, 1, 1)
    res_var = vars_[:, 1].view(-1, 1, 1)

    conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + float(eps))
    res_z = (res_mean - residual_variance) / torch.sqrt(res_var + float(eps))
    weight_conf = torch.exp(-(conf_z ** 2) / float(alpha))
    weight_res = torch.exp(-(res_z ** 2) / float(alpha))
    weight = weight_conf * weight_res
    confident_mask = (conf_z > 0) | (res_z > 0)
    weight = torch.where(confident_mask, torch.ones_like(weight), weight)
    weight = torch.where(valid_mask, weight, torch.zeros_like(weight))
    weight = torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).detach()

    if reliable_mode == "weight_eq_one":
        reliable_mask = torch.isclose(weight, torch.ones_like(weight), rtol=1e-5, atol=1e-6) & valid_mask
    elif reliable_mode == "weight_gt_zero":
        reliable_mask = weight.gt(0) & valid_mask
    else:
        raise ValueError("Unsupported reliable_mode: %s" % reliable_mode)

    valid_float = valid_mask.float()
    valid_count = valid_float.flatten(1).sum(dim=1).clamp_min(1.0)
    sample_reliability = (weight * valid_float).flatten(1).sum(dim=1) / valid_count
    sample_reliability = torch.nan_to_num(sample_reliability, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    pseudo_label = torch.argmax(probs, dim=1).long()
    pseudo_label = torch.where(valid_mask, pseudo_label, torch.zeros_like(pseudo_label))

    valid_weight = weight[valid_mask] if valid_mask.any() else weight.reshape(-1)
    stats = {
        "csl/mean_reliability": _float_stat(weight.mean()),
        "csl/reliability_std": _float_stat(weight.std(unbiased=False)),
        "csl/reliability_min": _float_stat(weight.min()),
        "csl/reliability_max": _float_stat(weight.max()),
        "csl/confidence_mean": _float_stat(torch.nan_to_num(max_confidence, nan=0.0).mean()),
        "csl/residual_variance_mean": _float_stat(torch.nan_to_num(residual_variance, nan=0.0).mean()),
        "csl/official_weight_valid_mean": _float_stat(valid_weight.mean()) if valid_weight.numel() else 0.0,
        "csl/reliable_ratio": _float_stat((reliable_mask & valid_mask).float().sum() / valid_float.sum().clamp_min(1.0)),
        "csl/sample_reliability_mean": _float_stat(sample_reliability.mean()),
    }

    return {
        "pseudo_label": pseudo_label.detach(),
        "raw_confidence": torch.nan_to_num(max_confidence, nan=0.0).detach(),
        "residual_variance": torch.nan_to_num(residual_variance, nan=0.0).detach(),
        "weight": weight,
        "reliable_mask": reliable_mask.detach(),
        "sample_reliability": sample_reliability.detach(),
        "stats": stats,
    }


@torch.no_grad()
def apply_csl_reliable_mask_perturbation(
    image,
    reliable_mask,
    mask_prob=0.3,
    block_size=1,
    cover_ratio=1.0,
    mode="zero",
):
    """
    Perturb reliable pixels on the input branch, matching CSL's idea of masking
    confident/reliable pixels before student forward. The official code combines
    a block cover mask with `weight == 1` reliable pixels.
    """
    if image.dim() != 4:
        raise ValueError("image must have shape [B,C,H,W]")
    if reliable_mask.dim() != 3:
        raise ValueError("reliable_mask must have shape [B,H,W]")
    if image.shape[0] != reliable_mask.shape[0] or image.shape[2:] != reliable_mask.shape[1:]:
        raise ValueError("image and reliable_mask shapes must match")

    batch, _, height, width = image.shape
    reliable_mask = reliable_mask.to(device=image.device, dtype=torch.bool)
    mask_prob = max(0.0, min(1.0, float(mask_prob)))
    cover_ratio = max(0.0, min(1.0, float(cover_ratio)))
    effective_prob = mask_prob * cover_ratio

    if int(block_size) > 1:
        bh = math.ceil(height / int(block_size))
        bw = math.ceil(width / int(block_size))
        random_mask = torch.rand((batch, 1, bh, bw), device=image.device).lt(effective_prob).float()
        random_mask = F.interpolate(random_mask, size=(height, width), mode="nearest").squeeze(1).bool()
    else:
        random_mask = torch.rand((batch, height, width), device=image.device).lt(effective_prob)

    perturb_mask = reliable_mask & random_mask
    perturbed = image.clone()
    if mode == "zero":
        perturbed = torch.where(perturb_mask.unsqueeze(1), torch.zeros_like(perturbed), perturbed)
    else:
        raise ValueError("Unsupported CSL perturbation mode: %s" % mode)

    denom = reliable_mask.float().sum().clamp_min(1.0)
    stats = {
        "csl/mask_prob": float(mask_prob),
        "csl/masked_ratio": _float_stat(perturb_mask.float().mean()),
        "csl/raw_reliability_mean": _float_stat(reliable_mask.float().mean()),
        "csl/effective_weight_mean": _float_stat((~perturb_mask).float().mean()),
        "csl/perturb_reliable_ratio": _float_stat(perturb_mask.float().sum() / denom),
    }
    return perturbed.detach(), perturb_mask.detach(), stats
