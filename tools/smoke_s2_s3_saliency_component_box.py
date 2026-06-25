import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.saliency_cutmix import (
    bbox_from_mask,
    connected_components_from_gt,
    expand_box,
    get_saliency_component_guided_boxes,
    get_saliency_component_guided_masks,
    get_saliency_guided_boxes,
)
from util.boundary_mix import cut_mix_label_adaptive_with_mask


class FakeTeacher(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x):
        return self.net(x), None


def _base_sampler(size, lam=None):
    batch = size[0]
    return (
        np.zeros(batch, dtype=np.int64),
        np.zeros(batch, dtype=np.int64),
        np.full(batch, size[2] // 2, dtype=np.int64),
        np.full(batch, size[3] // 2, dtype=np.int64),
    )


def test_s1_box_mode_still_returns_boxes():
    torch.manual_seed(1)
    np.random.seed(1)
    teacher = FakeTeacher(num_classes=3)
    image = torch.randn(2, 3, 16, 16)
    label = torch.randint(0, 3, (2, 16, 16))
    boxes, stats = get_saliency_guided_boxes(
        teacher,
        image,
        label,
        _base_sampler,
        num_candidates=2,
        temperature=0.2,
        ignore_index=255,
        lam_sampler=lambda: 0.5,
    )
    assert boxes.shape == (2, 4)
    assert "saliency/score_selected" in stats


def test_component_extraction_filters_ignore_background_and_area():
    label = torch.zeros(20, 20, dtype=torch.long)
    label[1:12, 1:12] = 1
    label[0:3, 0:3] = 255
    label[15:17, 15:17] = 2
    label[12:20, 0:8] = 0

    components = connected_components_from_gt(
        label,
        ignore_label=255,
        foreground_only=True,
        connectivity=8,
        min_component_area=16,
        max_component_area=200,
    )
    assert len(components) == 1
    assert components[0]["class_id"] == 1
    assert components[0]["area"] == 117

    no_area_filter = connected_components_from_gt(
        label,
        ignore_label=255,
        foreground_only=True,
        connectivity=8,
        min_component_area=1,
        max_component_area=200,
    )
    assert {component["class_id"] for component in no_area_filter} == {1, 2}


def test_bbox_and_expand_convention():
    mask = torch.zeros(10, 12, dtype=torch.bool)
    mask[2:6, 3:8] = True
    assert bbox_from_mask(mask) == (2, 3, 6, 8)
    assert expand_box((2, 3, 6, 8), (10, 12), expand_ratio=1.0) == (2, 3, 6, 8)
    expanded = expand_box((2, 3, 6, 8), (10, 12), expand_ratio=1.5)
    assert expanded[0] <= 2 and expanded[1] <= 3
    assert expanded[2] >= 6 and expanded[3] >= 8
    assert 0 <= expanded[0] < expanded[2] <= 10
    assert 0 <= expanded[1] < expanded[3] <= 12


def test_component_box_fallback_when_no_valid_component():
    torch.manual_seed(2)
    np.random.seed(2)
    teacher = FakeTeacher(num_classes=3)
    image = torch.randn(2, 3, 16, 16)
    label = torch.zeros(2, 16, 16, dtype=torch.long)
    label[1].fill_(255)

    boxes, stats = get_saliency_component_guided_boxes(
        teacher,
        image,
        label,
        _base_sampler,
        temperature=0.2,
        ignore_index=255,
        min_component_area=64,
        max_component_area=20000,
        lam_sampler=lambda: 0.5,
    )
    assert boxes.shape == (2, 4)
    assert torch.equal(boxes, torch.tensor([[0, 0, 8, 8], [0, 0, 8, 8]]))
    assert stats["saliency/fallback_ratio"] == 1.0


def test_component_box_returns_boxes_and_stats():
    torch.manual_seed(3)
    np.random.seed(3)
    teacher = FakeTeacher(num_classes=3)
    image = torch.randn(2, 3, 16, 16)
    label = torch.zeros(2, 16, 16, dtype=torch.long)
    label[0, 2:12, 2:12] = 1
    label[1, 3:14, 4:15] = 2

    boxes, stats = get_saliency_component_guided_boxes(
        teacher,
        image,
        label,
        _base_sampler,
        temperature=0.2,
        ignore_index=255,
        min_component_area=64,
        max_component_area=20000,
        box_expand_ratio=1.2,
        lam_sampler=lambda: 0.5,
    )
    assert boxes.shape == (2, 4)
    for key in (
        "saliency/num_components",
        "saliency/num_valid_components",
        "saliency/selected_component_class",
        "saliency/selected_component_area",
        "saliency/selected_component_score",
        "saliency/component_score_mean",
        "saliency/component_score_max",
        "saliency/component_box_area",
        "saliency/selection_entropy",
        "saliency/fallback_ratio",
    ):
        assert key in stats
        assert np.isfinite(stats[key])
    assert stats["saliency/fallback_ratio"] == 0.0


def test_component_mask_returns_bool_masks_and_stats():
    torch.manual_seed(5)
    np.random.seed(5)
    teacher = FakeTeacher(num_classes=3)
    image = torch.randn(2, 3, 16, 16)
    label = torch.zeros(2, 16, 16, dtype=torch.long)
    label[0, 2:12, 2:12] = 1
    label[1, 3:14, 4:15] = 2

    masks, stats = get_saliency_component_guided_masks(
        teacher,
        image,
        label,
        _base_sampler,
        temperature=0.2,
        ignore_index=255,
        min_component_area=64,
        max_component_area=20000,
        lam_sampler=lambda: 0.5,
    )
    assert masks.shape == (2, 16, 16)
    assert masks.dtype == torch.bool
    assert masks.device == image.device
    assert stats["saliency/fallback_ratio"] == 0.0


def test_direct_component_mask_paste_only_changes_selected_mask():
    batch, height, width = 1, 10, 12
    unlabeled_image = torch.zeros(batch, 3, height, width)
    unlabeled_mask = torch.zeros(batch, height, width, dtype=torch.long)
    unlabeled_logits = torch.zeros(batch, height, width)
    labeled_image = torch.full((batch, 3, height, width), 9.0)
    labeled_mask = torch.full((batch, height, width), 4, dtype=torch.long)
    labeled_masks = torch.zeros(batch, height, width, dtype=torch.bool)
    labeled_masks[0, 2:6, 3:8] = True

    image, mask, logits, source_mask = cut_mix_label_adaptive_with_mask(
        unlabeled_image,
        unlabeled_mask,
        unlabeled_logits,
        labeled_image,
        labeled_mask,
        [0.0],
        labeled_masks=labeled_masks,
        direct_labeled_mix=True,
    )

    outside = ~labeled_masks
    assert torch.equal(image[0, :, labeled_masks[0]], torch.full((3, int(labeled_masks.sum().item())), 9.0))
    assert torch.equal(image[0, :, outside[0]], torch.zeros_like(image[0, :, outside[0]]))
    assert torch.equal(mask[0, labeled_masks[0]], torch.full((int(labeled_masks.sum().item()),), 4, dtype=torch.long))
    assert torch.equal(mask[0, outside[0]], torch.zeros_like(mask[0, outside[0]]))
    assert source_mask.sum().item() == labeled_masks.sum().item()


def test_s3_direct_config_enables_component_mask_and_v3_d2_without_v2_or_csl():
    config_path = os.path.join(
        ROOT,
        "exps/boundary_mix_v2_v3/voc_semi662/s3_saliency_component_mask_direct_plus_v3_d2/config.yaml",
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["saliency_cutmix"]["enabled"] is True
    assert cfg["saliency_cutmix"]["mode"] == "component_mask"
    assert cfg["saliency_cutmix"]["direct_labeled_mix"] is True
    assert cfg["saliency_cutmix"]["paste_mode"] == "mask"
    assert cfg["boundary_component"]["enabled"] is False
    assert cfg["boundary_compatibility"]["enabled"] is True
    assert cfg["boundary_compatibility"]["pair_radius"] == 2
    assert cfg["boundary_compatibility"]["lambda_bcr"] == 0.01
    assert cfg["boundary_compatibility"]["use_component_gate"] is False
    assert cfg["csl"]["enabled"] is False


def test_s3_config_enables_v3_d2_without_v2_or_csl():
    config_path = os.path.join(
        ROOT,
        "exps/boundary_mix_v2_v3/voc_semi662/s3_saliency_component_box_plus_v3_d2/config.yaml",
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["saliency_cutmix"]["enabled"] is True
    assert cfg["saliency_cutmix"]["mode"] == "component_box"
    assert cfg["boundary_component"]["enabled"] is False
    assert cfg["boundary_compatibility"]["enabled"] is True
    assert cfg["boundary_compatibility"]["pair_radius"] == 2
    assert cfg["boundary_compatibility"]["lambda_bcr"] == 0.01
    assert cfg["boundary_compatibility"]["use_component_gate"] is False
    assert cfg["csl"]["enabled"] is False


def main():
    test_s1_box_mode_still_returns_boxes()
    test_component_extraction_filters_ignore_background_and_area()
    test_bbox_and_expand_convention()
    test_component_box_fallback_when_no_valid_component()
    test_component_box_returns_boxes_and_stats()
    test_component_mask_returns_bool_masks_and_stats()
    test_direct_component_mask_paste_only_changes_selected_mask()
    test_s3_config_enables_v3_d2_without_v2_or_csl()
    test_s3_direct_config_enables_component_mask_and_v3_d2_without_v2_or_csl()
    print("S2/S3 saliency component-box smoke tests passed")


if __name__ == "__main__":
    main()
