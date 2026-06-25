import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.boundary_mix import cut_mix_label_adaptive_with_mask
import util.boundary_mix as boundary_mix
from util.saliency_cutmix import (
    boxes_to_masks,
    box_mean_saliency,
    get_saliency_guided_boxes,
    normalize_saliency_per_image,
    sample_box_by_softmax,
)


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


def test_normalize():
    saliency = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ]
    )
    norm = normalize_saliency_per_image(saliency)
    assert torch.isfinite(norm).all()
    assert float(norm.min()) >= 0.0
    assert float(norm.max()) <= 1.0
    assert torch.allclose(norm[1], torch.zeros_like(norm[1]))


def test_box_scores_and_softmax():
    saliency = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
    boxes = torch.tensor(
        [
            [[0, 0, 2, 2], [2, 2, 4, 4]],
            [[0, 0, 1, 1], [0, 0, 4, 4]],
        ]
    )
    scores = box_mean_saliency(saliency, boxes)
    assert scores.shape == (2, 2)
    idx, probs = sample_box_by_softmax(scores, temperature=0.2)
    assert idx.shape == (2,)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_get_saliency_guided_boxes_restores_teacher():
    torch.manual_seed(7)
    np.random.seed(7)
    teacher = FakeTeacher(num_classes=3)
    teacher.train()
    flags_before = [p.requires_grad for p in teacher.parameters()]
    params_before = [p.detach().clone() for p in teacher.parameters()]

    x = torch.randn(2, 3, 8, 8)
    y = torch.randint(0, 3, (2, 8, 8))
    boxes, stats = get_saliency_guided_boxes(
        teacher,
        x,
        y,
        _base_sampler,
        num_candidates=4,
        temperature=0.2,
        ignore_index=255,
        lam_sampler=lambda: 0.5,
    )

    assert boxes.shape == (2, 4)
    assert teacher.training
    assert [p.requires_grad for p in teacher.parameters()] == flags_before
    for before, after in zip(params_before, teacher.parameters()):
        assert torch.allclose(before, after.detach())
        assert after.grad is None
    for key in (
        "saliency/score_selected",
        "saliency/score_candidate_mean",
        "saliency/score_candidate_max",
        "saliency/score_candidate_min",
        "saliency/score_candidate_std",
        "saliency/prob_selected",
        "saliency/prob_max",
        "saliency/selection_entropy",
    ):
        assert key in stats
        assert np.isfinite(stats[key])


def test_box_coordinate_convention_matches_boundary_mix():
    saliency = torch.zeros(1, 5, 7)
    saliency[:, 1:4, 2:6] = 2.0
    boxes = torch.tensor([[[1, 2, 4, 6], [0, 0, 1, 1]]])

    scores = box_mean_saliency(saliency, boxes)
    masks = boxes_to_masks(boxes[:, 0], (1, 5, 7))

    assert torch.allclose(scores[0, 0], torch.tensor(2.0))
    assert masks[0, 1:4, 2:6].sum().item() == 12
    assert masks[0].sum().item() == 12


def test_random_fallback_wrapper_path():
    torch.manual_seed(3)
    np.random.seed(3)
    batch, height, width = 2, 8, 8
    unlabeled_image = torch.randn(batch, 3, height, width)
    unlabeled_mask = torch.zeros(batch, height, width, dtype=torch.long)
    unlabeled_logits = torch.ones(batch, height, width)
    labeled_image = torch.randn(batch, 3, height, width)
    labeled_mask = torch.ones(batch, height, width, dtype=torch.long)
    confidence = [0.0, 0.0]

    image, mask, logits, source_mask = cut_mix_label_adaptive_with_mask(
        unlabeled_image,
        unlabeled_mask,
        unlabeled_logits,
        labeled_image,
        labeled_mask,
        confidence,
        labeled_boxes=None,
    )
    assert image.shape == unlabeled_image.shape
    assert mask.shape == unlabeled_mask.shape
    assert logits.shape == unlabeled_logits.shape
    assert source_mask.shape == unlabeled_mask.shape
    assert torch.isfinite(image).all()


def test_direct_box_paste_skips_target_random_second_step():
    torch.manual_seed(4)
    np.random.seed(4)
    batch, height, width = 1, 8, 8
    unlabeled_image = torch.zeros(batch, 3, height, width)
    unlabeled_mask = torch.zeros(batch, height, width, dtype=torch.long)
    unlabeled_logits = torch.zeros(batch, height, width)
    labeled_image = torch.full((batch, 3, height, width), 7.0)
    labeled_mask = torch.full((batch, height, width), 3, dtype=torch.long)
    confidence = [0.0]
    labeled_boxes = torch.tensor([[2, 3, 5, 7]])

    old_rand_bbox = boundary_mix._rand_bbox
    try:
        def fail_rand_bbox(*args, **kwargs):
            raise AssertionError("direct box path must not call target random _rand_bbox")

        boundary_mix._rand_bbox = fail_rand_bbox
        image, mask, logits, source_mask = cut_mix_label_adaptive_with_mask(
            unlabeled_image,
            unlabeled_mask,
            unlabeled_logits,
            labeled_image,
            labeled_mask,
            confidence,
            labeled_boxes=labeled_boxes,
            direct_labeled_mix=True,
        )
    finally:
        boundary_mix._rand_bbox = old_rand_bbox

    expected = torch.zeros_like(unlabeled_image)
    expected[:, :, 2:5, 3:7] = 7.0
    assert torch.equal(image, expected)
    assert torch.equal(mask[:, 2:5, 3:7], torch.full((1, 3, 4), 3, dtype=torch.long))
    assert source_mask[:, 2:5, 3:7].sum().item() == 12
    assert source_mask.sum().item() == 12


def test_numeric_logger_stats():
    _, probs = sample_box_by_softmax(torch.ones(2, 3), temperature=0.2)
    assert torch.isfinite(probs).all()
    mode_box = 1.0
    enabled = 1.0
    assert isinstance(mode_box, float)
    assert isinstance(enabled, float)


def main():
    test_normalize()
    test_box_scores_and_softmax()
    test_get_saliency_guided_boxes_restores_teacher()
    test_box_coordinate_convention_matches_boundary_mix()
    test_random_fallback_wrapper_path()
    test_direct_box_paste_skips_target_random_second_step()
    test_numeric_logger_stats()
    print("S1 saliency CutMix smoke tests passed")


if __name__ == "__main__":
    main()
