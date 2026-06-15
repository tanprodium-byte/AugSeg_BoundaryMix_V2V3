# S1 Saliency Box CutMix Implementation Audit

## Baseline Config Source

S1 config was copied from the AugSeg baseline profile at:

`exps/boundary_mix_v2_v3/voc_semi662_c321_gbs8/baseline/config.yaml`

It keeps VOC semi-supervised 662 labels, crop 321, ResNet-101, batch size 8, optimizer, LR schedule, confidence threshold 0.95, and the original AugSeg strong augmentation/CutMix settings.

## Existing Random CutMix Location

Baseline CutMix is created in `augseg/dataset/augs_ALIA.py::cut_mix_label_adaptive`.

The BoundaryMix/V2/V3 path uses the compatible wrapper:

`util/boundary_mix.py::cut_mix_label_adaptive_with_mask`

Both implementations sample two random rectangular boxes:

- `l_bbx*` with `lam=np.random.beta(8, 2)` for the labeled-source adaptive paste.
- `u_bbx*` with `lam=np.random.beta(4, 4)` for the unlabeled copy-paste stage.

## Baseline Mix Count And Sources

The baseline performs two rectangular paste stages inside one CutMix call:

1. Labeled-source paste into a temporary unlabeled batch when `np.random.random() > confidence[i]`.
2. Unlabeled copy-paste from the temporary mixed batch into the final strong unlabeled batch.

S1 changes only the labeled-source box from stage 1. The second unlabeled copy-paste box remains random.

## S1 Training Change

`train_semi.py` now reads `saliency_cutmix.enabled`.

When enabled, it computes teacher gradient saliency on the labeled batch and passes selected labeled-source boxes to `cut_mix_label_adaptive_with_mask(..., labeled_boxes=...)`.

When disabled, the existing baseline/A1/BoundaryMix paths are preserved:

- No saliency is computed.
- Baseline non-BoundaryMix config still calls `cut_mix_label_adaptive`.
- BoundaryMix/V2/V3 configs still call the wrapper only for their existing mask/metadata needs.

## Saliency Computation

Implemented in `util/saliency_cutmix.py`.

`compute_labeled_teacher_saliency`:

- Clones `x_l` and enables input gradients only on that clone.
- Saves teacher train/eval mode and all parameter `requires_grad` flags.
- Runs the teacher in eval mode with teacher params frozen.
- Computes supervised CE against `y_l`.
- Uses `torch.autograd.grad` only for gradients with respect to input image.
- Computes RGB L2 gradient norm per pixel.
- Normalizes per image into `[0, 1]`.
- Detaches saliency and restores teacher mode and `requires_grad` flags.

`L_sal` is not added to training loss and no optimizer step is applied to the teacher.

## Candidate Boxes

S1 samples `K=8` candidate boxes using the same baseline box sampler and same labeled-source beta distribution, `np.random.beta(8, 2)`.

Each candidate score is mean saliency inside the candidate box. Mean is used rather than sum to avoid always preferring larger boxes.

Selection uses softmax sampling with default temperature `0.2`, not argmax.

## Source Permutation

AugSeg permutes the labeled source batch with `u_rand_index`. S1 computes one saliency-guided box per labeled image, then the wrapper reorders those boxes by the same `u_rand_index` so the box follows the actual labeled source image used for paste.

## Fallback

If saliency computation or shape validation fails during training:

- The error is logged only when debug logging is enabled.
- `labeled_boxes=None` is passed to the wrapper.
- The wrapper falls back to the original random labeled-source box.
- Training does not crash because of saliency failure.

## What S1 Does Not Change

- Image selection for unlabeled data.
- Image selection for labeled/source data.
- Number of CutMix paste stages.
- Confidence threshold or confidence computation.
- Teacher pseudo-label generation for unlabeled weak images.
- Unsupervised CE loss.
- Total training loss.
- BoundaryMix, V2, V3/BCR, or CSL.
- AugSeg schedule, optimizer, or EMA update.

## Runtime And Memory Risk

S1 adds one teacher forward and one input-gradient computation on the labeled batch for each CutMix iteration. This increases runtime and activation memory compared with random CutMix. Teacher parameters are frozen for saliency, but input-gradient computation still needs the backward graph through the teacher forward.
