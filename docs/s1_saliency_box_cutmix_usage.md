# S1 Saliency Box CutMix Usage

## Meaning

`S1_saliency_box_cutmix` is a phase-signal experiment. It keeps the AugSeg baseline pipeline and changes only the labeled-source rectangular CutMix box selection from random sampling to teacher-gradient-saliency-guided softmax sampling.

It is not BoundaryMix, V2, V3/BCR, CSL, or a saliency loss experiment.

## Config

`exps/boundary_mix_v2_v3/voc_semi662/s1_saliency_box_cutmix/config.yaml`

Important disabled blocks:

- `boundary_component.enabled: false`
- `boundary_compatibility.enabled: false`
- `boundary_compatibility.lambda_bcr: 0.0`
- `csl.enabled: false`

Important S1 block:

- `saliency_cutmix.enabled: true`
- `saliency_cutmix.num_candidates: 8`
- `saliency_cutmix.selection: softmax`
- `saliency_cutmix.temperature: 0.2`
- `saliency_cutmix.apply_to: labeled_source_only`
- `saliency_cutmix.use_saliency_for_loss_weight: false`
- `saliency_cutmix.use_saliency_conf_product: false`

## Train Command

Example distributed command:

```bash
python -m torch.distributed.launch --nproc_per_node=1 train_semi.py \
  --config exps/boundary_mix_v2_v3/voc_semi662/s1_saliency_box_cutmix/config.yaml
```

Do not run full training for smoke validation.

## Smoke Commands

```bash
python -m compileall train_semi.py util tools
python tools/smoke_s1_saliency_cutmix.py
```

## Log Keys

Core S1 keys:

- `saliency/enabled`
- `saliency/mode`
- `saliency/num_candidates`
- `saliency/temperature`
- `saliency/score_selected`
- `saliency/score_candidate_mean`
- `saliency/score_candidate_max`
- `saliency/score_candidate_min`
- `saliency/selection_entropy`
- `saliency/fallback_ratio`
- `saliency/source_is_labeled_ratio`

Mix-specific keys:

- `saliency/mix1_score_selected`
- `saliency/mix1_score_candidate_mean`
- `saliency/mix1_selection_entropy`
- `saliency/mix2_score_selected`
- `saliency/mix2_score_candidate_mean`
- `saliency/mix2_selection_entropy`

For this AugSeg implementation, mix 1 is the labeled-source adaptive paste and mix 2 is the unlabeled copy-paste stage. S1 only applies saliency to mix 1, so mix 2 saliency keys are logged as `NaN`.

## Checklist Before Training

- Confirm the S1 config path above.
- Confirm `saliency_cutmix.enabled: true`.
- Confirm BoundaryMix/V2/V3/BCR/CSL blocks are disabled.
- Confirm `trainer.unsupervised.threshold` remains `0.95`.
- Confirm no saliency loss is added to total training loss.
- Confirm no full training is started during smoke validation.
