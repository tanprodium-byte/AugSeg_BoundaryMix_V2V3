# BoundaryMix 6 Experiments Usage

## Configs

- `exps/boundary_mix_v2_v3/voc_semi662/v23_d2_no_qc_gate/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v23_d2_soft_qc_gate_a05/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v23_d2_soft_same_target/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_d2_teacher_feature_gate/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_d2_teacher_relation_consistency/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_d2_affinity_bce/config.yaml`

## Experiment Meaning

- `v23_d2_no_qc_gate`: V2 weights mixed CE, V3-d2 BCR uses only confidence gate `R_a * R_b`.
- `v23_d2_soft_qc_gate_a05`: V2 weights mixed CE, BCR uses soft q gate `q_tilde = 0.5 + 0.5*q_C`.
- `v23_d2_soft_same_target`: V2 weights mixed CE, BCR same pairs regress to detached JS semantic compatibility instead of hard 1.
- `v3_d2_teacher_feature_gate`: standalone V3-d2; detached teacher feature relation gates same/diff losses.
- `v3_d2_teacher_relation_consistency`: standalone V3-d2; student feature relation regresses to detached teacher feature relation.
- `v3_d2_affinity_bce`: standalone V3-d2; raw cosine affinity is trained with BCE against hard same/diff semantic relation labels.

## Field Mapping

- Experiment 1: `relation_mode=base_margin`, `same_loss_mode=hard_one`, `use_component_gate=false`
- Experiment 2: `relation_mode=base_margin`, `same_loss_mode=hard_one`, `use_component_gate=true`, `component_gate_mode=soft`, `component_gate_alpha=0.5`
- Experiment 3: `relation_mode=base_margin`, `same_loss_mode=soft_semantic`, `use_component_gate=false`
- Experiment 4: `relation_mode=teacher_feature_gate`, `use_teacher_features=true`, `use_component_gate=false`
- Experiment 5: `relation_mode=teacher_relation_consistency`, `use_teacher_features=true`, `use_component_gate=false`
- Experiment 6: `relation_mode=affinity_bce`, `use_teacher_features=false`, `use_component_gate=false`

`lambda_bcr` is applied in `train_semi.py`, not inside `compute_js_boundary_compatibility_loss`.

The fourth BCR argument is the confidence source. It accepts `[B,H,W]` max-probability maps, `[B,1,H,W]` confidence maps, or `[B,C,H,W]` teacher probabilities/logits. In `train_semi.py`, `logits_u_aug` is a historical variable name for teacher max probability from `torch.max`, not raw class logits.

When `boundary_compatibility.enabled=false` or `lambda_bcr=0`, BCR is skipped and the BCR feature path is not used. q_C gates BCR only when `boundary_component.enabled=true` and `boundary_compatibility.use_component_gate=true`.

## Train Commands

Use the same launcher and GPU count as the existing V2/V3 runs. Example:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train_semi.py --config exps/boundary_mix_v2_v3/voc_semi662/v23_d2_no_qc_gate/config.yaml
python -m torch.distributed.launch --nproc_per_node=2 train_semi.py --config exps/boundary_mix_v2_v3/voc_semi662/v23_d2_soft_qc_gate_a05/config.yaml
python -m torch.distributed.launch --nproc_per_node=2 train_semi.py --config exps/boundary_mix_v2_v3/voc_semi662/v23_d2_soft_same_target/config.yaml
python -m torch.distributed.launch --nproc_per_node=2 train_semi.py --config exps/boundary_mix_v2_v3/voc_semi662/v3_d2_teacher_feature_gate/config.yaml
python -m torch.distributed.launch --nproc_per_node=2 train_semi.py --config exps/boundary_mix_v2_v3/voc_semi662/v3_d2_teacher_relation_consistency/config.yaml
python -m torch.distributed.launch --nproc_per_node=2 train_semi.py --config exps/boundary_mix_v2_v3/voc_semi662/v3_d2_affinity_bce/config.yaml
```

Teacher feature modes (`v3_d2_teacher_feature_gate` and `v3_d2_teacher_relation_consistency`) run one additional no-grad teacher forward on the mixed strong image. Expect higher runtime and peak memory than base V3-d2.

## Smoke Before Training

Run these in the `augseg-bm` environment before launching training:

```bash
python -m compileall train_semi.py util tools/smoke_boundary_mix_v2v3.py
python tools/smoke_boundary_mix_v2v3.py
```

The implementation audit did not push to GitHub.

## Sanity Checklist

- Dataset remains PASCAL VOC semi-supervised 662 labels.
- Crop remains `321x321`.
- Backbone remains ResNet-101.
- Batch, optimizer, and LR schedule match the source V2/V3-d2 configs.
- `boundary_component.enabled=false` means no V2 CE weighting.
- `boundary_compatibility.enabled=false` or `lambda_bcr=0` skips BCR feature forward and loss.
- Teacher probabilities, teacher features, JS targets, confidence gates, and component gates are detached.

## Reading Logs

- Pair volume: `bcr/num_pairs_candidate`, `bcr/num_pairs_sampled`, `bcr/num_pairs_active`
- Semantic split: `bcr/num_pairs_same`, `bcr/num_pairs_diff`, `bcr/num_pairs_uncertain`
- Student relation: `bcr/mean_s_S`, `bcr/mean_s_S_same`, `bcr/mean_s_S_diff`
- Gate strength: `bcr/mean_r_ab`, `bcr/mean_r_before_component_gate`, `bcr/mean_r_after_component_gate`
- V2 behavior: `v2/mean_q_C`, `v2/min_q_C`, `v2/max_q_C`, `v2/loss_mix_v2`
- Teacher modes: `bcr/mean_s_T`, `bcr/mean_abs_sS_minus_sT_active`
- Affinity BCE: `bcr/mean_A_affinity`, `bcr/mean_y_rel`, `bcr/loss_affinity`

## Baselines To Compare

1. A1 neutral control
2. V2 only
3. V3-d2 base
4. V2+V3 current/direct
5. The six new versions listed above
