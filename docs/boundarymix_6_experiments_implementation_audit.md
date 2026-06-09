# BoundaryMix 6 Experiments Implementation Audit

Branch audited: `feature/boundarymix-6-experiments`.

This audit covers the six BoundaryMix V2/V3 experiment configs added under
`exps/boundary_mix_v2_v3/voc_semi662/`. No GitHub push was performed.

## Train Loop Conclusions

- `compute_js_boundary_compatibility_loss(...)` returns raw BCR loss.
- `lambda_bcr` is applied exactly once in `train_semi.py` when adding BCR to total loss.
- `boundary_compatibility_enabled` is true only when `boundary_compatibility.enabled=true` and `lambda_bcr != 0`.
- When BCR is disabled or `lambda_bcr=0`, the train loop skips BCR computation and does not request student or teacher feature forwards for BCR.
- V2 CE weighting is controlled only by `boundary_component.enabled`.
- q_C/component weights are passed into BCR only when both `boundary_component.enabled=true` and `boundary_compatibility.use_component_gate=true`.
- V3 standalone configs set `boundary_component.enabled=false`, so BCR uses no q_C gate.

## Confidence/Logits Parameter

The fourth BCR argument is a confidence source, not the BCR semantic target.

Accepted shapes:

- `[B,H,W]` max-probability/confidence map.
- `[B,1,H,W]` confidence map.
- `[B,C,H,W]` teacher probabilities or raw teacher logits.

For `[B,C,H,W]`, the implementation now computes the max probability for the confidence gate. Probability tensors are detected by class sums near 1; otherwise softmax is applied as raw logits. In `train_semi.py`, the variable named `logits_u_aug` is the max probability returned by `torch.max(pred_u, dim=1)`, shape `[B,H,W]`.

## Teacher Feature Modes

- Teacher features are forwarded only when BCR is active and either `use_teacher_features=true` or `relation_mode` is `teacher_feature_gate` / `teacher_relation_consistency`.
- The teacher feature forward runs under `torch.no_grad()`.
- Teacher decoder features are detached before BCR.
- Student decoder features remain differentiable.
- Teacher feature source is currently restricted to `mixed`, aligned with the mixed strong image used for student BCR.
- Teacher feature modes add an extra no-grad teacher forward and hold additional decoder features, so they can increase runtime and peak memory.

## Loss Mode Checks

- `base_margin + hard_one`: same pairs use `(1 - s_S)^2`; diff pairs use `relu(s_S - margin)^2`.
- `base_margin + soft_semantic`: same pairs use detached semantic compatibility target; diff pairs keep margin loss.
- `teacher_feature_gate`: teacher feature relation is detached and gates same/diff losses.
- `teacher_relation_consistency`: active same/diff pairs regress student relation to detached teacher relation.
- `affinity_bce`: uses raw cosine through sigmoid temperature and BCE; hard target uses same=1, diff=0.
- JS probabilities are clamped and normalized by `log(2)`.
- No-active-pair path returns finite graph-safe zero on the feature device/dtype.

## Config Audit

Added configs:

- `exps/boundary_mix_v2_v3/voc_semi662/v23_d2_no_qc_gate/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v23_d2_soft_qc_gate_a05/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v23_d2_soft_same_target/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_d2_teacher_feature_gate/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_d2_teacher_relation_consistency/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_d2_affinity_bce/config.yaml`

All six keep VOC 662, crop `321x321`, ResNet-101, batch settings, optimizer, and LR schedule from the source V2/V3-d2 configs. All V3 configs use `pair_radius=2`, `band_width=3`, and `lambda_bcr=0.01`.

## Validation

Smoke commands to run before training:

```bash
python -m compileall train_semi.py util tools/smoke_boundary_mix_v2v3.py
python tools/smoke_boundary_mix_v2v3.py
```

The smoke test covers V2 component weighting, V3 JS/BCR, no q_C standalone behavior, component gate modes, teacher feature modes, affinity BCE, no-active-pair behavior, disabled/lambda-zero skip logic, and 4D teacher logits as the BCR confidence source.

Audit validation result:

- `python -m compileall train_semi.py util tools/smoke_boundary_mix_v2v3.py`: pass.
- `python tools/smoke_boundary_mix_v2v3.py`: pass.

## Limitations

- No full training was run during this audit.
- No performance improvement is claimed without training results.
- Teacher feature modes may require human review for GPU memory headroom on the target hardware.
