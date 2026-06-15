# CSL C1/C2/C3 Implementation Audit

## Backend status

This repo does not include an official CSL backend. The current implementation uses a clean CSL interface in `util/csl_reliability.py` with a replaceable proxy backend:

- `reliability_mode: entropy_margin`
- input: teacher probabilities `[B, C, H, W]`
- output: detached reliability `[B, H, W]` in `[0, 1]`

The proxy combines teacher confidence, normalized low entropy, and top-1/top-2 margin. It is intended as a backend placeholder and can be replaced without changing the training-loop call sites.

## Training integration

CSL reliability is computed under `torch.no_grad()` immediately after the teacher weak-view forward pass and before teacher probabilities are deleted. Reliability is detached and used only as a weight or CutMix selection signal.

C1 and C2 use `csl.use_csl_for_ce_weight=true`. In this mode the unsupervised CE path does not apply the hard confidence threshold. The mixed CE weight is:

- labeled source pixels: `1`
- pseudo/unlabeled pixels: CSL reliability, optionally after random reliable masking for C2

C3 uses `csl.use_csl_for_cutmix=true` and `csl.use_csl_for_ce_weight=false`. CSL only chooses the target paste box. The unsupervised CE remains the baseline confidence-threshold path, and V3-d2 BCR is enabled.

## Separation checks

- V2 is disabled in C1, C2, and C3 via `boundary_component.enabled=false`.
- Saliency CutMix is disabled in C1, C2, and C3 via `saliency_cutmix.enabled=false`.
- V3/BCR is disabled in C1 and C2 via `boundary_compatibility.enabled=false` and `lambda_bcr=0.0`.
- V3/BCR is enabled only in C3 with `pair_radius=2`, `lambda_bcr=0.01`, and `use_component_gate=false`.
- When `csl.enabled=false`, all CSL branches are skipped and existing baseline/S1/S2/S3 paths stay on their prior config flags.

## Files touched

- `train_semi.py`
- `util/boundary_mix.py`
- `util/csl_reliability.py`
- `util/csl_cutmix.py`
- `tools/smoke_csl_variants.py`
- `exps/boundary_mix_v2_v3/voc_semi662/c1_csl_pseudo_selection/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/c2_csl_random_reliable_masking/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/c3_csl_guided_cutmix_plus_v3_d2/config.yaml`

## Limitations

No performance improvement is claimed. The CSL backend is a proxy until an official CSL implementation is provided.
