# S2/S3 Saliency Component-Box Implementation Audit

## Scope

Implemented only:

- S2: saliency-guided component-box CutMix from labeled GT components.
- S3: S2 plus existing V3-d2 JS Boundary Compatibility Regularization.

Not implemented or enabled:

- V2 component weighting is disabled in both S2 and S3 configs.
- CSL is disabled in both S2 and S3 configs.
- Class-aware CutMix and unrelated SSL changes were not added.

## Code Paths

- `util/saliency_cutmix.py`
  - `connected_components_from_gt`
  - `bbox_from_mask`
  - `expand_box`
  - `get_saliency_component_guided_boxes`
- `train_semi.py`
  - Keeps `saliency_cutmix.mode: "box"` on the existing S1 path.
  - Adds `saliency_cutmix.mode: "component_box"` for S2/S3.
  - Passes selected labeled-source boxes into `cut_mix_label_adaptive_with_mask`.
  - S3 reuses the existing `mix_source_mask` returned by CutMix for BCR.

## Component Convention

- Components are extracted from labeled GT only.
- Ignore label `255` is skipped.
- With `foreground_only: true`, class `0` is skipped.
- Connectivity supports 4 or 8; S2/S3 configs use 8.
- Area filter uses inclusive component area thresholds:
  - `min_component_area: 64`
  - `max_component_area: 20000`
- Box convention is `[x1, y1, x2, y2]`, where `x` indexes height/rows and `y` indexes width/columns.
- `x2` and `y2` are exclusive, matching the existing CutMix slicing convention.

## Selection

- Teacher saliency uses supervised CE on labeled source images.
- Teacher parameters are temporarily frozen and restored.
- Saliency tensors are detached after per-image normalization.
- Component score is `mean_saliency * clipped_area_score`.
- Selection uses softmax sampling with `temperature: 0.2`.
- If no valid component exists, the implementation falls back to the existing random box sampler.

## S3 / V3-d2

S3 enables existing BCR with:

- `boundary_compatibility.enabled: true`
- `relation_mode: "base_margin"`
- `semantic_metric: "js"`
- `band_width: 3`
- `pair_radius: 2`
- `lambda_bcr: 0.01`
- `use_confidence_gate: true`
- `use_component_gate: false`

Because `boundary_component.enabled: false`, S3 standalone does not pass V2 `q_C` into BCR.

## Logging

Added saliency component fields:

- `saliency/num_components`
- `saliency/num_valid_components`
- `saliency/selected_component_class`
- `saliency/selected_component_area`
- `saliency/selected_component_score`
- `saliency/component_score_mean`
- `saliency/component_score_max`
- `saliency/component_box_area`
- `saliency/selection_entropy`
- `saliency/fallback_ratio`

`saliency/mode` numeric convention:

- `0`: disabled
- `1`: S1 box mode
- `2`: S2/S3 component-box mode

## Validation

Required commands:

```bash
python -m compileall train_semi.py util tools
python tools/smoke_s1_saliency_cutmix.py
python tools/smoke_s2_s3_saliency_component_box.py
```

No performance improvements are claimed without training results.
