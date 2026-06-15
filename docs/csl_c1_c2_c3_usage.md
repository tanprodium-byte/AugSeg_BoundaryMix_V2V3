# CSL C1/C2/C3 Usage

## Configs

- C1: `exps/boundary_mix_v2_v3/voc_semi662/c1_csl_pseudo_selection/config.yaml`
- C2: `exps/boundary_mix_v2_v3/voc_semi662/c2_csl_random_reliable_masking/config.yaml`
- C3-clean: `exps/boundary_mix_v2_v3/voc_semi662/c3_csl_guided_cutmix_plus_v3_d2/config.yaml`

## Behavior

C1 keeps random baseline CutMix and uses CSL reliability as a soft CE weight for pseudo pixels. Labeled mixed pixels keep weight `1`.

C2 is C1 plus random masking on reliable pseudo pixels. Labeled mixed pixels are not masked.

C3-clean uses CSL only to choose low-reliability target CutMix boxes. CE still uses the baseline confidence threshold. V3-d2 BCR is enabled. V2 and saliency are disabled.

## Smoke Tests

Run:

```bash
python -m compileall train_semi.py util tools
python tools/smoke_s1_saliency_cutmix.py
python tools/smoke_s2_s3_saliency_component_box.py
python tools/smoke_csl_variants.py
```

## Real-Train Smoke

Use temporary configs under `tmp/` and keep runs short:

```bash
timeout 4m bash -lc 'CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_semi.py --config tmp/c1_csl_pseudo_selection_train_smoke.yaml --seed 0' 2>&1 | tee tmp/c1_csl_pseudo_selection_train_smoke.log
timeout 4m bash -lc 'CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_semi.py --config tmp/c2_csl_random_reliable_masking_train_smoke.yaml --seed 0' 2>&1 | tee tmp/c2_csl_random_reliable_masking_train_smoke.log
timeout 4m bash -lc 'CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_semi.py --config tmp/c3_csl_guided_cutmix_v3_train_smoke.yaml --seed 0' 2>&1 | tee tmp/c3_csl_guided_cutmix_v3_train_smoke.log
```

Timeout after several iterations should be treated as a timeout, not a crash, unless the log contains OOM, NaN, or a traceback.
