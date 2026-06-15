# S2/S3 Saliency Component-Box Usage

## Configs

S2:

```bash
exps/boundary_mix_v2_v3/voc_semi662/s2_saliency_component_box_cutmix/config.yaml
```

S3:

```bash
exps/boundary_mix_v2_v3/voc_semi662/s3_saliency_component_box_plus_v3_d2/config.yaml
```

## Smoke Tests

```bash
python -m compileall train_semi.py util tools
python tools/smoke_s1_saliency_cutmix.py
python tools/smoke_s2_s3_saliency_component_box.py
```

## Short Train Smoke

Use temporary configs under `tmp/` with:

- `trainer.epochs: 1`
- `saver.auto_resume: false`
- `checkpoint.auto_resume: false`
- `wandb.enable: false`
- `hf.enabled: false`
- `saver.snapshot_dir` pointing outside the main experiment folders

Example:

```bash
timeout 4m bash -lc 'CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_semi.py --config tmp/s2_saliency_component_box_train_smoke.yaml --seed 0' 2>&1 | tee tmp/s2_saliency_component_box_train_smoke.log
timeout 4m bash -lc 'CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_semi.py --config tmp/s3_saliency_component_box_v3_train_smoke.yaml --seed 0' 2>&1 | tee tmp/s3_saliency_component_box_v3_train_smoke.log
```

If these commands hit the timeout after several iterations, treat that as a timeout smoke result rather than a training crash.

## Notes

- S1 remains `saliency_cutmix.mode: "box"` and uses the original candidate-box path.
- S2/S3 use `saliency_cutmix.mode: "component_box"`.
- S2 keeps `boundary_component.enabled: false` and `boundary_compatibility.enabled: false`.
- S3 keeps `boundary_component.enabled: false`, enables V3-d2 BCR, and keeps `use_component_gate: false`.
