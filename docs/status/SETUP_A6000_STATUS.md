# A6000 Postgres Worker Setup Status

- Date: 2026-06-02
- Server/user: islab-server3 / islabworker3
- Repo: `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3`
- Branch: `postgres-scheduler`
- Commit: `b21eceb`

## Environment

- Conda env path: `/home/islabworker3/tantv/envs/augseg-bm`
- HF_HOME: `/home/islabworker3/tantv/.cache/huggingface`
- Torch: `2.11.0+cu128`
- Torch CUDA: `12.8`
- CUDA available: `True`
- GPU 0: `NVIDIA RTX A6000`
- CUDA smoke matmul: PASS

## Resource Tar

- HuggingFace repo: `tanprodium/augseg-boundarymix-v2v3-runs`
- Repo type: `model`
- Filename: `resources/augseg_voc2012_splits_pretrained.tar.gz`
- Downloaded path: `/home/islabworker3/tantv/.cache/huggingface/hub/models--tanprodium--augseg-boundarymix-v2v3-runs/snapshots/2cec3ab179368c14ae23ecb89d82b88600d0d5d1/resources/augseg_voc2012_splits_pretrained.tar.gz`
- Download cache under HF_HOME: PASS
- Extract destination: `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3`
- Extract status: PASS, 64309 tar members extracted after path safety checks

## Dataset / Pretrained

- Required dataset/pretrained verify: PASS
- `data/VOC2012`: PASS
- `data/splitsall/pascal_u2pl/662/labeled.txt`: PASS, 662 lines
- `data/splitsall/pascal_u2pl/662/unlabeled.txt`: PASS, 9920 lines
- `data/splitsall/pascal_u2pl/val.txt`: PASS, 1449 lines
- `pretrained/resnet101.pth`: PASS, 179202351 bytes
- Sample structure check: PASS for first 5 labeled ids using `data/VOC2012/JPEGImages/<id>.jpg` and `data/VOC2012/SegmentationClassAug/<id>.png`
- Pretrained verification: PASS via `scripts/prepare_resnet101_pretrained.py`
- Pretrained load notes: 532 state_dict keys, 0 missing keys, 2 unexpected keys (`fc.weight`, `fc.bias`)

## Real Configs

- Real single-GPU config directory: `.codex_smoke/real_configs/voc5_single_gpu_gbs8`
- Verification: PASS via `scripts/verify_real_configs_preserve_paper_settings.py`
- Original config files were not modified.
- Required real configs present:
  - `exps/boundary_mix_v2_v3/voc_semi662/v2_component_weighting/config.yaml`
  - `exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d1/config.yaml`
  - `exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d2/config.yaml`
  - `exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d3/config.yaml`
  - `exps/boundary_mix_v2_v3/voc_semi662/v2_v3_best_template/config.yaml`

## HF Access

- Status: PASS
- Check command: `HF_HOME=/home/islabworker3/tantv/.cache/huggingface conda run -p /home/islabworker3/tantv/envs/augseg-bm python scripts/check_hf_access.py`
- Verified repo access and upload/download test path: `voc5_single_gpu_gbs8/_hf_test/supermaster_test.txt`

## Postgres

- Scheduler py_compile: PASS for `scheduler/postgres_db.py`, `scheduler/worker.py`, `scheduler/status_postgres.py`
- Status command: FAIL
- Failure reason: sourced `AUGSEG_SCHEDULER_DB_URL` is present but contains placeholder host text `HOST`, causing psycopg host resolution failure.
- DB URL was not printed.
- No jobs were reset.
- Lock tests were not run.

## GPU Gate

- Command: `bash scripts/check_gpu_free.sh`
- Status: BUSY
- Observed memory: 33725 MiB used, threshold 4000 MiB
- Training was not started.

## Systemd

- Service template path: `scripts/systemd/augseg-worker-a6000-postgres.service.template`
- Template status: created in repo
- Systemd install status: NOT INSTALLED
- Reason: strict `/home/islabworker3/tantv`-only rule; nothing was copied to `~/.config/systemd/user`.
- Service enable/start status: NOT RUN

## Files Changed / Added

- Added `scripts/systemd/augseg-worker-a6000-postgres.service.template`
- Updated `SETUP_A6000_STATUS.md`
- HuggingFace access check created `.codex_smoke/hf_access/supermaster_test.txt`
- Resource extraction populated ignored dataset/pretrained paths under repo: `data/` and `pretrained/`
