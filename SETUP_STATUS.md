# SETUP_STATUS

- Repo path: `/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3`
- Branch: `boundarymix-v2-v3`
- Commit: `c1176153100ad72ab06e9092956ab1f1ac5f26ea`
- Conda env: `augseg-bm` at `/home/jupyter-iec2024iot04/.conda/envs/augseg-bm`
- Python in env: `3.11`
- Torch version: `2.11.0+cu128`
- CUDA runtime reported by torch: `12.8`
- GPU: `NVIDIA GeForce RTX 5090`
- GPU check: `nvidia-smi` works; last observed memory was `25489 MiB / 32607 MiB` used.

## Configs

All 5 Pascal VOC configs exist:

- `exps/boundary_mix_v2_v3/voc_semi662/v2_component_weighting/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v2_v3_best_template/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d1/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d2/config.yaml`
- `exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d3/config.yaml`

Helper script:

- `scripts/list_voc5_configs.sh`

## Dataset And Splits

- Config path `./data/VOC2012`: present as symlink to `/home/jupyter-ytvn/SSS/data/VOC2012`
- Real VOC2012 path: `/home/jupyter-ytvn/SSS/data/VOC2012`
- Safety policy: external dataset/pretrained paths under `/home/jupyter-ytvn/` are read-only for this setup. They are only read through existing symlinks; no files outside this repo were intentionally written, chmodded, chowned, touched, deleted, or modified.
- `./data/splitsall/pascal_u2pl/662/labeled.txt`: present, symlink-resolves to `/home/jupyter-ytvn/SSSS-ADA/data/CPS/pascal_voc/662/labeled.txt`, 662 lines
- `./data/splitsall/pascal_u2pl/662/unlabeled.txt`: present, symlink-resolves to `/home/jupyter-ytvn/SSSS-ADA/data/CPS/pascal_voc/662/unlabeled.txt`, 9920 lines
- `./data/splitsall/pascal_u2pl/val.txt`: present, symlink-resolves to `/home/jupyter-ytvn/SSSS-ADA/data/CPS/pascal_voc/val.txt`, 1449 lines

## Pretrained

- `./pretrained/resnet101.pth`: present as symlink to `/home/jupyter-ytvn/SSS/src/networks/pretrained/resnet101.pth`
- Repo code loads this file with `torch.load(path)` then `model.load_state_dict(state_dict, strict=False)`.
- Verification script: `scripts/prepare_resnet101_pretrained.py`
- Verification result: `missing_keys=0`, `unexpected_keys=2` (`fc.weight`, `fc.bias`)

## Torch CUDA Test

Passed:

- `import torch`
- `torch.__version__`: `2.11.0+cu128`
- `torch.version.cuda`: `12.8`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.get_device_name(0)`: `NVIDIA GeForce RTX 5090`
- Small CUDA matrix multiplication passed.

## Smoke Test

- Script: `scripts/run_voc_smoke.sh`
- Base config: `exps/boundary_mix_v2_v3/voc_semi662/v2_component_weighting/config.yaml`
- Temporary config: `exps/boundary_mix_v2_v3/voc_semi662/_smoke_tmp_config.yaml`
- Temporary config changes: `hf.enabled=false`, `hf.auto_download=false`, `hf.auto_upload=false`, `wandb.enable=false`, `trainer.epochs=1`, `auto_resume=false`
- Launch: `torchrun --nproc_per_node=1 --master_port=53907 train_semi.py --config=<tmp> --seed=2`
- Result: fail due to CUDA OOM, after dataset/pretrained/model load and after training started.
- OOM detail: PyTorch tried to allocate `66.00 MiB`; GPU had only `51.31 MiB` free at failure time. Other processes were using about `3.03 GiB` and `21.84 GiB`; this smoke process had about `6.41 GiB`.
- Checkpoints: no checkpoint was created before the OOM; only smoke `config.yaml` and `run_id.txt` were generated under `exps/boundary_mix_v2_v3/voc_semi662/exp_smoke/voc5_1epoch/`.

## Smoke Ladder

- Script: `scripts/run_voc_smoke_ladder.sh`
- Base config: `exps/boundary_mix_v2_v3/voc_semi662/v2_component_weighting/config.yaml`
- Temporary configs: `.codex_smoke/configs/`
- Temporary smoke splits: `.codex_smoke/splits/pascal_ladder/`
- Logs: `.codex_smoke/logs/`
- HF upload disabled: `hf.enabled=false`, `hf.auto_download=false`, `hf.auto_upload=false`
- W&B disabled: `wandb.enable=false`
- Full 80 epoch not run.
- GPU processes observed before ladder:
  - PID `1595267`, `python`, `3104 MiB`
  - PID `1597040`, `/opt/tljh/user/bin/python3.12`, `22368 MiB`
- Case A `case_a_tiny_gpu`: PASS. It completed 4 training iterations and mini validation with crop `[129,129]`, batch size `1`, workers `0`, and strong augmentation disabled because this repo asserts `1 <= strong_aug.num_augs <= 11`.
- Case B/C/D: not run because ladder stops after first PASS.
- Peak VRAM: observed monitor stdout peak `28374 MiB` total GPU used during Case A. The first ladder script run had a memlog redirect bug and wrote `peak_vram_mib=0`; the script has been fixed for future runs, and `.codex_smoke/logs/ladder_result.txt` records the observed stdout peak.
- Smoke checkpoint/output path: `.codex_smoke/configs/exp_smoke/case_a_tiny_gpu/`
- Research-use note: the earlier Case A smoke run that disabled `strong_aug` was only a technical GPU/runtime test and must not be used as a research result or paper-setting experiment.
- Current ladder script/configs have been corrected to preserve base `dataset.train.strong_aug` (`num_augs=3`, `flag_use_random_num_sampling=true`) and still only reduce technical smoke fields such as batch size, crop size, workers, validation batch size, temporary split size, and logging/upload controls.

## Single GPU Global Batch 8 Configs

Created only; not launched:

- `.codex_smoke/real_configs/voc5_single_gpu_gbs8/v2_component_weighting/config.yaml`
- `.codex_smoke/real_configs/voc5_single_gpu_gbs8/v2_v3_best_template/config.yaml`
- `.codex_smoke/real_configs/voc5_single_gpu_gbs8/v3_js_bcr_d1/config.yaml`
- `.codex_smoke/real_configs/voc5_single_gpu_gbs8/v3_js_bcr_d2/config.yaml`
- `.codex_smoke/real_configs/voc5_single_gpu_gbs8/v3_js_bcr_d3/config.yaml`

These configs set `dataset.train.batch_size=8`, `dataset.val.batch_size=1`, keep `trainer.epochs=80`, disable HF and W&B temporarily, and use per-version `saver.snapshot_dir=./exp_smoke/real_single_gpu_gbs8/<version>`.

Verification:

- Script: `scripts/verify_real_configs_preserve_paper_settings.py`
- Log: `.codex_smoke/logs/verify_real_configs_preserve_paper_settings.log`
- Result: PASS
- The verifier compared each generated real config against its original config and allowed only `dataset.train.batch_size`, `dataset.val.batch_size`, `hf`, `wandb`, `saver.snapshot_dir`, optional `saver.auto_resume`, and optional `checkpoint.auto_resume`.
- Verified unchanged paper-critical settings include `dataset.train.strong_aug`, `dataset.train.crop`, `dataset.train.resize_base_size`, optimizer, scheduler, epochs, unsupervised trainer settings, boundary modules, and model/net settings.

## Real Batch 8 One Epoch Test

- Script: `scripts/run_real_config_1epoch_test.sh`
- Source config: `.codex_smoke/real_configs/voc5_single_gpu_gbs8/v2_component_weighting/config.yaml`
- Temporary config: `.codex_smoke/real_1epoch_test/config.yaml`
- Log: `.codex_smoke/real_1epoch_test/logs/run.log`
- Result: OOM
- No automatic reduction of batch size, crop size, resize, or `strong_aug` was performed.
- Temporary config preserved real batch-8 settings: `dataset.train.batch_size=8`, crop `[513,513]`, `resize_base_size=500`, and `strong_aug.num_augs=3`.
- Only changed fields for the 1-epoch test: `trainer.epochs=1`, `hf.enabled=false`, `wandb.enable=false`, and `saver.snapshot_dir=./exp_smoke/real_batch8_1epoch_test`.
- OOM detail: `torch.OutOfMemoryError`, tried to allocate `262.00 MiB`; GPU had only `141.31 MiB` free. Other processes were using about `3.03 GiB` and `21.84 GiB`; this process had about `6.32 GiB`.

## Scheduler MVP

Created MVP files:

- `scripts/check_gpu_free.sh`
- `scripts/check_hf_access.py`
- `scripts/scheduler_status.sh`
- `scripts/test_scheduler_lock.sh`
- `scripts/run_one_real_segment_when_free.sh`
- `scheduler/hf_settings.yaml`
- `scheduler/config.py`
- `scheduler/db.py`
- `scheduler/init_voc5_jobs.py`
- `scheduler/coordinator.py`
- `scheduler/worker.py`
- `scheduler/run_train_job.py`
- `scheduler/status.py`

Database:

- Path: `scheduler/state.sqlite`
- Initialized 5 VOC configs, all `idle`, `current_epoch=0`, `max_epoch=80`, `step_epoch=20`.
- First real segment assigned by the scheduler will be `0 -> 20` epochs for one config.

HuggingFace:

- Settings file: `scheduler/hf_settings.yaml`
- Repo configured: `tanprodium/augseg-boundarymix-v2v3-runs`
- `huggingface_hub` installed in conda env `augseg-bm`.
- HF auth whoami result: PASS, logged in as `tanprodium`.
- HF access check result: FAIL because repo `tanprodium/augseg-boundarymix-v2v3-runs` is inaccessible/not found via the HuggingFace API.
- No checkpoint/artifact upload was attempted beyond the access script; no checkpoint was uploaded.
- To enable upload, create the repo or grant the logged-in account access, then rerun `python scripts/check_hf_access.py`.

Scheduler lock test:

- Command: `bash scripts/test_scheduler_lock.sh`
- Result: PASS
- Two fake concurrent workers were assigned different `config_id` values, proving the SQLite `BEGIN IMMEDIATE` lock path does not hand out the same config twice.

GPU gate:

- Command: `bash scripts/check_gpu_free.sh`
- Result: BUSY
- Observed GPU memory: `25489 MiB / 32607 MiB` used, above the `4000 MiB` threshold.
- Observed processes: PID `1595267` using about `3104 MiB`; PID `1597040` using about `22368 MiB`.

Real segment command:

- `bash scripts/run_one_real_segment_when_free.sh`
- This script checks GPU free first, initializes DB with `--no-reset-if-exists`, then runs one scheduler worker once: `python scheduler/worker.py --server-name supermaster --gpu-id 0 --coordinator-db scheduler/state.sqlite --once`.
- Not run for training in this setup pass because GPU is currently BUSY. No 20-epoch or 80-epoch real train segment was launched.

## Latest Gate Run - 2026-06-02 01:34 UTC

- `hf auth whoami`: PASS, user `tanprodium`.
- `python scripts/check_hf_access.py`: FAIL, repo `tanprodium/augseg-boundarymix-v2v3-runs` inaccessible/not found.
- `bash scripts/test_scheduler_lock.sh`: PASS, fake concurrent workers received different configs: `v2_component_weighting` and `v2_v3_best_template`.
- `python scheduler/status.py`: PASS, all five jobs idle at `current_epoch=0`.
- `bash scripts/check_gpu_free.sh`: BUSY, `25489 MiB / 32607 MiB` used, above the `4000 MiB` threshold.
- Real segment: not run because GPU gate was BUSY.
- Segment metadata: no `config_id`, `job_id`, `from_epoch`, `to_epoch`, or HF artifact path was produced.

## Production-Lite Systemd Setup - 2026-06-02 01:50 UTC

Systemd user service files created:

- `scripts/systemd/augseg-coordinator.service`
- `scripts/systemd/augseg-worker-5090.service`

Operational scripts created:

- `scripts/install_systemd_user_services.sh`
- `scripts/check_services.sh`
- `scripts/follow_worker_logs.sh`
- `scripts/follow_coordinator_logs.sh`

Remote worker documentation created:

- `docs/SETUP_A6000_SYSTEMD_WORKER.md`

Coordinator API:

- Script: `scheduler/coordinator_api.py`
- Local health URL: `http://127.0.0.1:8787/health`
- Service binds `0.0.0.0:8787` so A6000 workers can call `http://<supermaster-ip>:8787`.
- Endpoints implemented: `GET /health`, `GET /status`, `POST /request_job`, `POST /heartbeat`, `POST /report_done`, `POST /report_failed`.
- Job assignment uses SQLite `BEGIN IMMEDIATE`, selects only `idle` or `failed_retryable`, prioritizes lowest `current_epoch`, updates config to `running` in the same transaction, creates a unique `job_id`, and sets `worker_id` plus `lease_until`.

Worker loop:

- Script: `scheduler/worker.py`
- Supermaster service command:
  `python scheduler/worker.py --server-name supermaster --gpu-id 0 --coordinator-url http://127.0.0.1:8787 --loop --sleep-sec 60`
- With `--coordinator-url`, worker does not write SQLite directly. It uses coordinator API calls for request/done/failed/heartbeat.
- Worker checks GPU memory before requesting a job. If GPU is busy, it sleeps and retries. If no job is available, it sleeps and retries.
- Worker uploads segment artifacts through the existing HF artifact path after a successful segment, then reports completion to the coordinator.

Install/start commands:

- `bash scripts/install_systemd_user_services.sh`
- `systemctl --user enable augseg-coordinator.service`
- `systemctl --user enable augseg-worker-5090.service`
- `systemctl --user start augseg-coordinator.service`
- `systemctl --user start augseg-worker-5090.service`
- `systemctl --user status augseg-coordinator.service --no-pager`
- `systemctl --user status augseg-worker-5090.service --no-pager`

Install and start in one step:

- `bash scripts/install_systemd_user_services.sh --enable-now`

Health/log commands:

- `bash scripts/check_services.sh`
- `bash scripts/follow_coordinator_logs.sh`
- `bash scripts/follow_worker_logs.sh`

Verification results:

- `python -m py_compile scheduler/coordinator_api.py scheduler/worker.py scheduler/status.py scheduler/db.py`: PASS
- Coordinator API smoke test with a temporary `.codex_smoke/api_smoke/state.sqlite`: PASS for `/health`, `/request_job`, `/heartbeat`, and `/report_done`.
- `python scripts/check_hf_access.py`: PASS, repo access/upload/download verified for `tanprodium/augseg-boundarymix-v2v3-runs`.
- `bash scripts/test_scheduler_lock.sh`: PASS, fake concurrent workers received different configs: `v2_component_weighting` and `v2_v3_best_template`.
- `python scheduler/status.py`: PASS, read-only status showed all five configs idle at `current_epoch=0`.
- `bash scripts/check_gpu_free.sh`: BUSY, `25489 MiB / 32607 MiB` used, above the `4000 MiB` threshold.

No training segment was launched because the GPU gate was BUSY. When the worker service is running, it will keep watching the GPU and only request/run a job after the GPU is below threshold.

## Cloud Postgres Multi-Server Scheduler - 2026-06-02

Strategy changed from coordinator-over-LAN to Cloud Postgres because supermaster and islab-server3 do not have a working LAN route to each other, and the user does not have sudo for Tailscale setup.

New production model:

- No supermaster-to-islab-server3 routing is required.
- Every worker connects directly to the same Cloud Postgres database through `AUGSEG_SCHEDULER_DB_URL`.
- Postgres is the only production state and lock backend.
- HuggingFace remains the artifact store.
- Workers check local GPU memory before claiming work.
- Claiming uses a Postgres transaction with `FOR UPDATE SKIP LOCKED` to prevent duplicate config assignment.
- Existing SQLite/coordinator code is kept for local development only.

Postgres files created:

- `scheduler/postgres_db.py`
- `scheduler/init_voc5_jobs_postgres.py`
- `scheduler/status_postgres.py`
- `scripts/scheduler_status_postgres.sh`
- `scripts/test_postgres_scheduler_lock.sh`
- `scripts/systemd/augseg-worker-5090-postgres.service`
- `scripts/install_postgres_worker_systemd_user.sh`
- `docs/SETUP_A6000_POSTGRES_WORKER.md`

Install worker systemd Postgres command:

- `bash scripts/install_postgres_worker_systemd_user.sh`
- Optional start after verifying secrets and GPU state: `bash scripts/install_postgres_worker_systemd_user.sh --enable-now`

A6000 worker setup doc:

- `docs/SETUP_A6000_POSTGRES_WORKER.md`

Postgres lock test:

- Command: `bash scripts/test_postgres_scheduler_lock.sh`
- Result: PASS
- Two concurrent test workers received different test configs: `test_pg_lock_0` and `test_pg_lock_1`.
- Cleanup only targets rows with prefix `test_pg_lock_`; production config rows are not deleted.

Verification status:

- `python -m py_compile scheduler/postgres_db.py scheduler/worker.py scheduler/status_postgres.py scheduler/init_voc5_jobs_postgres.py`: PASS
- `bash -n scripts/scheduler_status_postgres.sh scripts/test_postgres_scheduler_lock.sh scripts/install_postgres_worker_systemd_user.sh`: PASS
- `python scheduler/init_voc5_jobs_postgres.py`: PASS, initialized/updated five VOC configs without epoch reset.
- `python scheduler/status_postgres.py`: PASS, five VOC configs idle at `current_epoch=0`.
- `bash scripts/test_postgres_scheduler_lock.sh`: PASS, concurrent claims received different test configs.
- `python scripts/check_hf_access.py`: PASS, repo access/upload/download verified.
- `bash scripts/check_gpu_free.sh`: BUSY, `25489 MiB / 32607 MiB` used, above the `4000 MiB` threshold.

No training segment should be launched during the Postgres migration, and workers must not run train while GPU is BUSY.
