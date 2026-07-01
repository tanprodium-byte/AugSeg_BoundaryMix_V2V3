# SYSTEM_AUDIT_A6000

## Summary

- Timestamp UTC: 2026-06-02T09:03:43Z
- Hostname: islab-server3
- User: islabworker3
- Repo: /home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3
- Branch: postgres-scheduler
- Commit: b21eceb
- Git status: untracked `.codex_smoke/hf_access/`, `SETUP_A6000_STATUS.md`, `SYSTEM_AUDIT_A6000.md`, `scripts/systemd/augseg-worker-a6000-postgres.service.template`
- Training started: no
- Jobs claimed/reset: no
- Processes killed: no

## Required Status

- Code compile: PASS
- CUDA: PASS
- Real config verify: PASS
- Dataset/pretrained: PASS
- HuggingFace: PASS
- HF resource tar: PASS
- Postgres: PASS
- Service: PASS
- GPU: BUSY
- GitHub branch completeness: PASS
- Direct server-to-server LAN required: no
- Cross-server communication via Postgres/HF: PASS

## Code Audit

Required files/directories:

```text
OK scheduler/postgres_db.py
OK scheduler/worker.py
OK scheduler/status_postgres.py
OK scheduler/run_train_job.py
OK scripts/check_gpu_free.sh
OK scripts/check_hf_access.py
OK scripts/verify_real_configs_preserve_paper_settings.py
OK .codex_smoke/real_configs/voc5_single_gpu_gbs8
```

Compile command:

```bash
conda run -p /home/islabworker3/tantv/envs/augseg-bm python -m py_compile scheduler/postgres_db.py scheduler/worker.py scheduler/status_postgres.py scheduler/run_train_job.py
```

Result: PASS

## CUDA Audit

Command used the requested conda environment:

```text
torch: 2.11.0+cu128
torch.version.cuda: 12.8
torch.cuda.is_available(): True
GPU: NVIDIA RTX A6000
CUDA matmul smoke value: 0.04949413612484932
```

Result: PASS

## Real Config Audit

Command:

```bash
conda run -p /home/islabworker3/tantv/envs/augseg-bm python scripts/verify_real_configs_preserve_paper_settings.py
```

Result: PASS

Verified configs:

```text
v2_component_weighting
v2_v3_best_template
v3_js_bcr_d1
v3_js_bcr_d2
v3_js_bcr_d3
VERIFY_REAL_CONFIGS: PASS
```

Only allowed smoke differences were reported: batch size, HF enable/auto flags, saver snapshot dir, and wandb enable. No original config files were modified.

## Dataset And Pretrained Audit

Required artifacts:

```text
OK data/VOC2012
OK data/splitsall/pascal_u2pl/662/labeled.txt
OK data/splitsall/pascal_u2pl/662/unlabeled.txt
OK data/splitsall/pascal_u2pl/val.txt
OK pretrained/resnet101.pth
```

Split counts:

```text
   662 data/splitsall/pascal_u2pl/662/labeled.txt
  9920 data/splitsall/pascal_u2pl/662/unlabeled.txt
  1449 data/splitsall/pascal_u2pl/val.txt
 12031 total
```

Pretrained file:

```text
-rw-r--r-- 1 islabworker3 islabworker3 171M Nov  1  2025 pretrained/resnet101.pth
```

Result: PASS

## HuggingFace Audit

Command:

```bash
HF_HOME=/home/islabworker3/tantv/.cache/huggingface conda run -p /home/islabworker3/tantv/envs/augseg-bm python scripts/check_hf_access.py
```

Output summary:

```text
login: tanprodium
repo: access ok tanprodium/augseg-boundarymix-v2v3-runs (model)
uploaded_and_verified: voc5_single_gpu_gbs8/_hf_test/supermaster_test.txt
HF_ACCESS: PASS
```

Required resource file:

```text
FOUND resources/augseg_voc2012_splits_pretrained.tar.gz
```

Resource re-check command used direct env Python, not heredoc through `conda run`:

```bash
HF_HOME=/home/islabworker3/tantv/.cache/huggingface /home/islabworker3/tantv/envs/augseg-bm/bin/python -c 'from huggingface_hub import HfApi; repo_id="tanprodium/augseg-boundarymix-v2v3-runs"; target="resources/augseg_voc2012_splits_pretrained.tar.gz"; files=HfApi().list_repo_files(repo_id=repo_id, repo_type="model"); print("HF_RESOURCE_TAR_FOUND:", target in files)'
```

Output summary:

```text
HF_LIST_REPO_FILES: PASS
HF_RESOURCE_TAR_FOUND: True
HF_FILE_COUNT: 3
```

Result: PASS. Previous HF PARTIAL was a false negative; the resource tar exists on HuggingFace. No tar was downloaded during this audit.

## Postgres Audit

DB URL was sourced from `/home/islabworker3/tantv/.secrets/augseg_scheduler.env` and was not printed.

Scheduler status:

```text
config_id | current_epoch | status | worker_id | attempts | last_error | updated_at
------------------------------------------------------------------------------------------------------------------------
v2_component_weighting | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
v2_v3_best_template | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
v3_js_bcr_d1 | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
v3_js_bcr_d2 | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
v3_js_bcr_d3 | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
```

Audit heartbeat written:

```text
audit-islab-server3 | islab-server3 | gpu=0 | audit_ok | 2026-06-02 08:53:41.049854+00:00 | A6000 audit ran
```

Recent heartbeat rows observed:

```text
islab-server3:gpu0 | islab-server3 | gpu=0 | busy | 2026-06-02 09:03:15.594423+00:00 | memory.used=33725 MiB
supermaster:gpu0 | supermaster | gpu=0 | busy | 2026-06-02 09:02:37.676403+00:00 | memory.used=25489 MiB
audit-supermaster | supermaster | gpu=-1 | audit_ok | 2026-06-02 08:52:11.217138+00:00 | supermaster audit ran
```

Result: PASS

## GitHub Branch Audit

Branch state:

```text
branch: postgres-scheduler
local HEAD: b21eceb
origin/postgres-scheduler: b21eceb
remote: https://github.com/tanprodium-byte/AugSeg_BoundaryMix_V2V3.git
```

Local-vs-origin diff:

```text
origin/postgres-scheduler..HEAD: none
HEAD..origin/postgres-scheduler: none
```

Required files present on `origin/postgres-scheduler`:

```text
PASS scheduler/postgres_db.py
PASS scheduler/worker.py
PASS scheduler/status_postgres.py
PASS scheduler/init_voc5_jobs_postgres.py
PASS scheduler/run_train_job.py
PASS scripts/test_postgres_scheduler_lock.sh
PASS scripts/check_hf_access.py
PASS scripts/check_gpu_free.sh
PASS scripts/systemd/augseg-worker-5090-postgres.service
PASS docs/SETUP_A6000_POSTGRES_WORKER.md
PASS .codex_smoke/real_configs/voc5_single_gpu_gbs8/v2_component_weighting/config.yaml
PASS .codex_smoke/real_configs/voc5_single_gpu_gbs8/v2_v3_best_template/config.yaml
PASS .codex_smoke/real_configs/voc5_single_gpu_gbs8/v3_js_bcr_d1/config.yaml
PASS .codex_smoke/real_configs/voc5_single_gpu_gbs8/v3_js_bcr_d2/config.yaml
PASS .codex_smoke/real_configs/voc5_single_gpu_gbs8/v3_js_bcr_d3/config.yaml
```

Local files safe to save to GitHub if not already present:

```text
SYSTEM_AUDIT_A6000.md
SETUP_A6000_STATUS.md
scripts/systemd/augseg-worker-a6000-postgres.service.template
```

Files intentionally not committed:

```text
.codex_smoke/hf_access/
```

Result: PASS for required GitHub branch completeness. Local audit/setup files still need a local commit and push after secret scan.

## Service Audit

Service:

```text
augseg-worker-a6000-postgres.service
```

Current status:

```text
Active: active (running) since Tue 2026-06-02 08:43:28 UTC
Main PID: 4087337 (conda)
NRestarts=0
ExecMainStatus=0
ActiveState=active
SubState=running
```

Journal conclusion:

- Current service is active/running.
- `NRestarts=0`.
- No new runtime error was observed after the current 08:43 UTC start.
- Earlier `conda: command not found` messages from 08:36-08:41 UTC are old errors before the current active service start and are not counted as current failures.

Result: PASS

## GPU Audit

`nvidia-smi` summary:

```text
NVIDIA-SMI 535.309.01
Driver Version: 535.309.01
CUDA Version: 12.2
GPU 0: NVIDIA RTX A6000
Memory-Usage: 33725MiB / 49140MiB
GPU-Util: 0%
```

Processes reported by `nvidia-smi`:

```text
PID 3012397: ...naconda3/envs/t_yolo/bin/python3.11, 15998MiB
PID 3206511: python, 2378MiB
PID 3475669: python, 15334MiB
```

`scripts/check_gpu_free.sh`:

```text
memory.used=33725 MiB threshold=4000 MiB
BUSY
```

Result: BUSY. No action was taken and no training was started.

## Cross-Server Communication Audit

Current architecture uses Cloud Postgres and HuggingFace. Direct LAN/ping is not a requirement for the current setup.

Communication checks:

- A6000 can read Cloud Postgres scheduler status: PASS
- A6000 HuggingFace access check: PASS
- Supermaster heartbeat observed in Cloud Postgres: PASS

Result: PASS

## Disabled/Unchanged Training Logic

This audit did not modify training code, scheduler code, or any experiment config. The five original config files under `exps/boundary_mix_v2_v3/voc_semi662/*/config.yaml` were not edited. Baseline/A1 behavior preservation is unchanged from the repository state; this audit only verified setup and scheduler connectivity.

## Smoke Test Command

Safe smoke checks used in this audit:

```bash
conda run -p /home/islabworker3/tantv/envs/augseg-bm python -m py_compile scheduler/postgres_db.py scheduler/worker.py scheduler/status_postgres.py scheduler/run_train_job.py
conda run -p /home/islabworker3/tantv/envs/augseg-bm python scripts/verify_real_configs_preserve_paper_settings.py
HF_HOME=/home/islabworker3/tantv/.cache/huggingface conda run -p /home/islabworker3/tantv/envs/augseg-bm python scripts/check_hf_access.py
bash scripts/check_gpu_free.sh
```

These commands do not start training.

## Limitations And Untested Parts

- No training was run by request.
- No job was claimed, reset, or advanced.
- No process was killed.
- GPU is currently BUSY, so a free-GPU worker run was not tested.
- HF resource tar exists in the HuggingFace model repo. It was not downloaded during this audit.

## Next Actions

- Wait for GPU memory usage to fall below the scheduler threshold before allowing worker training jobs.
- Keep monitoring `worker_heartbeats` in Cloud Postgres to confirm both A6000 and supermaster remain visible.

## Final Summary

- A6000 service: PASS
- GPU: BUSY
- Postgres: PASS
- HF access: PASS
- HF resource tar: PASS
- Dataset/pretrained: PASS
- GitHub branch completeness: PASS
- Local files needing push: `SYSTEM_AUDIT_A6000.md`, `SETUP_A6000_STATUS.md`, `scripts/systemd/augseg-worker-a6000-postgres.service.template`
- Direct server-to-server LAN required: no
- Communication via Postgres/HF: PASS

## HF resource re-check

Re-checked from A6000 using direct env Python:

- repo_id: tanprodium/augseg-boundarymix-v2v3-runs
- repo_type: model
- target: resources/augseg_voc2012_splits_pretrained.tar.gz
- FOUND: True
- exit_code: 0

Conclusion:
- HF access: PASS
- HF resource tar: PASS
- Previous HF PARTIAL was a false negative.
- Local dataset/pretrained had already been extracted and verified PASS.
