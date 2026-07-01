# SYSTEM AUDIT - SUPERMASTER

Timestamp UTC: 2026-06-02 09:13:18 UTC

Scope: final read-only audit after A6000 pushed commit `90ac9a5` to `origin/postgres-scheduler`. No train was run. No jobs were claimed/reset. No process was killed. No dataset/pretrained files or original configs were modified.

## Summary

| Area | Result |
|---|---|
| GitHub branch sync | PASS |
| Code compile | PASS |
| Real config verify | PASS |
| Dataset/pretrained | PASS |
| HF access | PASS |
| HF resource tar | PASS |
| Postgres | PASS |
| Heartbeat/cross-server via Postgres/HF | PASS |
| Service 5090 | PASS |
| GPU | BUSY |
| Train | NOT RUN |

Direct LAN between the two servers is no longer a requirement. The main communication requirements are Cloud Postgres for scheduler state/heartbeats and HuggingFace for artifacts/resources.

## A. GitHub / Repo Audit

Commands run:

```text
git branch --show-current
git fetch origin postgres-scheduler
git rev-parse --short HEAD
git rev-parse --short origin/postgres-scheduler
git log --oneline --decorate -5
git status --short
```

Results:

```text
branch: postgres-scheduler
local HEAD: 90ac9a5
origin/postgres-scheduler: 90ac9a5
sync: PASS, local matches origin/postgres-scheduler
```

Recent log:

```text
90ac9a5 (HEAD -> postgres-scheduler, origin/postgres-scheduler) Add A6000 audit and worker service template
b21eceb Add Postgres scheduler backend
808a98c Add Postgres scheduler backend
c117615 (tag: v2v3-audit-pass, origin/boundarymix-v2-v3, origin/HEAD, boundarymix-v2-v3) Wire BoundaryMix V2 and V3 combined mode
4b0a7f2 Implement BoundaryMix V3 JS boundary compatibility
```

Untracked files/directories:

```text
?? .codex_smoke/api_smoke/
?? .codex_smoke/configs/
?? .codex_smoke/hf_access/
?? .codex_smoke/logs/
?? .codex_smoke/real_1epoch_test/
?? .codex_smoke/scheduler_lock_test/
?? .codex_smoke/splits/
?? SYSTEM_AUDIT_SUPERMASTER.md
?? exps/boundary_mix_v2_v3/voc_semi662/_smoke_tmp_config.yaml
?? exps/boundary_mix_v2_v3/voc_semi662/exp_smoke/
```

No merge was needed. No commit or push was made. No data, pretrained, `.cache`, `.secrets`, `.scheduler_runs`, `exp_smoke`, `*.pth`, or `*.tar.gz` files were added.

## B. Code / Config Audit

Required files all exist:

```text
scheduler/postgres_db.py
scheduler/worker.py
scheduler/status_postgres.py
scheduler/run_train_job.py
scheduler/init_voc5_jobs_postgres.py
scripts/check_gpu_free.sh
scripts/check_hf_access.py
scripts/verify_real_configs_preserve_paper_settings.py
scripts/systemd/augseg-worker-5090-postgres.service
scripts/systemd/augseg-worker-a6000-postgres.service.template
docs/SETUP_A6000_POSTGRES_WORKER.md
SYSTEM_AUDIT_A6000.md
SETUP_A6000_STATUS.md
```

Compile command:

```text
python -m py_compile scheduler/postgres_db.py scheduler/worker.py scheduler/status_postgres.py scheduler/run_train_job.py
```

Result: PASS.

Real config verification command:

```text
python scripts/verify_real_configs_preserve_paper_settings.py
```

Result: PASS.

Output ended with:

```text
VERIFY_REAL_CONFIGS: PASS
```

The script only reported allowed smoke changes for batch size, HF toggles, saver snapshot dir, and wandb. The 5 original config files were not modified.

Exact config files added during this audit: none.

Required original configs preserved:

```text
exps/boundary_mix_v2_v3/voc_semi662/v2_component_weighting/config.yaml
exps/boundary_mix_v2_v3/voc_semi662/v2_v3_best_template/config.yaml
exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d1/config.yaml
exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d2/config.yaml
exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d3/config.yaml
```

Disabled-module behavior: this audit made no code changes. Existing verification confirms the real configs preserve paper settings except explicitly allowed smoke overrides. The implementation rule remains that when `boundary_component.enabled=false` and `boundary_compatibility.enabled=false`, the loss path should behave like the existing A1/neutral or baseline path depending on config.

## C. Dataset / Pretrained Audit

Resolved paths:

```text
FOUND /home/jupyter-ytvn/SSS/data/VOC2012
FOUND /home/jupyter-ytvn/SSSS-ADA/data/CPS/pascal_voc/662/labeled.txt
FOUND /home/jupyter-ytvn/SSSS-ADA/data/CPS/pascal_voc/662/unlabeled.txt
FOUND /home/jupyter-ytvn/SSSS-ADA/data/CPS/pascal_voc/val.txt
FOUND /home/jupyter-ytvn/SSS/src/networks/pretrained/resnet101.pth
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
lrwxrwxrwx 1 jupyter-iec2024iot04 jupyter-iec2024iot04 60 Jun  2 00:50 pretrained/resnet101.pth -> /home/jupyter-ytvn/SSS/src/networks/pretrained/resnet101.pth
```

Result: PASS.

## D. HuggingFace Audit

Command:

```text
python scripts/check_hf_access.py
```

Result:

```text
login: tanprodium
repo: access ok tanprodium/augseg-boundarymix-v2v3-runs (model)
uploaded_and_verified: voc5_single_gpu_gbs8/_hf_test/supermaster_test.txt
HF_ACCESS: PASS
```

HF token was not printed.

Resource check using `HfApi().list_repo_files`:

```text
repo_id: tanprodium/augseg-boundarymix-v2v3-runs
repo_type: model
target: resources/augseg_voc2012_splits_pretrained.tar.gz
HF_LIST_REPO_FILES: PASS
RESOURCE_TAR_FOUND: True
```

No resource tar download was performed.

## E. Postgres / Communication Audit

Environment sourced:

```text
~/.secrets/augseg_scheduler.env
```

`AUGSEG_SCHEDULER_DB_URL` was not printed.

Command:

```text
python scheduler/status_postgres.py
```

Status:

```text
config_id | current_epoch | status | worker_id | attempts | last_error | updated_at
------------------------------------------------------------------------------------------------------------------------
v2_component_weighting | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
v2_v3_best_template | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
v3_js_bcr_d1 | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
v3_js_bcr_d2 | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
v3_js_bcr_d3 | 0 | idle |  | 0 |  | 2026-06-02 06:29:28.693431+00:00
```

Worker heartbeat support exists in `scheduler/postgres_db.py`.

Heartbeat query:

```text
worker_id | server_name | gpu_id | status | last_seen | detail
----------------------------------------------------------------------------------------------------
supermaster:gpu0 | supermaster | 0 | busy | 2026-06-02 09:12:26.305729+00:00 | memory.used=25489 MiB
islab-server3:gpu0 | islab-server3 | 0 | busy | 2026-06-02 09:11:58.439402+00:00 | memory.used=33725 MiB
audit-islab-server3 | islab-server3 | 0 | audit_ok | 2026-06-02 08:53:41.049854+00:00 | A6000 audit ran
audit-supermaster | supermaster | -1 | audit_ok | 2026-06-02 08:52:11.217138+00:00 | supermaster audit ran
```

Postgres result: PASS.

Heartbeat/cross-server via Postgres/HF result: PASS.

No jobs were claimed, reset, or modified.

## F. Service 5090 Audit

Commands:

```text
systemctl --user status augseg-worker-5090-postgres.service --no-pager
systemctl --user show augseg-worker-5090-postgres.service -p ActiveState -p SubState -p NRestarts -p ExecMainStatus
journalctl --user -u augseg-worker-5090-postgres.service -n 100 --no-pager
```

Status:

```text
ActiveState=active
SubState=running
NRestarts=0
ExecMainStatus=0
```

Service has been active since:

```text
Tue 2026-06-02 06:30:46 UTC
```

Recent journal:

```text
Jun 02 06:30:46 supermaster systemd[845279]: Started augseg-worker-5090-postgres.service - AugSeg BoundaryMix RTX 5090 Postgres Worker.
```

No new service errors were visible in the last 100 journal lines.

Service 5090 result: PASS.

## G. GPU Audit

Commands:

```text
nvidia-smi
bash scripts/check_gpu_free.sh
```

GPU status:

```text
GPU: NVIDIA GeForce RTX 5090
memory.used: 25489 MiB / 32607 MiB
GPU-Util: 74%
processes:
  PID 1595267 python, 3104 MiB
  PID 1597040 /opt/tljh/user/bin/python3.12, 22368 MiB
```

`scripts/check_gpu_free.sh` result:

```text
memory.used=25489 MiB threshold=4000 MiB
BUSY
```

GPU result: BUSY.

No train was run.

## Smoke Test Command

Smoke test command for future validation only, not run during this audit:

```text
python -m py_compile scheduler/postgres_db.py scheduler/worker.py scheduler/status_postgres.py scheduler/run_train_job.py
python scripts/verify_real_configs_preserve_paper_settings.py
python scripts/check_hf_access.py
set -a; . ~/.secrets/augseg_scheduler.env; set +a; python scheduler/status_postgres.py
bash scripts/check_gpu_free.sh
```

Do not run training as part of this audit smoke path.

## Limitations / Untested Parts

- No training was run, by request.
- No job claim/reset path was exercised, by request.
- GPU was busy, so no train readiness beyond detection was tested.
- HF resource tar existence was checked by listing repo files only; no tar download or extraction was performed.
- Service was inspected only via systemd status/show and last 100 journal lines.

## Next Actions

- Leave scheduler jobs idle until GPU resources are intentionally scheduled.
- Keep using Cloud Postgres for scheduler state/heartbeats and HuggingFace for artifacts/resources.
- Do not start training from this audit session.
