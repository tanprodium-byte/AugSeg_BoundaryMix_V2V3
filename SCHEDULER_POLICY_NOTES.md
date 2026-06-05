# Scheduler Policy Notes

## Worker loop semantics

`--sleep-sec 15` or `--sleep-sec 60` is only the polling delay while a worker is idle or busy. It does not mean the worker launches a job every 15 seconds.

When a worker has an assigned training process, it stays inside that train subprocess and only sends heartbeats/lease renewals. It does not poll or claim another job until the subprocess exits and the worker reports DONE or FAILED.

Avoid restarting the worker service while training unless it is necessary. If a stop/restart is required, the worker catches SIGTERM/SIGINT, terminates only the child process it spawned, and reports the active job as interrupted/retryable before exiting.

## One GPU, one scheduler job

Postgres workers must not claim a new job while the database already has a `jobs.status='running'` row with:

- the same `worker_id`, or
- the same `server_name` and `gpu_id`.

This protects a GPU after service restart: if the database still has a running job row, the restarted worker waits instead of launching a second train. It heartbeats `blocked_running_job` when the row still looks healthy, or `blocked_stale_running` when lease/heartbeat evidence suggests the row may be stale.

By default, `AUGSEG_AUTO_REPAIR_STALE_RUNNING=0`; the worker does not reset production running rows automatically. If explicitly set to `1`, the worker may repair only a stale running job for its exact `worker_id`, `server_name`, and `gpu_id`, and only when the worker has no matching current child process and the lease or heartbeat is stale.

## Graceful systemd stop

The Postgres worker service templates use:

```text
KillMode=mixed
TimeoutStopSec=180
```

`KillMode=mixed` sends SIGTERM to the main worker first, giving it a chance to report `interrupted` and return the config to `failed_retryable`. If the service has not stopped after `TimeoutStopSec`, systemd may SIGKILL the control group.

Interrupted jobs are recorded as:

- `jobs.status='failed'`
- `jobs.error='Interrupted by service stop/restart SIGTERM'`
- `configs.status='failed_retryable'`
- `configs.worker_id=NULL`
- `configs.last_error_class='interrupted'`
- `configs.next_retry_at=NOW()`

Interrupted handling does not increment `configs.attempts`.

## OOM policy

OOM is transient resource pressure, not final experiment failure.

Errors containing `OOM`, `out of memory`, `CUDA out of memory`, or `torch.cuda.OutOfMemoryError` are reported as:

- `jobs.status='failed'`
- `jobs.error='OOM'`
- `configs.status='failed_retryable'`
- `configs.worker_id=NULL`
- `configs.last_error='OOM'`
- `configs.last_error_class='oom'`
- `configs.next_retry_at=NOW() + AUGSEG_OOM_COOLDOWN_MINUTES`

OOM does not set `failed_final`, and max attempts do not make an OOM config final. Non-OOM errors still use the normal attempts/max-attempts path and can become `failed_final`.

## returncode=1 bootstrap failures

`returncode=1` is not a root cause by itself. Scheduler failures must include either:

- `log_path=.scheduler_runs/logs/<job_id>.log` plus a tail of the train stdout/stderr, or
- an `error_tail` from `scheduler/run_train_job.py` stdout when the runner fails before creating the train log.

Do not reset `failed_final` configs that only show `returncode=1` until the root cause has been diagnosed and fixed. Common bootstrap causes include stale absolute config paths from another host, missing dependencies, missing data/pretrained files, or invalid temporary config generation.

Diagnose recent failed jobs without updating DB:

```bash
cd /home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3
source /home/islabworker3/tantv/.secrets/augseg_scheduler.env
python scripts/diagnose_recent_failed_jobs.py --limit 20
```

After the bootstrap root cause is fixed, dry-run the `returncode=1` failed-final reset:

```bash
python scripts/reset_returncode1_failed_final_to_retryable.py --dry-run
```

Apply only after reviewing the affected configs and confirming there are no running jobs for them:

```bash
python scripts/reset_returncode1_failed_final_to_retryable.py --apply
```

## Free VRAM gate

Workers now gate on `memory.free`, not `memory.used`.

The worker claims only when:

```text
memory.free MiB >= AUGSEG_REQUIRED_FREE_VRAM_MB
```

Defaults if `AUGSEG_REQUIRED_FREE_VRAM_MB` is missing:

- GPU name contains `5090`: `28000`
- GPU name contains `A6000`: `30000`
- fallback: `28000`

Heartbeat details include `memory.used`, `memory.free`, and `required_free`.

## Repair command

Dry-run OOM `failed_final` repair first:

```bash
cd /home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3
source /home/islabworker3/tantv/.secrets/augseg_scheduler.env
python scripts/reset_oom_failed_final_to_retryable.py --dry-run
```

Apply only after reviewing affected config IDs:

```bash
python scripts/reset_oom_failed_final_to_retryable.py --apply
```

The repair script does not print the DB URL and does not touch configs with a running job.

## Stale running repair

A real running job has both:

- `jobs.status='running'` / `configs.status='running'`
- a live scheduler worker child process responsible for that `job_id` on the recorded `worker_id`, `server_name`, and `gpu_id`

A stale running job is a Postgres row where `jobs.status='running'` and `configs.status='running'`, but no current worker child process is responsible for that job anymore. This can happen after worker restart, SIGKILL, OOM-kill, host reboot, or a child process dying before it reports done/failed/interrupted.

Do not use generic `nvidia-smi` process ownership to decide that a scheduler job is real. A GPU may be occupied by another user's process while the scheduler DB still contains an old running row. The scheduler only treats a row as repairable when ownership matches the current worker/server/GPU and lease/heartbeat evidence is stale.

Read-only diagnosis:

```bash
python scripts/diagnose_running_jobs.py
```

`POSSIBLE_STALE_RUNNING` means the DB has a running job whose worker heartbeat is missing, stale, or no longer `running`, or whose lease has expired. It is a warning, not a process kill decision.

Dry-run the current stale job first:

```bash
cd /home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3
source /home/islabworker3/tantv/.secrets/augseg_scheduler.env
python scripts/reset_stale_running_job_to_retryable.py --config-id v2_component_weighting --worker-id supermaster:gpu0 --dry-run
python scripts/reset_stale_running_job_to_retryable.py --config-id v2_v3_best_template --worker-id islab-server3:gpu0 --dry-run
```

Apply only after reviewing the dry-run output:

```bash
python scripts/reset_stale_running_job_to_retryable.py --config-id v2_component_weighting --worker-id supermaster:gpu0 --apply
python scripts/reset_stale_running_job_to_retryable.py --config-id v2_v3_best_template --worker-id islab-server3:gpu0 --apply
```

After repair, `scheduler/status_postgres.py` will show the config as `failed_retryable`; the scheduler can run it again when GPU free VRAM is high enough.

Auto repair remains off by default to avoid resetting a production job incorrectly. To enable it explicitly for a worker, set:

```text
AUGSEG_AUTO_REPAIR_STALE_RUNNING=1
```

When enabled, repair updates only the selected running row:

- `jobs.status='failed'`
- `jobs.error='Auto-repaired stale running job after worker restart/lease expiry'`
- `configs.status='failed_retryable'`
- `configs.worker_id=NULL`
- `configs.attempts=0`
- `configs.last_error_class='interrupted'`
- `configs.next_retry_at=NOW()`

## Safe deploy order

Use this order so workers get the signal/stale/OOM policy before production repair rows are applied:

1. Commit and push the scheduler/service/docs changes.
2. On A6000: `git pull`, copy `scripts/systemd/augseg-worker-a6000-postgres.service.template`, then restart that worker.
3. On supermaster: `git pull`, copy `scripts/systemd/augseg-worker-5090-postgres.service`, then restart that worker.
4. Dry-run stale running repair.
5. Apply stale running repair.
6. Dry-run OOM `failed_final` repair.
7. Apply OOM `failed_final` repair.
8. Check `python scheduler/status_postgres.py`.

## Deploy: supermaster 5090

Do this later, after the current running job is safe to interrupt/restart at the service level:

```bash
cd /home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3
git pull --ff-only
cp scripts/systemd/augseg-worker-5090-postgres.service ~/.config/systemd/user/augseg-worker-5090-postgres.service
systemctl --user daemon-reload
systemctl --user restart augseg-worker-5090-postgres.service
systemctl --user status augseg-worker-5090-postgres.service --no-pager
```

## Deploy: A6000

On the A6000 host:

```bash
cd /home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3
git pull --ff-only
cp scripts/systemd/augseg-worker-a6000-postgres.service.template ~/.config/systemd/user/augseg-worker-a6000-postgres.service
systemctl --user daemon-reload
systemctl --user restart augseg-worker-a6000-postgres.service
systemctl --user status augseg-worker-a6000-postgres.service --no-pager
```

## Smoke tests

These tests do not run training:

```bash
python -m py_compile scheduler/postgres_db.py scheduler/worker.py scheduler/status_postgres.py
python scripts/test_postgres_scheduler_policy.py
python -m py_compile scheduler/postgres_db.py scheduler/worker.py scheduler/run_train_job.py scheduler/status_postgres.py scripts/diagnose_recent_failed_jobs.py scripts/diagnose_running_jobs.py scripts/reset_returncode1_failed_final_to_retryable.py scripts/reset_stale_running_job_to_retryable.py scripts/test_scheduler_signal_and_stale_policy.py
python scripts/diagnose_recent_failed_jobs.py --limit 10
python scripts/diagnose_running_jobs.py
python scripts/test_scheduler_signal_and_stale_policy.py
```
