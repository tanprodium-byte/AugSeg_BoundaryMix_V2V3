# Scheduler Policy Notes

## Worker loop semantics

`--sleep-sec 15` or `--sleep-sec 60` is only the polling delay while a worker is idle or busy. It does not mean the worker launches a job every 15 seconds.

When a worker has an assigned training process, it stays inside that train subprocess and only sends heartbeats/lease renewals. It does not poll or claim another job until the subprocess exits and the worker reports DONE or FAILED.

## One GPU, one scheduler job

Postgres workers must not claim a new job while the database already has a `jobs.status='running'` row with:

- the same `worker_id`, or
- the same `server_name` and `gpu_id`.

This protects a GPU after service restart: if the train process is still alive and the database still has the job running, the restarted worker heartbeats `busy_running_job` and waits instead of launching a second train.

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

Dry-run first:

```bash
cd /home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3
source ~/.secrets/augseg_scheduler.env
python scripts/reset_oom_failed_final_to_retryable.py
```

Apply only after reviewing affected config IDs:

```bash
python scripts/reset_oom_failed_final_to_retryable.py --apply
```

The repair script does not print the DB URL and does not touch configs with a running job.

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
```
