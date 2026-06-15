# VOC662 12-Method GPU Suite Runner

This is a registry-based suite runner for the 12 VOC662 experiments in `configs/experiment_registry_voc662_12_methods.yaml`. It is intentionally separate from the older Postgres scheduler.

The runner only orchestrates work:

- selects methods from the registry
- checks GPU memory before each method
- creates a per-suite/per-GPU lock for local runs
- optionally uses a shared Postgres queue for multi-server runs
- calls `train_semi.py` with the selected config
- writes per-method logs and JSONL status
- supports resume/skip from status
- parses the final log after each process exits

In full mode it does not own artifacts. It does not disable HF, does not change `hf.path_in_repo`, does not promote `latest`, does not upload job artifacts, and does not change saver or wandb settings. Checkpoint saving and HF upload behavior remain controlled by the original config and `train_semi.py`. If you want HF upload every epoch, configure that in the experiment config/training code, not in this runner.

Do not use the older Postgres scheduler for this suite if that path overrides HF settings, uploads job bundles, or promotes `latest.tar.gz`. The queue backend here is orchestration-only.

## Python And Launcher

Use the Python interpreter from the intended training environment. On the A6000 server, use:

```bash
/home/islabworker3/tantv/envs/augseg-bm/bin/python
```

The safest launcher is `--launcher python-module`, which runs:

```bash
sys.executable -m torch.distributed.run --standalone ...
```

This avoids accidentally using `torchrun` from the base shell. `--launcher auto` is the default; it uses `torchrun` only when `torchrun` is from the same environment as `sys.executable`, otherwise it falls back to `sys.executable -m torch.distributed.run`. `--launcher torchrun` is available for explicit torchrun use and prints a warning if PATH points outside the active Python environment.

## Phase 1: Local Runner

The default backend is local:

```bash
python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend local \
  --gpu 0 \
  --resume
```

Local mode runs selected methods sequentially on one server. It uses local JSONL status and a local lock file.

## Phase 2: Multi-Server Postgres Queue

The Postgres backend uses a shared table, `experiment_suite_queue`, as a queue/status store. It does not manage checkpoints or HF artifacts.

Atomic claim uses a transaction with `SELECT ... FOR UPDATE SKIP LOCKED`, followed by an `UPDATE ... RETURNING`. That prevents `supermaster` and `islab-server3` from claiming the same method. Before claiming a new method, the same transaction also checks for an active `running` row with the same `worker_id` or the same `server_name` plus `gpu_id`. If one exists, the worker prints `worker already has running job ...` and does not claim another method for that GPU. Rows with `success` or `running` are not claimed. Failed rows are retried only with `--retry-failed` and while `retries < max_retries`.

Heartbeat is written while `train_semi.py` runs. If a running row has no fresh heartbeat for `--max-stale-minutes`, another worker can mark it `failed_stale`. The runner does not kill the old process; stale handling only updates queue state.

### Init Queue

On `supermaster`:

```bash
source ~/.secrets/augseg_scheduler.env

python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id supermaster:gpu0 \
  --server-name supermaster \
  --gpu 0 \
  --nproc-per-node 1 \
  --min-free-mb 17000 \
  --launcher python-module \
  --init-queue \
  --status
```

`--init-queue` creates the table if needed and seeds the 12 methods. Existing `success` or `running` rows are not reset unless `--force` is explicitly provided.

### Worker: RTX 5090

```bash
source ~/.secrets/augseg_scheduler.env

python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id supermaster:gpu0 \
  --server-name supermaster \
  --gpu 0 \
  --nproc-per-node 1 \
  --min-free-mb 17000 \
  --launcher python-module \
  --loop \
  --resume
```

### Worker: RTX A6000

```bash
source /home/islabworker3/tantv/.secrets/augseg_scheduler.env

/home/islabworker3/tantv/envs/augseg-bm/bin/python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id islab-server3:gpu0 \
  --server-name islab-server3 \
  --gpu 0 \
  --nproc-per-node 1 \
  --min-free-mb 17000 \
  --launcher python-module \
  --loop \
  --resume
```

### Queue Status

```bash
python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --status
```

To show only running rows, including worker, server, GPU, pid, heartbeat, and log path:

```bash
python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --status-running
```

For A6000 status checks, use the same interpreter:

```bash
source /home/islabworker3/tantv/.secrets/augseg_scheduler.env

/home/islabworker3/tantv/envs/augseg-bm/bin/python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --status
```

## Preflight

Preflight does not run training and does not claim queue rows unless `--claim-test` is also used. It loads the registry, validates 12 methods and full-mode crop/global batch, prints Python/torch/CUDA details, checks `nvidia-smi`, checks GPU free memory, checks Postgres status when requested, and prints the resolved launcher command.

On `supermaster`:

```bash
source ~/.secrets/augseg_scheduler.env

python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id supermaster:gpu0 \
  --server-name supermaster \
  --gpu 0 \
  --nproc-per-node 1 \
  --min-free-mb 17000 \
  --launcher python-module \
  --preflight
```

On the A6000 server:

```bash
source /home/islabworker3/tantv/.secrets/augseg_scheduler.env

/home/islabworker3/tantv/envs/augseg-bm/bin/python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id islab-server3:gpu0 \
  --server-name islab-server3 \
  --gpu 0 \
  --nproc-per-node 1 \
  --min-free-mb 17000 \
  --launcher python-module \
  --preflight
```

## Claim Test

`--claim-test` is only allowed with `--mode smoke` or `--mode dry-run`; it is rejected for `full`. It atomically claims one Postgres row, writes heartbeat metadata, then releases that row back to `pending`. It does not launch training.

```bash
/home/islabworker3/tantv/envs/augseg-bm/bin/python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode smoke \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id islab-server3:gpu0 \
  --server-name islab-server3 \
  --gpu 0 \
  --launcher python-module \
  --init-queue \
  --claim-test
```

### Stop Safely

Use `Ctrl-C` or terminate the runner process. The child `torchrun` receives termination through the runner process group handling. If a worker or host dies unexpectedly, the queue row remains `running` until another worker marks it `failed_stale` after `--max-stale-minutes`.

### Retry Failed

Failed rows are left as final states by default. To retry failed, OOM, traceback, timeout, GPU-busy, or stale rows:

```bash
python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode full \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id supermaster:gpu0 \
  --server-name supermaster \
  --gpu 0 \
  --retry-failed \
  --max-retries 1 \
  --once
```

## Methods

The registry contains:

- `v23_d2_no_qc_gate`
- `v23_d2_soft_qc_gate_a05`
- `v23_d2_soft_same_target`
- `v3_d2_affinity_bce`
- `v3_d2_teacher_feature_gate`
- `v3_d2_teacher_relation_consistency`
- `s1_saliency_box_cutmix`
- `s2_saliency_component_box_cutmix`
- `s3_saliency_component_box_plus_v3_d2`
- `c1_csl_pseudo_selection`
- `c2_csl_random_reliable_masking`
- `c3_csl_guided_cutmix_plus_v3_d2`

CSL in this repo is an entropy-margin proxy / CSL-style reliability interface, not an official CSL backend.

## Full-Mode Invariants

Full mode requires every selected config to preserve:

- `dataset.train.crop.size: [321, 321]`
- global batch size `dataset.train.batch_size * nproc_per_node == 8`

For `--nproc-per-node 1`, each full config must have `dataset.train.batch_size: 8`. For `--nproc-per-node 2`, each full config must have `dataset.train.batch_size: 4`. The runner fails early if this does not match. It does not silently edit full configs.

## Dry Run

Print the exact commands without running training:

```bash
python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --gpu 0 \
  --mode dry-run
```

Dry-run validates registry paths and full-mode crop/global batch.

## Smoke

Smoke mode writes temporary configs under `tmp/suite_smoke_configs/`, leaves original configs unchanged, sets `trainer.epochs=1`, disables HF and wandb in the temp config, and writes smoke checkpoints under `tmp/suite_smoke_runs/`.

```bash
python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --gpu 0 \
  --mode smoke \
  --only s1_saliency_box_cutmix,c1_csl_pseudo_selection \
  --timeout-sec 180 \
  --lowmem-batch-size 2 \
  --min-free-mb 8000 \
  --launcher python-module
```

Smoke mode is only for runtime-path checking, not research results.

To test one real launcher path through Postgres without starting full training, run smoke once with a short timeout only when GPU 0 is free:

```bash
source /home/islabworker3/tantv/.secrets/augseg_scheduler.env

/home/islabworker3/tantv/envs/augseg-bm/bin/python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --mode smoke \
  --queue-backend postgres \
  --db-url-env AUGSEG_SCHEDULER_DB_URL \
  --worker-id islab-server3:gpu0 \
  --server-name islab-server3 \
  --gpu 0 \
  --nproc-per-node 1 \
  --min-free-mb 17000 \
  --lowmem-batch-size 2 \
  --timeout-sec 180 \
  --launcher python-module \
  --once \
  --resume
```

## Full Run

```bash
PYTHON=/home/islabworker3/tantv/envs/augseg-bm/bin/python scripts/run_voc662_12_methods.sh --resume --launcher python-module
```

Or explicitly:

```bash
python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --gpu 0 \
  --mode full \
  --resume \
  --min-free-mb 12000 \
  --launcher python-module
```

Do not run the full suite while developing unless you intend to start real training.

## Selection

Run one or more methods:

```bash
python tools/run_experiment_suite.py --mode dry-run --only s1_saliency_box_cutmix,c1_csl_pseudo_selection
```

Run a group:

```bash
python tools/run_experiment_suite.py --mode dry-run --group boundarymix
```

Skip methods:

```bash
python tools/run_experiment_suite.py --mode full --resume --skip v23_d2_no_qc_gate
```

Groups are `boundarymix`, `saliency`, `saliency_v3`, `csl`, and `csl_v3`.

## Resume, Logs, And Status

Status is appended to:

```text
runs/suite_status/voc662_12_methods/status.jsonl
```

Per-method logs are written under:

```text
runs/suite_logs/voc662_12_methods/<timestamp>/<method>.log
```

Use `--resume` to skip methods whose latest status is `success`. Use `--force` to run again even after success.

## GPU Busy Or OOM

Before each method the runner checks:

- GPU memory used/total/utilization
- active compute processes and their memory usage

If free VRAM is below `--min-free-mb`, it waits `--poll-sec` seconds and checks again. With `--no-wait`, it records `gpu_busy` and continues. It never kills GPU processes.

After each method exits, logs are parsed for CUDA OOM, `Traceback`, `NaN`/`nan`, `SignalException`, `RuntimeError`, and `KeyboardInterrupt`. OOM is classified as `failed_oom`; tracebacks are classified as `failed_traceback`.

## Lock

The suite lock is:

```text
runs/locks/voc662_12_methods_gpu0.lock
```

Use `--force-lock` only when you know the lock is stale.

## Other GPUs And Multi-Process

Use another GPU:

```bash
GPU=1 scripts/run_voc662_12_methods.sh --resume
```

Use another process count only when the original config batch size still gives global batch 8:

```bash
python tools/run_experiment_suite.py --mode full --nproc-per-node 2
```

With the current registry configs, `nproc_per_node=1` is the full-mode setting because `dataset.train.batch_size=8`.

## Outputs

Do not commit generated logs, status, locks, temporary smoke configs, temporary smoke runs, checkpoints, or compressed model artifacts.
