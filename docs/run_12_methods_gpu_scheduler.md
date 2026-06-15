# VOC662 12-Method GPU Suite Runner

This is a registry-based suite runner for the 12 VOC662 experiments in `configs/experiment_registry_voc662_12_methods.yaml`. It is intentionally separate from the older Postgres scheduler.

The runner only orchestrates work:

- selects methods from the registry
- checks GPU memory before each method
- creates a per-suite/per-GPU lock
- calls `train_semi.py` with the selected config
- writes per-method logs and JSONL status
- supports resume/skip from status
- parses the final log after each process exits

In full mode it does not own artifacts. It does not disable HF, does not change `hf.path_in_repo`, does not promote `latest`, does not upload job artifacts, and does not change saver or wandb settings. Checkpoint saving and HF upload behavior remain controlled by the original config and `train_semi.py`. If you want HF upload every epoch, configure that in the experiment config/training code, not in this runner.

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
  --min-free-mb 8000
```

Smoke mode is only for runtime-path checking, not research results.

## Full Run

```bash
scripts/run_voc662_12_methods.sh --resume
```

Or explicitly:

```bash
python tools/run_experiment_suite.py \
  --registry configs/experiment_registry_voc662_12_methods.yaml \
  --gpu 0 \
  --mode full \
  --resume \
  --min-free-mb 12000
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
