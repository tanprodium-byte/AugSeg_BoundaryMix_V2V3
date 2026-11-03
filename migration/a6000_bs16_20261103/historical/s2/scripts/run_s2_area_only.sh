#!/usr/bin/env bash
# Explicit manual commands only: no service, queue, polling, or auto-resume.
set -Eeuo pipefail
set +x
umask 077
readonly ROOT="/home/jupyter-iec2024iot04/augseg_deployments/s2_mask_relocated_area_only_bs16_20261030"
readonly PY="/home/jupyter-iec2024iot04/.conda/envs/augseg-bm/bin/python"
readonly AUTH="/home/jupyter-iec2024iot04/.secrets/augseg_scheduler.env"
case "${1:-}" in
  check|paths) ;;
  launch|finalize) ;;
  *) echo "Usage: bash scripts/run_s2_area_only.sh {check|paths|launch|finalize} [arguments]" >&2; exit 2 ;;
esac
# Same credential injection as the verified Q/R controller. Never print it.
if [[ "${1}" == launch || "${1}" == finalize || " $* " == *" --remote "* ]]; then
  [[ -f "$AUTH" && -r "$AUTH" ]] || { echo "Existing authentication injection unavailable" >&2; exit 2; }
  set -a
  source "$AUTH"
  set +a
fi
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME
export PATH="/home/jupyter-iec2024iot04/.conda/envs/augseg-bm/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1
export GIT_OPTIONAL_LOCKS=0 CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online WANDB_BASE_URL=https://api.wandb.ai WANDB_ANONYMOUS=never
export PYTHONPATH="$ROOT"
cd "$ROOT"
exec "$PY" -B tools/s2_area_only_cohort.py "$@"
