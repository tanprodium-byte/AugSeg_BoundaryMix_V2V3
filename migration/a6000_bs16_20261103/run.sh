#!/usr/bin/env bash
set -Eeuo pipefail
set +x
umask 077
readonly PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PY="${AUGSEG_PYTHON:-/home/islabworker2/minhhb/tanvt/envs/augseg-bs16/bin/python}"
case "${1:-}" in
 check-package|check-inputs|bootstrap) ;;
 upgrade-input-paths|preflight|launch|finalize|verify-done)
  readonly AUTH="${AUGSEG_AUTH_FILE:-/home/islabworker2/minhhb/tanvt/augseg_auth/credentials.env}"
  [[ -f "$AUTH" && ! -L "$AUTH" && $(stat -c %a "$AUTH") == 600 && $(stat -c %u "$AUTH") == $(id -u) ]] || { echo 'Authentication file must exist, owned by caller, mode600; never stored in Git.' >&2; exit 2; }
  set -a; source "$AUTH"; set +a ;;
 *) echo 'Usage: run.sh {check-package|check-inputs|bootstrap|upgrade-input-paths|preflight|launch|finalize|verify-done} [--method METHOD]' >&2; exit 2 ;;
esac
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME
export WANDB_MODE=online WANDB_BASE_URL=https://api.wandb.ai WANDB_ANONYMOUS=never
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0 HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0
exec "$PY" -B "$PACKAGE/controller.py" "$@"
