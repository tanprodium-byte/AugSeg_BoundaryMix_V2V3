#!/usr/bin/env bash
set -euo pipefail
DIAG_CODE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DIAG_PYTHON=${AUGSEG_PYTHON:-/home/islabworker2/minhhb/tanvt/envs/augseg-bs16/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
case "${1:-}" in
  check-package|check-inputs|check-deps|test|prepare)
    exec "$DIAG_PYTHON" -B "$DIAG_CODE/portable.py" "$@" ;;
  preflight|run)
    exec "$DIAG_PYTHON" -B "$DIAG_CODE/runner.py" "$@" ;;
  analyze)
    shift
    exec "$DIAG_PYTHON" -B "$DIAG_CODE/analyze_replay.py" "$@" ;;
  *) echo 'Usage: run.sh {check-package|check-inputs|check-deps|test|prepare|preflight|run|analyze}' >&2; exit 2 ;;
esac
