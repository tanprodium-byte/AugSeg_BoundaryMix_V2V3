#!/usr/bin/env bash
set -Eeuo pipefail
set +x
readonly PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PY="${AUGSEG_PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -B "$PACKAGE/controller.py" check-package
"$PY" -B "$PACKAGE/tests/run_isolated.py" "$PACKAGE/tests/test_operations.py"
"$PY" -B "$PACKAGE/tests/run_isolated.py" "$PACKAGE/tests/test_inputs.py"
"$PY" -B "$PACKAGE/tests/run_isolated.py" "$PACKAGE/tests/test_science_s1.py"
"$PY" -B "$PACKAGE/tests/run_isolated.py" "$PACKAGE/tests/test_science_s2.py"
"$PY" -B "$PACKAGE/tests/check_syntax.py"
bash -n "$PACKAGE/run.sh"
bash -n "$PACKAGE/check.sh"
