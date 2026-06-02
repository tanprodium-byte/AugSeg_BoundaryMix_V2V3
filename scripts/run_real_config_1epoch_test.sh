#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${ROOT}/.codex_smoke/real_configs/voc5_single_gpu_gbs8/v2_component_weighting/config.yaml"
TEST_DIR="${ROOT}/.codex_smoke/real_1epoch_test"
TEST_CONFIG="${TEST_DIR}/config.yaml"
LOG_DIR="${TEST_DIR}/logs"
PORT="${PORT:-53917}"
SEED="${SEED:-2}"

cd "${ROOT}"
mkdir -p "${TEST_DIR}" "${LOG_DIR}"

python - "${BASE_CONFIG}" "${TEST_CONFIG}" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with src.open("r") as f:
    cfg = yaml.safe_load(f)

cfg["trainer"]["epochs"] = 1
cfg.setdefault("hf", {})["enabled"] = False
cfg.setdefault("wandb", {})["enable"] = False
cfg.setdefault("saver", {})["snapshot_dir"] = "./exp_smoke/real_batch8_1epoch_test"

with dst.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(dst)
PY

echo "[gpu] current nvidia-smi"
nvidia-smi
echo "[gpu] compute apps: pid, process_name, used_memory_MiB"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true

set +e
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port="${PORT}" \
  train_semi.py \
  --config="${TEST_CONFIG}" \
  --seed "${SEED}" 2>&1 | tee "${LOG_DIR}/run.log"
rc="${PIPESTATUS[0]}"
set -e

if [[ "${rc}" -eq 0 ]]; then
  echo "PASS real_batch8_1epoch_test" | tee "${TEST_DIR}/result.txt"
  exit 0
fi

if grep -qi "out of memory\\|CUDA out of memory\\|torch.OutOfMemoryError" "${LOG_DIR}/run.log"; then
  echo "OOM real_batch8_1epoch_test" | tee "${TEST_DIR}/result.txt"
else
  echo "FAIL real_batch8_1epoch_test rc=${rc}" | tee "${TEST_DIR}/result.txt"
fi

exit "${rc}"
