#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${ROOT}/exps/boundary_mix_v2_v3/voc_semi662/v2_component_weighting/config.yaml"
PORT_BASE="${PORT_BASE:-53907}"
SEED="${SEED:-2}"
SMOKE_ROOT="${ROOT}/.codex_smoke"
CONFIG_DIR="${SMOKE_ROOT}/configs"
SPLIT_DIR="${SMOKE_ROOT}/splits/pascal_ladder"
LOG_DIR="${SMOKE_ROOT}/logs"

cd "${ROOT}"
mkdir -p "${CONFIG_DIR}" "${SPLIT_DIR}" "${LOG_DIR}"

echo "[gpu] current nvidia-smi"
nvidia-smi
echo "[gpu] compute apps: pid, process_name, used_memory_MiB"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true

python - "${BASE_CONFIG}" "${CONFIG_DIR}" "${SPLIT_DIR}" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys
import yaml

base = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
split_dir = Path(sys.argv[3])

with base.open("r") as f:
    base_cfg = yaml.safe_load(f)

src_labeled = Path(base_cfg["dataset"]["train"]["data_list"])
src_unlabeled = Path(str(src_labeled).replace("labeled.txt", "unlabeled.txt"))
src_val = Path(base_cfg["dataset"]["val"]["data_list"])

case_specs = {
    "case_a_tiny_gpu": {
        "batch_size": 1,
        "crop": [129, 129],
        "resize_base_size": 160,
    },
    "case_b_light": {
        "batch_size": 1,
        "crop": [257, 257],
        "resize_base_size": 300,
    },
    "case_c_mid": {
        "batch_size": 1,
        "crop": [321, 321],
        "resize_base_size": 400,
    },
    "case_d_original_crop_b1": {
        "batch_size": 1,
        "crop": [513, 513],
        "resize_base_size": 500,
    },
}

def write_head(src: Path, dst: Path, n: int) -> None:
    lines = src.read_text().splitlines()
    if len(lines) < n:
        raise RuntimeError(f"{src} has only {len(lines)} lines, need {n}")
    dst.write_text("\n".join(lines[:n]) + "\n")

for case_name, spec in case_specs.items():
    case_split = split_dir / case_name
    case_split.mkdir(parents=True, exist_ok=True)
    write_head(src_labeled, case_split / "labeled.txt", 4)
    write_head(src_unlabeled, case_split / "unlabeled.txt", 4)
    write_head(src_val, case_split / "val.txt", 2)

    cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
    cfg.setdefault("hf", {})["enabled"] = False
    cfg["hf"]["auto_download"] = False
    cfg["hf"]["auto_upload"] = False
    cfg.setdefault("wandb", {})["enable"] = False
    cfg.setdefault("trainer", {})["epochs"] = 1
    cfg.setdefault("saver", {})["auto_resume"] = False
    cfg.setdefault("checkpoint", {})["auto_resume"] = False
    cfg["saver"]["snapshot_dir"] = f"./exp_smoke/{case_name}"
    cfg["dataset"]["workers"] = 0
    cfg["dataset"]["n_sup"] = 10578
    cfg["dataset"]["train"]["data_list"] = str(case_split / "labeled.txt")
    cfg["dataset"]["val"]["data_list"] = str(case_split / "val.txt")
    cfg["dataset"]["val"]["batch_size"] = 1
    cfg["dataset"]["train"]["batch_size"] = spec["batch_size"]
    cfg["dataset"]["train"]["crop"]["size"] = spec["crop"]
    cfg["dataset"]["train"]["resize_base_size"] = spec["resize_base_size"]

    if "boundary_component" in cfg and "debug_log" in cfg["boundary_component"]:
        cfg["boundary_component"]["debug_log"] = False
    if "boundary_compatibility" in cfg and "debug_log" in cfg["boundary_compatibility"]:
        cfg["boundary_compatibility"]["debug_log"] = False

    out = config_dir / f"{case_name}.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(out)
PY

run_case() {
  local case_name="$1"
  local port="$2"
  local cfg="${CONFIG_DIR}/${case_name}.yaml"
  local log="${LOG_DIR}/${case_name}.log"
  local memlog="${LOG_DIR}/${case_name}_mem.csv"

  echo "[case] ${case_name}"
  echo "timestamp,memory_used_mib" > "${memlog}"
  (
    while true; do
      printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
      sleep 1
    done
  ) >> "${memlog}" &
  local monitor_pid="$!"

  set +e
  CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port="${port}" \
    train_semi.py \
    --config="${cfg}" \
    --seed "${SEED}" 2>&1 | tee "${log}"
  local rc="${PIPESTATUS[0]}"
  set -e

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true

  local peak
  peak="$(awk -F, 'NR > 1 {gsub(/ /, "", $2); if ($2+0 > max) max=$2+0} END {print max+0}' "${memlog}")"
  echo "[case] ${case_name} peak_vram_mib=${peak}"

  if [[ "${rc}" -eq 0 ]]; then
    echo "PASS ${case_name} peak_vram_mib=${peak}" | tee "${LOG_DIR}/ladder_result.txt"
    return 0
  fi

  if grep -qi "out of memory\\|CUDA out of memory\\|torch.OutOfMemoryError" "${log}"; then
    echo "OOM ${case_name} peak_vram_mib=${peak}" | tee -a "${LOG_DIR}/ladder_result.txt"
  else
    echo "FAIL ${case_name} rc=${rc} peak_vram_mib=${peak}" | tee -a "${LOG_DIR}/ladder_result.txt"
  fi
  return 1
}

: > "${LOG_DIR}/ladder_result.txt"
cases=(case_a_tiny_gpu case_b_light case_c_mid case_d_original_crop_b1)
for i in "${!cases[@]}"; do
  case_name="${cases[$i]}"
  port="$((PORT_BASE + i))"
  if run_case "${case_name}" "${port}"; then
    exit 0
  fi
done

echo "No ladder case passed. See ${LOG_DIR}/ladder_result.txt"
exit 1
