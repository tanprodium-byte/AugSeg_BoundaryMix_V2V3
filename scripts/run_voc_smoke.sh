#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${1:-${ROOT}/exps/boundary_mix_v2_v3/voc_semi662/v2_component_weighting/config.yaml}"
SMOKE_CONFIG="${ROOT}/exps/boundary_mix_v2_v3/voc_semi662/_smoke_tmp_config.yaml"
PORT="${PORT:-53907}"
SEED="${SEED:-2}"

cd "${ROOT}"

python - "${BASE_CONFIG}" "${SMOKE_CONFIG}" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with src.open("r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("trainer", {})["epochs"] = 1
cfg.setdefault("hf", {})["enabled"] = False
cfg["hf"]["auto_download"] = False
cfg["hf"]["auto_upload"] = False
cfg.setdefault("wandb", {})["enable"] = False
cfg.setdefault("saver", {})["auto_resume"] = False
cfg.setdefault("checkpoint", {})["auto_resume"] = False
cfg["saver"]["snapshot_dir"] = "./exp_smoke/voc5_1epoch"

with dst.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(dst)
PY

torchrun \
  --nproc_per_node=1 \
  --master_port="${PORT}" \
  train_semi.py \
  --config="${SMOKE_CONFIG}" \
  --seed="${SEED}"
