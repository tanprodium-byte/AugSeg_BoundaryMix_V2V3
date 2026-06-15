#!/usr/bin/env bash
set -e

tport=53907
ngpu=${NPROC_PER_NODE:-1}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${CONFIG:-"$ROOT/exps/boundary_mix_v2_v3/voc_semi662/s1_saliency_box_cutmix/config.yaml"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "python=$(command -v python)"
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

if command -v torchrun >/dev/null 2>&1; then
  launcher=(torchrun)
else
  launcher=(python -m torch.distributed.run)
fi

"${launcher[@]}" --standalone --nproc_per_node="${ngpu}" --master_port="${tport}" \
  "$ROOT/train_semi.py" \
  --config="${CONFIG}" \
  --seed 2 --port "${tport}"
