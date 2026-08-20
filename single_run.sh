#!/usr/bin/env bash
set -e

ENV_FILE="/home/jupyter-iec2024iot04/.secrets/augseg_scheduler.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

cd "$(dirname "$0")"

PORT=${PORT:-53947}
CONFIG="exps/boundary_mix_v2_v3/voc_semi662/c4_csl_official_direct_labeled_guided_cutmix_plus_ce_weight/config.yaml"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
conda run --no-capture-output -n augseg-bm \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_semi.py \
  --config="$CONFIG" \
  --seed 2 \
  --port "$PORT"