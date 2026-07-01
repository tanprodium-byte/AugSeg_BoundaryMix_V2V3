#!/usr/bin/env bash
set -euo pipefail

cd "/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3"
source "/home/jupyter-iec2024iot04/.secrets/augseg_scheduler.env"

export AUGSEG_WANDB_ENABLE=1
export AUGSEG_WANDB_PROJECT="${AUGSEG_WANDB_PROJECT:-augseg-voc662}"
export AUGSEG_WANDB_GROUP="${AUGSEG_WANDB_GROUP:-voc662_12_methods_segments_20_40_60_80}"

bash "run_voc12_5090_segments.sh"
