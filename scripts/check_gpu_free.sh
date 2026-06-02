#!/usr/bin/env bash
set -euo pipefail

REQUIRED_FREE_MIB="${AUGSEG_REQUIRED_FREE_VRAM_MB:-${REQUIRED_FREE_MIB:-28000}}"

nvidia-smi

IFS=',' read -r total used free < <(
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits |
    head -n 1 |
    tr -d ' '
)
echo "memory.total=${total} MiB memory.used=${used} MiB memory.free=${free} MiB required_free=${REQUIRED_FREE_MIB} MiB"

if [[ "${free}" =~ ^[0-9]+$ ]] && (( free >= REQUIRED_FREE_MIB )); then
  echo "PASS"
  exit 0
fi

echo "BUSY"
exit 1
