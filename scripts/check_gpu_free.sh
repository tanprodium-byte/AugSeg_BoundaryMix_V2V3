#!/usr/bin/env bash
set -euo pipefail

THRESHOLD_MIB="${THRESHOLD_MIB:-4000}"

nvidia-smi

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
echo "memory.used=${used} MiB threshold=${THRESHOLD_MIB} MiB"

if [[ "${used}" =~ ^[0-9]+$ ]] && (( used < THRESHOLD_MIB )); then
  echo "PASS"
  exit 0
fi

echo "BUSY"
exit 1
