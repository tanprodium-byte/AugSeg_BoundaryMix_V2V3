#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

configs=(
  "exps/boundary_mix_v2_v3/voc_semi662/v2_component_weighting/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/v2_v3_best_template/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d1/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d2/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d3/config.yaml"
)

for rel in "${configs[@]}"; do
  path="${ROOT}/${rel}"
  if [[ -f "${path}" ]]; then
    echo "OK ${rel}"
  else
    echo "MISSING ${rel}"
  fi
done
