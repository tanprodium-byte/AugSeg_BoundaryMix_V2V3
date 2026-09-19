#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3_legacy_c321_rerun01"
BRANCH="experiment/legacy-c321-rerun01"
PARENT="bbe2224c50db2c277e5d76dbb64f6527489d5b5c"
BASE="exps/boundary_mix_v2_v3/voc_semi662_legacy_c321_rtx5090"
METHODS=(
  baseline_augseg_fair80_rerun01
  v3_js_bcr_d2
  v3_js_bcr_d1
  v3_js_bcr_d3
  v3_d2_affinity_bce
  v3_d2_teacher_feature_gate
  v3_d2_teacher_relation_consistency
  s2_saliency_component_box_cutmix
  s3_saliency_component_box_plus_v3_d2
  s2_saliency_component_mask_direct_cutmix
  s3_saliency_component_mask_direct_plus_v3_d2
  c3_csl_guided_cutmix_plus_v3_d2
  c3_csl_official_guided_cutmix_plus_v3_d2
)
SUFFIX="legacy_c321_rerun01_rtx5090_r101_c321_bs8x1_gbs8_seed2_attempt01"
if [[ "${1:-}" == "--dry-run" ]]; then
  for method in "${METHODS[@]}"; do printf '%s/%s/config.yaml\n' "$BASE" "$method"; done
  exit 0
fi
run_sequence() {
  for method in "${METHODS[@]}"; do
    run_method "$method"
  done
}
if [[ "${1:-}" == "--self-test-failure" ]]; then
  run_method() { echo "attempted: $1"; return 23; }
  run_sequence
  exit 99
fi
[[ "$#" -eq 0 ]] || { echo "usage: $0 [--dry-run|--self-test-failure]" >&2; exit 2; }
cd "$ROOT"
[[ "$(git branch --show-current)" == "$BRANCH" ]] || { echo "wrong branch" >&2; exit 1; }
head_sha="$(git rev-parse HEAD)"
[[ "$(git rev-parse "$head_sha^")" == "$PARENT" ]] || { echo "wrong Legacy parent" >&2; exit 1; }
[[ "$(git rev-list --count "$PARENT..$head_sha")" == 1 ]] || { echo "unexpected history" >&2; exit 1; }
git diff --quiet -- && git diff --cached --quiet -- || { echo "dirty tracked tree" >&2; exit 1; }
[[ "$(git rev-parse '@{upstream}')" == "$head_sha" ]] || { echo "upstream drift" >&2; exit 1; }
[[ "$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)" == "$head_sha" ]] || { echo "live remote drift" >&2; exit 1; }
command -v torchrun >/dev/null
command -v nvidia-smi >/dev/null
command -v flock >/dev/null
command -v python >/dev/null
state="/home/jupyter-iec2024iot04/.local/state/augseg-legacy-c321-rerun01-rtx5090/$head_sha"_seed2
mkdir -p "$state"
exec 9>"$state/runner.lock"
flock -n 9 || { echo "runner lock occupied" >&2; exit 1; }
run_method() {
  local method="$1"
  config="$BASE/$method/config.yaml"
  output="$ROOT/exp_boundary_mix_v2_v3_legacy_c321_rtx5090/$method"_"$SUFFIX"
  logdir="$ROOT/$BASE/$method/log"
  [[ ! -e "$state/$method.done" && ! -e "$state/$method.started" ]] || { echo "prior attempt state: $method" >&2; exit 1; }
  [[ ! -e "$output" && ! -e "$logdir" ]] || { echo "output or W&B log exists: $method" >&2; exit 1; }
  python tools/finalize_legacy_hf_upload.py --check-fresh "$config"
  avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  [[ "$avail_kb" =~ ^[0-9]+$ && "$avail_kb" -ge 104857600 ]] || { echo "less than 100 GiB free" >&2; exit 1; }
  gpu_jobs="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "$gpu_jobs" ]] || { echo "GPU 0 occupied; stopping" >&2; exit 1; }
  gpu_used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "$gpu_used" =~ ^[0-9]+$ && "$gpu_used" -le 512 ]] || { echo "GPU memory occupied; stopping" >&2; exit 1; }
  [[ "$(git rev-parse HEAD)" == "$head_sha" ]] || { echo "Git drift" >&2; exit 1; }
  [[ "$(git rev-parse '@{upstream}')" == "$head_sha" ]] || { echo "upstream drift" >&2; exit 1; }
  [[ "$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)" == "$head_sha" ]] || { echo "live remote drift" >&2; exit 1; }
  git diff --quiet -- && git diff --cached --quiet -- || { echo "dirty tracked tree" >&2; exit 1; }
  date -u +%FT%TZ > "$state/$method.started"
  mkdir -p "$output"
  python -c 'import secrets; print("legacy5090_" + secrets.token_hex(12))' > "$output/run_id.txt"
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_semi.py --config "$ROOT/$config" --seed 2 > "$state/$method.train.log" 2>&1
  python tools/finalize_legacy_hf_upload.py --check-local "$config"
  python tools/finalize_legacy_hf_upload.py "$config"
  python tools/finalize_legacy_hf_upload.py --check-wandb "$config"
  date -u +%FT%TZ > "$state/$method.done"
}
run_sequence
