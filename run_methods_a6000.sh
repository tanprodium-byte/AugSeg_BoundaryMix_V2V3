#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly EXPECTED_ROOT="/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3"
readonly PYTHON="/home/islabworker3/tantv/envs/augseg-bm/bin/python"
readonly ACCEPTED_PARENT="ecced7957895108ff8bae6114fcf9c1da0a3f954"
readonly BRANCH="feature/auto-run-12-methods-gpu-suite"
readonly STATE_ROOT="/home/islabworker3/.local/state/augseg-a6000/a6000_rerun01"
readonly HF_REPO="tanprodium/augseg-checkpoints-a6000-rerun01"
readonly GPU=0
readonly SEED=2
readonly MAX_CONCURRENT=1
readonly MIN_FREE_VRAM_MIB=45000
readonly MIN_FREE_DISK_BYTES=16106127360
readonly MIN_FREE_INODES=100000
readonly POLL_SECONDS=120
readonly BASE_PORT=54947

readonly -a CONFIGS=(
  "exps/boundary_mix_v2_v3/voc_semi662/u1_self_pseudo_saliency_u2u_cutmix_a6000_rerun01_c321_bs16x1_gbs16/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/u2_cross_view_saliency_u2u_cutmix_a6000_rerun01_c321_bs16x1_gbs16/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/u3_confidence_filtered_cross_view_saliency_u2u_cutmix_a6000_rerun01_c321_bs16x1_gbs16/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/u4_confidence_filtered_self_pseudo_saliency_u2u_cutmix_a6000_rerun01_c321_bs16x1_gbs16/config.yaml"
)
readonly -a METHODS=("u1" "u2" "u3" "u4")
readonly -a METHOD_NAMES=(
  "u1_self_pseudo_saliency_u2u_cutmix_a6000_rerun01_r101_c321_bs16x1_gbs16"
  "u2_cross_view_saliency_u2u_cutmix_a6000_rerun01_r101_c321_bs16x1_gbs16"
  "u3_confidence_filtered_cross_view_saliency_u2u_cutmix_a6000_rerun01_r101_c321_bs16x1_gbs16"
  "u4_confidence_filtered_self_pseudo_saliency_u2u_cutmix_a6000_rerun01_r101_c321_bs16x1_gbs16"
)
readonly -a METHOD_BLOCKS=(
  "u1_saliency_u2u"
  "u2_cross_view_saliency_u2u"
  "u3_confidence_filtered_cross_view_saliency_u2u"
  "u4_confidence_filtered_self_pseudo_saliency_u2u"
)
readonly -a HF_METHODS=(
  "u1_self_pseudo_saliency_u2u_cutmix"
  "u2_cross_view_saliency_u2u_cutmix"
  "u3_confidence_filtered_cross_view_saliency_u2u_cutmix"
  "u4_confidence_filtered_self_pseudo_saliency_u2u_cutmix"
)

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

resolve_repository_root() {
  [[ ! -L "${BASH_SOURCE[0]}" ]] || die "runner script must not be a symlink"
  local root
  root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  [[ "$root" == "$EXPECTED_ROOT" ]] || die "repository root mismatch: $root"
  printf '%s\n' "$root"
}

assert_no_symlink_components() {
  local path="$1" current="/" part
  [[ "$path" == /* && "$path" != "/" ]] || return 1
  IFS='/' read -r -a parts <<< "${path#/}"
  for part in "${parts[@]}"; do
    [[ -n "$part" && "$part" != "." && "$part" != ".." ]] || return 1
    current="${current%/}/$part"
    [[ ! -L "$current" ]] || return 1
  done
}

validate_state_root() {
  [[ "$STATE_ROOT" == "/home/islabworker3/.local/state/augseg-a6000/a6000_rerun01" ]] || die "state-root constant changed"
  [[ "$(realpath -m -- "$STATE_ROOT")" == "$STATE_ROOT" ]] || die "state root is unresolved"
  assert_no_symlink_components "$STATE_ROOT" || die "state root contains an unsafe component"
  mkdir -p -- "$STATE_ROOT"
  [[ -d "$STATE_ROOT" && ! -L "$STATE_ROOT" ]] || die "state root is unsafe"
}

validate_state_inventory() {
  local entry name allowed method
  shopt -s nullglob dotglob
  for entry in "$STATE_ROOT"/*; do
    name="${entry##*/}"
    allowed=0
    [[ "$name" == "runner.lock" ]] && allowed=1
    for method in "${METHODS[@]}"; do [[ "$name" == "$method" ]] && allowed=1; done
    (( allowed == 1 )) || die "unexpected cohort state entry: $entry"
    if [[ "$name" != "runner.lock" ]]; then
      [[ -d "$entry" && ! -L "$entry" && "$(realpath -m -- "$entry")" == "$STATE_ROOT/$name" ]] ||
        die "unsafe method state entry: $entry"
    fi
  done
  shopt -u dotglob
}

validate_repository_state() {
  local root="$1" head branch upstream counts
  [[ "$MAX_CONCURRENT" == 1 ]] || die "concurrency policy changed"
  branch="$(git -C "$root" branch --show-current)"
  [[ "$branch" == "$BRANCH" ]] || die "branch mismatch: $branch"
  head="$(git -C "$root" rev-parse HEAD)"
  [[ -n "${RUN_HEAD:-}" ]] || RUN_HEAD="$head"
  [[ "$head" == "$RUN_HEAD" ]] || die "repository HEAD changed during cohort execution"
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B - "$root" <<'PY'
import importlib.util
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = root / "tools/smoke_u1_u4_full_bs16.py"
spec = importlib.util.spec_from_file_location("a6000_smoke_guard", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.validate_repository("u1")
PY
  upstream="$(git -C "$root" rev-parse '@{upstream}')"
  [[ "$upstream" == "$head" ]] || die "upstream mismatch"
  counts="$(git -C "$root" rev-list --left-right --count HEAD...'@{upstream}')"
  [[ "$counts" == $'0\t0' ]] || die "ahead/behind is not 0/0"
}

validate_config() {
  local root="$1" index="$2"
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B - \
    "$root/${CONFIGS[$index]}" "${METHOD_NAMES[$index]}" "${METHOD_BLOCKS[$index]}" \
    "$HF_REPO" "${HF_METHODS[$index]}" <<'PY'
import copy
import os
from pathlib import Path
import sys
import yaml
config_path = Path(sys.argv[1]).resolve()
expected_name, expected_block, expected_repo, hf_method = sys.argv[2:]
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
blocks = (
    "u1_saliency_u2u", "u2_cross_view_saliency_u2u",
    "u3_confidence_filtered_cross_view_saliency_u2u",
    "u4_confidence_filtered_self_pseudo_saliency_u2u",
)
assert cfg["name"] == cfg["run"]["name"] == cfg["wandb"]["name"] == expected_name
assert [name for name in blocks if cfg.get(name, {}).get("enabled")] == [expected_block]
assert cfg["dataset"]["train"]["crop"]["size"] == [321, 321]
assert cfg["dataset"]["train"]["batch_size"] == 16 and cfg["dataset"]["workers"] == 4
assert cfg["trainer"]["epochs"] == 80
assert cfg["trainer"]["optimizer"]["kwargs"]["lr"] == 0.000125
assert cfg["trainer"]["unsupervised"]["threshold"] == 0.95
assert cfg["hf"]["repo_id"] == expected_repo
assert cfg["hf"]["enabled"] and cfg["hf"]["auto_download"] and cfg["hf"]["auto_upload"]
profile = "r101_c321_bs16x1_gbs16"
raw = cfg["hf"]["path_in_repo"]
expected_raw = f"boundary_mix_v2_v3/voc_semi662/a6000_rerun01/{hf_method}/latest.tar.gz"
assert raw == expected_raw and profile not in raw.split("/")
effective = str(Path(raw).parent / profile / Path(raw).name)
snapshot = os.path.normpath(os.path.join(str(config_path.parent), cfg["saver"]["snapshot_dir"]))
expected_output = str(config_path.parents[4] / "exp_boundary_mix_v2_v3" / expected_name)
assert snapshot == expected_output
print(snapshot)
print(str(config_path.parent / "log"))
print(effective)
PY
}

ensure_fresh_local_state() {
  local output="$1" log_dir="$2" state_dir="$3"
  [[ ! -e "$output" && ! -L "$output" ]] || die "official output already exists: $output"
  [[ ! -e "$log_dir" && ! -L "$log_dir" ]] || die "official config log already exists: $log_dir"
  [[ ! -e "$state_dir" && ! -L "$state_dir" ]] || die "unexpected method state already exists: $state_dir"
}

require_hf_object_absent() {
  local effective_path="$1"
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B - "$HF_REPO" "$effective_path" <<'PY'
from huggingface_hub import HfApi
import sys
repo, path = sys.argv[1:]
api = HfApi()
api.repo_info(repo_id=repo, repo_type="model")
if api.file_exists(repo_id=repo, filename=path, repo_type="model"):
    raise SystemExit("target HF object already exists")
PY
}

production_resource_snapshot() {
  local free_vram gpu_name gpu_pids free_bytes free_inodes
  gpu_name="$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')" || return 1
  [[ "$gpu_name" == "NVIDIA RTX A6000" ]] || { printf 'WAIT gpu_identity=%s\n' "$gpu_name"; return 0; }
  free_vram="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')" || return 1
  gpu_pids="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
  free_bytes="$(df -PB1 "$EXPECTED_ROOT" | awk 'NR==2 {print $4}')"
  free_inodes="$(df -Pi "$EXPECTED_ROOT" | awk 'NR==2 {print $4}')"
  if (( gpu_pids > 0 )); then printf 'WAIT gpu_processes=%s\n' "$gpu_pids"
  elif (( free_vram < MIN_FREE_VRAM_MIB )); then printf 'WAIT free_vram_mib=%s required=%s\n' "$free_vram" "$MIN_FREE_VRAM_MIB"
  elif (( free_bytes < MIN_FREE_DISK_BYTES )); then printf 'WAIT free_disk_bytes=%s required=%s\n' "$free_bytes" "$MIN_FREE_DISK_BYTES"
  elif (( free_inodes < MIN_FREE_INODES )); then printf 'WAIT free_inodes=%s required=%s\n' "$free_inodes" "$MIN_FREE_INODES"
  else printf 'READY free_vram_mib=%s free_disk_bytes=%s free_inodes=%s\n' "$free_vram" "$free_bytes" "$free_inodes"
  fi
}

production_wait() { sleep "$POLL_SECONDS"; }

wait_for_resources() {
  local probe="$1" sleeper="$2" status
  while true; do
    status="$($probe)" || die "resource probe failed"
    if [[ "$status" == READY\ * ]]; then log "$status"; return 0; fi
    [[ "$status" == WAIT\ * ]] || die "invalid resource status: $status"
    log "$status; polling again in ${POLL_SECONDS}s"
    "$sleeper"
  done
}

write_failure_marker() {
  local state_dir="$1" reason="$2" exit_code="$3"
  mkdir -p -- "$state_dir"
  printf 'status=failure\nreason=%s\nexit_code=%s\nhead=%s\nseed=%s\ntime=%s\n' \
    "$reason" "$exit_code" "$RUN_HEAD" "$SEED" "$(timestamp)" > "$state_dir/failure.marker"
}

validate_completion() {
  local output="$1" state_dir="$2" expected_name="$3" method="$4" effective_path="$5" run_log="$6"
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B - \
    "$output" "$state_dir" "$expected_name" "$method" "$effective_path" "$run_log" "$RUN_HEAD" "$HF_REPO" <<'PY'
import csv, json, math, re
from pathlib import Path
import sys
output, state, expected_name, method, hf_path, run_log, head, hf_repo = sys.argv[1:]
output, state, run_log = Path(output), Path(state), Path(run_log)
required = ["ckpt.pth", "ckpt_best.pth", "manifest.json", "epoch_metrics.csv", "iter_metrics.csv", "run_id.txt"]
assert all((output / name).is_file() for name in required), "missing completion artifact"
with (output / "epoch_metrics.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
epochs = [int(row["epoch"]) for row in rows]
assert len(rows) == 80 and sorted(epochs) == list(range(80)) and len(set(epochs)) == 80
manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
assert manifest["epoch"] == 79 and manifest["run_name"] == expected_name and manifest["world_size"] == 1
assert str(manifest["git_commit"]) == head[:7]
run_id = (output / "run_id.txt").read_text(encoding="utf-8").strip()
assert run_id and "Resumed checkpoint" not in run_log.read_text(encoding="utf-8", errors="replace")
marker = {
 "u1":"u1_saliency_u2u", "u2":"u2_cross_view_saliency_u2u",
 "u3":"u3_confidence_filtered_cross_view_saliency_u2u",
 "u4":"u4_confidence_filtered_self_pseudo_saliency_u2u",
}[method]
log_text = run_log.read_text(encoding="utf-8", errors="replace")
pattern = re.compile(rf"\[{re.escape(marker)}\] epoch=(\d+) step=(\d+) global_iter=(\d+) (.*)")
diagnostics = pattern.findall(log_text)
assert diagnostics and any(int(epoch) == 79 for epoch, _, _, _ in diagnostics)
_, _, _, diagnostic_text = diagnostics[-1]
fields = dict(item.split("=", 1) for item in diagnostic_text.split() if "=" in item)
assert fields.get("u1/rng_policy") == "u1_rng_policy_v1"
for key in ("u1/total_receivers", "u1/total_candidate_draws", "u1/paste_attempts", "u1/nonempty_paste_attempts", "u1/paste_attempt_rate", "u1/nonempty_mixed_target_rate"):
    assert key in fields and math.isfinite(float(fields[key]))
receivers = int(float(fields["u1/total_receivers"]))
candidates = int(float(fields["u1/total_candidate_draws"]))
pastes = int(float(fields["u1/paste_attempts"]))
nonempty = int(float(fields["u1/nonempty_paste_attempts"]))
assert receivers == 16 and candidates == 128 and pastes == receivers and 0 <= nonempty <= pastes
assert math.isclose(float(fields["u1/paste_attempt_rate"]), pastes / receivers)
assert math.isclose(float(fields["u1/nonempty_mixed_target_rate"]), nonempty / receivers)
if method in ("u3", "u4"):
    for suffix in ("total_probe_pixels", "confidence_valid_probe_pixels", "ignored_label_pixels", "zero_valid_donors", "zero_valid_donor_rate", "all_valid_donors", "all_valid_donor_rate", "finite_probe_events", "near_flat_donor_rate"):
        key = f"{method}/{suffix}"
        assert key in fields and math.isfinite(float(fields[key]))
with (output / "iter_metrics.csv").open(newline="", encoding="utf-8") as stream:
    iter_rows = [row for row in csv.DictReader(stream) if row.get("meta/log_type") == "train_iter"]
assert iter_rows
for row in iter_rows:
    for key in ("iter/sup_loss", "iter/uns_loss"):
        assert math.isfinite(float(row[key]))

from huggingface_hub import HfApi
api = HfApi()
info = api.get_paths_info(repo_id=hf_repo, paths=[hf_path], repo_type="model", expand=True)
assert len(info) == 1 and info[0].path == hf_path
lfs_value = getattr(info[0], "lfs", None)
if isinstance(lfs_value, dict):
    lfs = lfs_value
elif lfs_value is None:
    lfs = {}
else:
    lfs = {
        key: getattr(lfs_value, key)
        for key in ("sha256", "size", "pointer_size")
        if hasattr(lfs_value, key)
    }
hf_record = {"repo_id": hf_repo, "path": hf_path, "size": int(info[0].size), "lfs": lfs}
(state / "hf_metadata.json").write_text(json.dumps(hf_record, sort_keys=True) + "\n", encoding="utf-8")

import wandb
run = wandb.Api().run(f"tanprodium-uit/AugsegResearch/{run_id}")
assert run.name == expected_name and run.state == "finished"
(state / "wandb_metadata.json").write_text(json.dumps({"run_id": run_id, "name": run.name, "state": run.state}, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_method() {
  local root="$1" index="$2" method config expected_name state_dir metadata output log_dir effective_path port run_log rc
  method="${METHODS[$index]}"; config="${CONFIGS[$index]}"; expected_name="${METHOD_NAMES[$index]}"
  state_dir="$STATE_ROOT/$method"
  validate_repository_state "$root"
  mapfile -t metadata < <(validate_config "$root" "$index")
  [[ "${#metadata[@]}" == 3 ]] || die "invalid config metadata for $method"
  output="${metadata[0]}"; log_dir="${metadata[1]}"; effective_path="${metadata[2]}"
  if [[ -e "$state_dir" || -L "$state_dir" ]]; then
    [[ -d "$state_dir" && ! -L "$state_dir" ]] || die "unsafe existing method state: $state_dir"
    [[ ! -e "$state_dir/failure.marker" ]] || die "$method has preserved failure evidence; no retry is automatic"
    if [[ -f "$state_dir/done.marker" && -f "$state_dir/run.log" ]]; then
      validate_completion "$output" "$state_dir" "$expected_name" "$method" "$effective_path" "$state_dir/run.log" ||
        die "$method runner-owned completion evidence is invalid"
      log "$method runner-owned completion evidence revalidated; skipping completed method"
      return 0
    fi
    die "$method has unexpected partial state; no automatic resume or retry"
  fi
  ensure_fresh_local_state "$output" "$log_dir" "$state_dir"
  require_hf_object_absent "$effective_path"
  wait_for_resources production_resource_snapshot production_wait
  validate_repository_state "$root"
  ensure_fresh_local_state "$output" "$log_dir" "$state_dir"
  require_hf_object_absent "$effective_path"
  mkdir -p -- "$state_dir"
  run_log="$state_dir/run.log"; port=$((BASE_PORT + index))
  printf 'head=%s\nmethod=%s\nconfig=%s\nseed=%s\nworld_size=1\nbatch_size=16\nstarted=%s\n' \
    "$RUN_HEAD" "$method" "$config" "$SEED" "$(timestamp)" > "$state_dir/launch.marker"
  log "launching $method with world_size=1 seed=$SEED"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node=1 --master_port="$port" \
    train_semi.py --config="$config" --seed "$SEED" --port "$port" 2>&1 | tee "$run_log"
  rc=${PIPESTATUS[0]}
  set -e
  if (( rc != 0 )); then write_failure_marker "$state_dir" "training-process-nonzero" "$rc"; die "$method failed with exit $rc"; fi
  if ! validate_completion "$output" "$state_dir" "$expected_name" "$method" "$effective_path" "$run_log"; then
    write_failure_marker "$state_dir" "completion-validation-failed" 0
    die "$method completion evidence failed validation"
  fi
  printf 'status=complete\nhead=%s\nmethod=%s\nseed=%s\nfinished=%s\n' \
    "$RUN_HEAD" "$method" "$SEED" "$(timestamp)" > "$state_dir/done.marker"
  log "$method completion and external identities verified"
}

self_test() {
  local test_root test_device test_inode lock_file counter calls rc
  test_root="$(mktemp -d /tmp/augseg-a6000-runner-test.XXXXXX)"
  test_device="$(stat -c %d -- "$test_root")"
  test_inode="$(stat -c %i -- "$test_root")"
  cleanup_test() {
    [[ -n "${test_root:-}" && "$test_root" == /tmp/augseg-a6000-runner-test.* && "$test_root" != "/tmp" ]] || return 1
    [[ "$(realpath -m -- "$test_root")" == "$test_root" && "$(dirname -- "$test_root")" == "/tmp" ]] || return 1
    [[ -d "$test_root" && ! -L "$test_root" ]] || return 1
    [[ "$(stat -c %d -- "$test_root")" == "$test_device" && "$(stat -c %i -- "$test_root")" == "$test_inode" ]] || return 1
    rm -rf -- "$test_root"
  }
  trap cleanup_test EXIT
  [[ "$MAX_CONCURRENT" == 1 && "$SEED" == 2 && "$PYTHON" == "/home/islabworker3/tantv/envs/augseg-bm/bin/python" ]]
  [[ "${METHODS[*]}" == "u1 u2 u3 u4" && "${#CONFIGS[@]}" == 4 ]]
  [[ "$STATE_ROOT" == "/home/islabworker3/.local/state/augseg-a6000/a6000_rerun01" ]]
  mkdir "$test_root/state" "$test_root/outside"
  assert_no_symlink_components "$test_root/state"
  ln -s "$test_root/outside" "$test_root/state/escape"
  ! assert_no_symlink_components "$test_root/state/escape/child"
  lock_file="$test_root/runner.lock"; exec 8>"$lock_file"; flock -n 8
  ! flock -n "$lock_file" -c true
  counter="$test_root/probe.count"; printf '0\n' > "$counter"
  fake_probe() { local n; n="$(<"$counter")"; n=$((n+1)); printf '%s\n' "$n" > "$counter"; if ((n<3)); then echo 'WAIT simulated'; else echo 'READY simulated'; fi; }
  fake_sleep() { :; }
  wait_for_resources fake_probe fake_sleep
  [[ "$(<"$counter")" == 3 ]]
  calls="$test_root/calls"; : > "$calls"
  simulate_sequence() { local fail="$1" m; for m in "${METHODS[@]}"; do printf '%s\n' "$m" >> "$calls"; [[ "$m" != "$fail" ]] || return 9; done; }
  simulate_sequence none; [[ "$(tr '\n' ' ' < "$calls")" == "u1 u2 u3 u4 " ]]
  : > "$calls"; set +e; simulate_sequence u2; rc=$?; set -e
  [[ "$rc" == 9 && "$(tr '\n' ' ' < "$calls")" == "u1 u2 " ]]
  [[ ! -e "$test_root/production-training" && ! -e "$test_root/wandb" && ! -e "$test_root/hf" ]]
  echo "PASS runner_self_test order seed environment root concurrency flock symlink wait success failure-stop isolation"
}

if (($#)); then
  [[ "$#" == 1 && "$1" == "--self-test" ]] || die "runner accepts no production arguments"
  self_test
  exit 0
fi

ROOT="$(resolve_repository_root)"
[[ -x "$PYTHON" ]] || die "target Python is unavailable"
command -v flock >/dev/null || die "flock is unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
validate_state_root
exec 9>"$STATE_ROOT/runner.lock"
flock -n 9 || die "another A6000 cohort runner holds the single-instance lock"
validate_state_inventory
RUN_HEAD="$(git -C "$ROOT" rev-parse HEAD)"
validate_repository_state "$ROOT"
for index in "${!METHODS[@]}"; do run_method "$ROOT" "$index"; done
log "A6000 U1-U4 cohort completed with all gates satisfied"
