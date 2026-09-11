#!/usr/bin/env bash

# Chạy duy nhất AugSeg baseline crop321 BS16 trên một GPU.
# train_semi.py tự quản lý checkpoint, auto-resume, W&B và Hugging Face.

set -uo pipefail
shopt -s nullglob

# ============================================================
# THAM SỐ CÓ THỂ CHỈNH
# ============================================================

ROOT="/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3"
ENV_FILE="/home/jupyter-iec2024iot04/.secrets/augseg_scheduler.env"

CONDA_EXE="/opt/tljh/user/bin/conda"
CONDA_ENV="augseg-bm"

GPU="${GPU:-0}"
SEED="${SEED:-2}"

# Chỉ chạy một training job tại một thời điểm.
MAX_JOBS="${MAX_JOBS:-1}"

# Chỉ khởi chạy job mới khi còn ít nhất lượng VRAM này.
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-16000}"

# Kiểm tra lại GPU và trạng thái job mỗi 120 giây.
POLL_SECONDS="${POLL_SECONDS:-120}"

# Sau khi mở một job, chờ VRAM ổn định rồi mới xét job tiếp theo.
LAUNCH_SETTLE_SECONDS="${LAUNCH_SETTLE_SECONDS:-120}"

# Nếu OOM, chờ 300 giây trước khi chạy lại.
OOM_RETRY_SECONDS="${OOM_RETRY_SECONDS:-300}"

# Mỗi phương pháp dùng một port riêng.
BASE_PORT="${BASE_PORT:-53947}"

CONFIGS=(
  "exps/boundary_mix_v2_v3/voc_semi662/baseline_augseg_fair80_c321_bs16x1_gbs16/config.yaml"
)

METHODS=(
  "baseline_augseg_fair80_r101_c321_bs16x1_gbs16"
)

# ============================================================
# HÀM CƠ BẢN
# ============================================================

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

# ============================================================
# KIỂM TRA ĐẦU VÀO
# ============================================================

[[ -d "$ROOT/.git" ]] ||
  die "Không tìm thấy repository: $ROOT"

[[ -x "$CONDA_EXE" ]] ||
  die "Không tìm thấy Conda: $CONDA_EXE"

command -v flock >/dev/null 2>&1 ||
  die "Không tìm thấy lệnh flock"

command -v nvidia-smi >/dev/null 2>&1 ||
  die "Không tìm thấy lệnh nvidia-smi"

[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] ||
  die "MAX_JOBS phải là số nguyên dương"

(( MAX_JOBS <= 2 )) ||
  die "MAX_JOBS không được lớn hơn 2"

cd "$ROOT" || die "Không thể truy cập repository"

for config in "${CONFIGS[@]}"; do
  [[ -f "$config" ]] ||
    die "Không tìm thấy config: $config"
done

# Không chạy training trên tracked code chưa commit.
git diff --quiet -- ||
  die "Tracked working tree đang có thay đổi chưa commit"

git diff --cached --quiet -- ||
  die "Staging area đang có thay đổi chưa commit"

RUN_HEAD="$(git rev-parse HEAD)" ||
  die "Không đọc được Git HEAD"

STATE_ROOT="$ROOT/.method_queue/u1_u4_simple"
STATE_DIR="$STATE_ROOT/${RUN_HEAD}_seed${SEED}"
LOG_DIR="$STATE_DIR/logs"

mkdir -p "$LOG_DIR"

# Không cho hai scheduler cùng quản lý queue.
exec 9>"$STATE_ROOT/scheduler.lock"
flock -n 9 ||
  die "Một phiên run_methods.sh khác đang chạy"

log "Repository HEAD: $RUN_HEAD"
log "GPU=$GPU"
log "Seed=$SEED"
log "Tối đa $MAX_JOBS job"
log "VRAM yêu cầu: ${MIN_FREE_VRAM_MIB} MiB"

# ============================================================
# QUẢN LÝ TRẠNG THÁI
# ============================================================

pid_file() {
  printf '%s/%s.pid' "$STATE_DIR" "$1"
}

done_file() {
  printf '%s/%s.done' "$STATE_DIR" "$1"
}

retry_file() {
  printf '%s/%s.retry_at' "$STATE_DIR" "$1"
}

solo_file() {
  printf '%s/%s.requires_solo' "$STATE_DIR" "$1"
}

fatal_file() {
  printf '%s/%s.fatal' "$STATE_DIR" "$1"
}

attempt_file() {
  printf '%s/%s.attempt' "$STATE_DIR" "$1"
}

method_is_done() {
  [[ -f "$(done_file "$1")" ]]
}

method_is_running() {
  local method="$1"
  local file
  local pid=""

  file="$(pid_file "$method")"

  [[ -f "$file" ]] || return 1

  read -r pid < "$file" || true

  if [[ "$pid" =~ ^[0-9]+$ ]] &&
     kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  # PID cũ không còn tồn tại.
  rm -f "$file"
  return 1
}

active_job_count() {
  local count=0
  local method

  for method in "${METHODS[@]}"; do
    if method_is_running "$method"; then
      count=$((count + 1))
    fi
  done

  printf '%s\n' "$count"
}

gpu_free_vram() {
  local value

  value="$(
    nvidia-smi \
      -i "$GPU" \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits \
      2>/dev/null |
      head -n 1 |
      tr -d '[:space:]'
  )"

  [[ "$value" =~ ^[0-9]+$ ]] || return 1

  printf '%s\n' "$value"
}

retry_is_ready() {
  local method="$1"
  local file
  local retry_at=0
  local now

  file="$(retry_file "$method")"

  [[ -f "$file" ]] || return 0

  read -r retry_at < "$file" || retry_at=0

  [[ "$retry_at" =~ ^[0-9]+$ ]] || retry_at=0

  now="$(date +%s)"

  (( now >= retry_at ))
}

next_attempt() {
  local method="$1"
  local file
  local attempt=0

  file="$(attempt_file "$method")"

  if [[ -f "$file" ]]; then
    read -r attempt < "$file" || attempt=0
  fi

  [[ "$attempt" =~ ^[0-9]+$ ]] || attempt=0

  attempt=$((attempt + 1))

  printf '%s\n' "$attempt" > "$file"
  printf '%s\n' "$attempt"
}

log_contains_oom() {
  local file="$1"

  grep -Eiq \
    'CUDA out of memory|CUDA error: out of memory|CUDA_ERROR_OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED|std::bad_alloc|out of memory' \
    "$file"
}

# ============================================================
# CHẠY MỘT PHƯƠNG PHÁP
# ============================================================

run_method() {
  local index="$1"
  local method="${METHODS[$index]}"
  local config="${CONFIGS[$index]}"
  local port=$((BASE_PORT + index))

  local attempt
  local output_log
  local exit_code
  local retry_at

  attempt="$(next_attempt "$method")"
  output_log="$LOG_DIR/${method}.attempt_${attempt}.log"

  # Xóa PID khi process kết thúc.
  trap 'rm -f "$(pid_file "$method")"' EXIT

  log "$method: bắt đầu attempt $attempt"
  log "$method: config=$config"
  log "$method: port=$port"
  log "$method: log=$output_log"

  # Giữ nguyên W&B/Hugging Face credentials và policy hiện tại.
  if [[ -f "$ENV_FILE" ]]; then
    set -a

    # shellcheck disable=SC1090
    source "$ENV_FILE"

    set +a
  fi

  # Đây chỉ là lệnh chạy train_semi.py.
  # Không có Python nhúng và không tạo config tạm.
  CUDA_VISIBLE_DEVICES="$GPU" \
    "$CONDA_EXE" run \
      --no-capture-output \
      -n "$CONDA_ENV" \
    python -m torch.distributed.run \
      --standalone \
      --nproc_per_node=1 \
      --master_port="$port" \
      train_semi.py \
      --config="$config" \
      --seed "$SEED" \
      --port "$port" \
      2>&1 | tee -a "$output_log"

  exit_code=${PIPESTATUS[0]}

  if (( exit_code == 0 )); then
    {
      printf 'STATUS=success\n'
      printf 'HEAD=%s\n' "$RUN_HEAD"
      printf 'CONFIG=%s\n' "$config"
      printf 'SEED=%s\n' "$SEED"
      printf 'ATTEMPT=%s\n' "$attempt"
      printf 'FINISHED_AT=%s\n' "$(timestamp)"
    } > "$(done_file "$method")"

    rm -f \
      "$(retry_file "$method")" \
      "$(solo_file "$method")" \
      "$(fatal_file "$method")"

    log "$method: hoàn thành thành công"
    return 0
  fi

  if log_contains_oom "$output_log"; then
    retry_at=$(($(date +%s) + OOM_RETRY_SECONDS))

    printf '%s\n' "$retry_at" > "$(retry_file "$method")"

    # Sau OOM, lần retry tiếp theo phải chạy một mình.
    : > "$(solo_file "$method")"

    log "$method: phát hiện OOM"
    log "$method: chờ ${OOM_RETRY_SECONDS}s rồi retry"
    log "$method: retry dùng lại đúng config gốc"
    log "$method: train_semi.py sẽ tự auto-resume"

    return 0
  fi

  {
    printf 'STATUS=fatal\n'
    printf 'HEAD=%s\n' "$RUN_HEAD"
    printf 'CONFIG=%s\n' "$config"
    printf 'SEED=%s\n' "$SEED"
    printf 'ATTEMPT=%s\n' "$attempt"
    printf 'EXIT_CODE=%s\n' "$exit_code"
    printf 'LOG=%s\n' "$output_log"
    printf 'FAILED_AT=%s\n' "$(timestamp)"
  } > "$(fatal_file "$method")"

  : > "$STATE_DIR/FATAL"

  log "$method: lỗi không phải OOM, exit code=$exit_code"
  log "$method: scheduler sẽ không mở thêm job"

  return 0
}

launch_method() {
  local index="$1"
  local method="${METHODS[$index]}"
  local worker_pid

  run_method "$index" &

  worker_pid=$!

  printf '%s\n' "$worker_pid" > "$(pid_file "$method")"

  log "$method: đã khởi chạy PID $worker_pid"
}

# ============================================================
# SCHEDULER
# ============================================================

while true; do
  # Không mở job mới nếu repository bị thay đổi trong lúc scheduler chạy.
  if [[ "$(git rev-parse HEAD 2>/dev/null)" != "$RUN_HEAD" ]] ||
     ! git diff --quiet -- ||
     ! git diff --cached --quiet --; then
    : > "$STATE_DIR/FATAL"
    log "Repository đã thay đổi; dừng mở job mới"
  fi

  active="$(active_job_count)"

  # Lỗi không phải OOM: không mở job mới.
  if [[ -f "$STATE_DIR/FATAL" ]]; then
    if (( active == 0 )); then
      die "Scheduler dừng. Kiểm tra $STATE_DIR/*.fatal"
    fi

    log "Có lỗi fatal; chờ $active job đang chạy kết thúc"
    sleep "$POLL_SECONDS"
    continue
  fi

  # Kiểm tra tất cả phương pháp đã hoàn tất chưa.
  all_done=1

  for method in "${METHODS[@]}"; do
    if ! method_is_done "$method"; then
      all_done=0
      break
    fi
  done

  if (( all_done == 1 )); then
    if (( active == 0 )); then
      log "AugSeg baseline crop321 BS16 đã hoàn thành"
      exit 0
    fi

    sleep "$POLL_SECONDS"
    continue
  fi

  # Đã đạt giới hạn job thì chỉ chờ.
  if (( active >= MAX_JOBS )); then
    log "Đang chạy $active/$MAX_JOBS job"
    sleep "$POLL_SECONDS"
    continue
  fi

  # ==========================================================
  # ƯU TIÊN METHOD ĐÃ TỪNG OOM
  # ==========================================================

  solo_index=-1

  for index in "${!METHODS[@]}"; do
    method="${METHODS[$index]}"

    if [[ -f "$(solo_file "$method")" ]] &&
       ! method_is_done "$method" &&
       ! method_is_running "$method"; then
      solo_index="$index"
      break
    fi
  done

  if (( solo_index >= 0 )); then
    method="${METHODS[$solo_index]}"

    # Method từng OOM chỉ retry khi không còn job nào khác.
    if (( active > 0 )); then
      log "$method cần chạy solo; chờ $active job còn lại"
      sleep "$POLL_SECONDS"
      continue
    fi

    if ! retry_is_ready "$method"; then
      log "$method chưa tới thời điểm retry"
      sleep "$POLL_SECONDS"
      continue
    fi
  fi

  # ==========================================================
  # KIỂM TRA VRAM
  # ==========================================================

  free_vram="$(gpu_free_vram)" || {
    log "Không đọc được VRAM GPU $GPU"
    sleep "$POLL_SECONDS"
    continue
  }

  if (( free_vram < MIN_FREE_VRAM_MIB )); then
    log "GPU còn ${free_vram} MiB; cần ${MIN_FREE_VRAM_MIB} MiB"
    sleep "$POLL_SECONDS"
    continue
  fi

  # Nếu có method cần retry solo, chạy method đó trước.
  if (( solo_index >= 0 )); then
    launch_method "$solo_index"
    sleep "$LAUNCH_SETTLE_SECONDS"
    continue
  fi

  # ==========================================================
  # TÌM METHOD TIẾP THEO CHƯA CHẠY
  # ==========================================================

  launch_index=-1

  for index in "${!METHODS[@]}"; do
    method="${METHODS[$index]}"

    if method_is_done "$method"; then
      continue
    fi

    if method_is_running "$method"; then
      continue
    fi

    if [[ -f "$(fatal_file "$method")" ]]; then
      continue
    fi

    if ! retry_is_ready "$method"; then
      continue
    fi

    launch_index="$index"
    break
  done

  if (( launch_index >= 0 )); then
    launch_method "$launch_index"

    # Đợi job vừa mở cấp phát VRAM ổn định.
    sleep "$LAUNCH_SETTLE_SECONDS"
  else
    log "Chưa có method nào sẵn sàng để mở"
    sleep "$POLL_SECONDS"
  fi
done