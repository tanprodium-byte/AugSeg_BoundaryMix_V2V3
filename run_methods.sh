#!/usr/bin/env bash
set -Eeuo pipefail

# Chỉ user hiện tại mới đọc được các file mới tạo.
umask 077

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================

ROOT="/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3"
ENV_FILE="/home/jupyter-iec2024iot04/.secrets/augseg_scheduler.env"

CONDA_EXE="/opt/tljh/user/bin/conda"
CONDA_ENV="augseg-bm"

# ============================================================
# CHỈ CẦN THÊM, XÓA HOẶC ĐỔI THỨ TỰ CONFIG Ở ĐÂY
# ============================================================

CONFIGS=(
  "exps/boundary_mix_v2_v3/voc_semi662/baseline_augseg_fair80_c513_bs8/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/c4_csl_official_direct_labeled_adaptive_gate_guided_cutmix_plus_ce_weight_v1_c513_bs8/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/s1_saliency_box_adaptive_relocated_cutmix_c513_bs8/config.yaml"
  "exps/boundary_mix_v2_v3/voc_semi662/s2_saliency_component_mask_relocated_cutmix_c513_bs8/config.yaml"
)

# ============================================================
# KIỂM TRA CÁC ĐƯỜNG DẪN CƠ BẢN
# ============================================================

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] Repository không tồn tại:" >&2
  echo "$ROOT" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] Thiếu environment file:" >&2
  echo "$ENV_FILE" >&2
  exit 1
fi

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "[ERROR] Conda executable không tồn tại hoặc không có quyền chạy:" >&2
  echo "$CONDA_EXE" >&2
  exit 1
fi

# ============================================================
# NẠP BIẾN MÔI TRƯỜNG
# ============================================================

set -a
source "$ENV_FILE"
set +a

# Có thể ghi đè khi gọi script từ bên ngoài.
#
# Chạy bình thường:
#   ./run_methods.sh
#
# Chạy GPU 1, seed 3:
#   CUDA_VISIBLE_DEVICES=1 SEED=3 ./run_methods.sh

GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-2}"
BASE_PORT="${PORT:-53947}"

# ============================================================
# THƯ MỤC TRẠNG THÁI VÀ LOG
# ============================================================

STATE_DIR="$ROOT/.method_queue"
DONE_DIR="$STATE_DIR/done"
LOG_DIR="$STATE_DIR/logs"
LOCK_FILE="$STATE_DIR/queue.lock"

mkdir -p "$DONE_DIR" "$LOG_DIR"

cd "$ROOT"

# ============================================================
# KHÔNG CHO HAI QUEUE CHẠY ĐỒNG THỜI
# ============================================================

exec 9>"$LOCK_FILE"

if ! flock -n 9; then
  echo "[ERROR] Một phiên run_methods.sh khác đang chạy." >&2
  exit 1
fi

# ============================================================
# THÔNG TIN ĐẦU PHIÊN
# ============================================================

echo "============================================================"
echo "AugSeg sequential runner"
echo "Repository: $ROOT"
echo "Conda:     $CONDA_EXE"
echo "Environment: $CONDA_ENV"
echo "GPU:       $GPU"
echo "Seed:      $SEED"
echo "Jobs:      ${#CONFIGS[@]}"
echo "Started:   $(date --iso-8601=seconds)"
echo "============================================================"

# ============================================================
# CHẠY LẦN LƯỢT TỪNG CONFIG
# ============================================================

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"

  # ----------------------------------------------------------
  # Kiểm tra config
  # ----------------------------------------------------------

  if [[ ! -f "$config" ]]; then
    echo >&2
    echo "[ERROR] Config không tồn tại:" >&2
    echo "$ROOT/$config" >&2
    exit 1
  fi

  # Lấy tên thư mục chứa config làm tên phương pháp.
  method_name="$(basename "$(dirname "$config")")"

  done_file="$DONE_DIR/${method_name}_seed${SEED}.done"
  log_file="$LOG_DIR/${method_name}_seed${SEED}.log"

  # ----------------------------------------------------------
  # Bỏ qua phương pháp đã hoàn thành
  # ----------------------------------------------------------

  if [[ -f "$done_file" ]]; then
    echo "[SKIP] $method_name đã hoàn thành với seed $SEED"
    continue
  fi

  # Mỗi phương pháp dùng một port riêng theo vị trí trong danh sách.
  port=$((BASE_PORT + index))

  echo
  echo "============================================================"
  echo "[RUN]      $method_name"
  echo "Config:    $config"
  echo "Seed:      $SEED"
  echo "GPU:       $GPU"
  echo "Port:      $port"
  echo "Log:       $log_file"
  echo "Started:   $(date --iso-8601=seconds)"
  echo "============================================================"

  # ----------------------------------------------------------
  # Khởi tạo log
  #
  # Dấu > ghi đè log cũ của method chưa hoàn thành.
  # ----------------------------------------------------------

  {
    echo "METHOD=$method_name"
    echo "CONFIG=$config"
    echo "SEED=$SEED"
    echo "GPU=$GPU"
    echo "PORT=$port"
    echo "CONDA_EXE=$CONDA_EXE"
    echo "CONDA_ENV=$CONDA_ENV"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
    echo "============================================================"
  } > "$log_file"

  # ----------------------------------------------------------
  # Chạy training
  #
  # Tạm tắt set -e để lấy exit code và tự xử lý lỗi.
  # ----------------------------------------------------------

  set +e

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
    2>&1 | tee -a "$log_file"

  # Lưu ngay trạng thái của từng lệnh trong pipeline:
  # [0] = conda/Python training
  # [1] = tee ghi log
  pipeline_status=("${PIPESTATUS[@]}")

  train_exit_code="${pipeline_status[0]}"
  tee_exit_code="${pipeline_status[1]}"

  set -e

  # ----------------------------------------------------------
  # Xử lý lỗi training hoặc lỗi ghi log
  # ----------------------------------------------------------

  if [[ "$train_exit_code" -ne 0 || "$tee_exit_code" -ne 0 ]]; then
    echo
    echo "============================================================"
    echo "[FAILED] $method_name"
    echo "Training exit code: $train_exit_code"
    echo "Log writer exit code: $tee_exit_code"
    echo "Log: $log_file"
    echo
    echo "Queue dừng tại phương pháp này."
    echo "Không tạo marker .done."
    echo
    echo "Sau khi xử lý lỗi, chạy lại run_methods.sh:"
    echo "- phương pháp đã hoàn thành sẽ được bỏ qua;"
    echo "- phương pháp này sẽ được gọi lại."
    echo "============================================================"

    if [[ "$train_exit_code" -ne 0 ]]; then
      exit "$train_exit_code"
    fi

    exit "$tee_exit_code"
  fi

  # ----------------------------------------------------------
  # Chỉ tạo marker khi training và ghi log đều thành công
  # ----------------------------------------------------------

  {
    echo "METHOD=$method_name"
    echo "CONFIG=$config"
    echo "SEED=$SEED"
    echo "GPU=$GPU"
    echo "PORT=$port"
    echo "FINISHED_AT=$(date --iso-8601=seconds)"
    echo "STATUS=success"
  } > "$done_file"

  echo
  echo "[DONE] $method_name"
done

# ============================================================
# KẾT THÚC
# ============================================================

echo
echo "============================================================"
echo "TẤT CẢ METHOD ĐÃ HOÀN THÀNH"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
