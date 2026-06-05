#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_DIR="${HOME}/.config/systemd/user"
SERVICE="augseg-worker-5090-postgres.service"

mkdir -p "${USER_DIR}"
cp "${ROOT}/scripts/systemd/${SERVICE}" "${USER_DIR}/"

systemctl --user daemon-reload

cat <<'EOF'
Installed Postgres worker user service.

Before starting, create ~/.secrets/augseg_scheduler.env with:
  export AUGSEG_SCHEDULER_DB_URL='...'

Enable/start manually:
  systemctl --user enable augseg-worker-5090-postgres.service
  systemctl --user start augseg-worker-5090-postgres.service
  systemctl --user status augseg-worker-5090-postgres.service

Or install and start in one step:
  bash scripts/install_postgres_worker_systemd_user.sh --enable-now
EOF

if [[ "${1:-}" == "--enable-now" ]]; then
  systemctl --user enable "${SERVICE}"
  systemctl --user start "${SERVICE}"
fi
