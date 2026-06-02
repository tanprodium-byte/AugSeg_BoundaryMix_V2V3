#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_DIR="${HOME}/.config/systemd/user"

mkdir -p "${USER_DIR}"
cp "${ROOT}/scripts/systemd/augseg-coordinator.service" "${USER_DIR}/"
cp "${ROOT}/scripts/systemd/augseg-worker-5090.service" "${USER_DIR}/"

systemctl --user daemon-reload

cat <<'EOF'
Installed user service files.

Enable/start manually:
  systemctl --user enable augseg-coordinator.service
  systemctl --user enable augseg-worker-5090.service
  systemctl --user start augseg-coordinator.service
  systemctl --user start augseg-worker-5090.service
  systemctl --user status augseg-coordinator.service
  systemctl --user status augseg-worker-5090.service

Or install and start in one step:
  bash scripts/install_systemd_user_services.sh --enable-now
EOF

if [[ "${1:-}" == "--enable-now" ]]; then
  systemctl --user enable augseg-coordinator.service
  systemctl --user enable augseg-worker-5090.service
  systemctl --user start augseg-coordinator.service
  systemctl --user start augseg-worker-5090.service
fi
