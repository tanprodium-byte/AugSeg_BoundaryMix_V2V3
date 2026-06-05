#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

systemctl --user status augseg-coordinator.service --no-pager
systemctl --user status augseg-worker-5090.service --no-pager
curl http://127.0.0.1:8787/health
echo
python scheduler/status.py
