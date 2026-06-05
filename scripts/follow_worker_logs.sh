#!/usr/bin/env bash
set -euo pipefail

journalctl --user -u augseg-worker-5090.service -f
