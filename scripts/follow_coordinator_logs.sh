#!/usr/bin/env bash
set -euo pipefail

journalctl --user -u augseg-coordinator.service -f
