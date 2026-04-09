#!/usr/bin/env bash
set -euo pipefail

BAT00_HOST="${BAT00_HOST:-adpena@bat00.local}"
BAT00_WSL_ROOT="${BAT00_WSL_ROOT:-/home/adpena/pact-side}"

cd "$(dirname "$0")/.."

payload=(
  pyproject.toml
  uv.lock
  src/comma_lab
  submissions/robust_current
  experiments/configs
  experiments/bat00_sync.sh
  experiments/run_bat00_smoke_job.sh
  experiments/launch_bat00_smokes.sh
  experiments/poll_bat00_runs.sh
)

COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar -czf - "${payload[@]}" \
  | base64 \
  | ssh "$BAT00_HOST" "wsl -e bash -lc 'set -euo pipefail; tmp=\"/tmp/pact-bat00-sync.tgz.b64\"; mkdir -p \"$BAT00_WSL_ROOT\"; cat > \"\$tmp\"; base64 -d \"\$tmp\" | tar -xzf - -C \"$BAT00_WSL_ROOT\"; rm -f \"\$tmp\"'"

echo "Synced payload to $BAT00_HOST:$BAT00_WSL_ROOT"
