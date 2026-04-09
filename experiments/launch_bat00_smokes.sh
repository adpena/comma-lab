#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <config-path-or-basename> [...]" >&2
  exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BAT00_HOST="${BAT00_HOST:-adpena@bat00.local}"
BAT00_WSL_ROOT="${BAT00_WSL_ROOT:-/home/adpena/pact-side}"
BAT00_RUN_ROOT="${BAT00_RUN_ROOT:-/home/adpena/bat00-runs}"
LOCAL_LOG_ROOT="${LOCAL_LOG_ROOT:-$REPO_ROOT/.omx/logs/bat00}"

mkdir -p "$LOCAL_LOG_ROOT"
cd "$SCRIPT_DIR"
bash ./bat00_sync.sh

launch_one() {
  local input="$1"
  local config_rel
  if [ -f "$input" ]; then
    config_rel="${input#../}"
  elif [ -f "$REPO_ROOT/experiments/configs/$input" ]; then
    config_rel="experiments/configs/$input"
  else
    echo "missing config: $input" >&2
    return 65
  fi
  local slug
  slug="$(basename "$config_rel" .env)"
  local log_path="$LOCAL_LOG_ROOT/${slug}.log"
  nohup ssh "$BAT00_HOST" \
    "wsl -e bash -lc 'cd \"$BAT00_WSL_ROOT\" && bash experiments/run_bat00_smoke_job.sh \"$slug\" \"$BAT00_WSL_ROOT/$config_rel\" \"$BAT00_WSL_ROOT\" \"$BAT00_RUN_ROOT\"'" \
    >"$log_path" 2>&1 < /dev/null &
  echo "$slug launched (local pid $!, log $log_path)"
}

for config in "$@"; do
  launch_one "$config"
done
