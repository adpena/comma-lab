#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${COMMA_CHALLENGE_ROOT:-$ROOT/workspace/upstream/comma_video_compression_challenge}"

if [ ! -f "$UPSTREAM_ROOT/evaluate.sh" ]; then
  echo "ERROR: upstream repo not found at $UPSTREAM_ROOT" >&2
  exit 1
fi

cd "$UPSTREAM_ROOT"

echo "[smoke] exact_current"
bash evaluate.sh --submission-dir ./submissions/exact_current --device "${COMMA_DEVICE:-cpu}"

echo
echo "[smoke] robust_current package"
bash ./submissions/robust_current/compress.sh
