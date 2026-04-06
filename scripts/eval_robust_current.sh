#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ROOT="$(cd "$ROOT/workspace/upstream/comma_video_compression_challenge" 2>/dev/null && pwd || true)"
UPSTREAM_ROOT="${COMMA_CHALLENGE_ROOT:-$DEFAULT_ROOT}"
DEVICE="${COMMA_DEVICE:-cpu}"

if [ -z "${UPSTREAM_ROOT}" ] || [ ! -f "${UPSTREAM_ROOT}/evaluate.sh" ]; then
  echo "ERROR: Could not find upstream challenge root. Set COMMA_CHALLENGE_ROOT." >&2
  exit 1
fi

cd "$UPSTREAM_ROOT"
bash evaluate.sh --submission-dir "./submissions/robust_current" --device "$DEVICE"
