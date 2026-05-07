#!/usr/bin/env bash
# Deterministic WR01 static release-surface wrapper.
# Delegates to the audited PR106 runtime without mutating the public-intake tree.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
fi
SOURCE_INFLATE="${REPO_ROOT}/experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/inflate.sh"
if [ ! -x "$SOURCE_INFLATE" ]; then
  echo "FATAL: delegated inflate.sh is missing or not executable: $SOURCE_INFLATE" >&2
  exit 66
fi
exec "$SOURCE_INFLATE" "$@"
