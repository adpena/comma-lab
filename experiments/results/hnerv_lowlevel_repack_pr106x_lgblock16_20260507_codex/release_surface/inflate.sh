#!/usr/bin/env bash
# Deterministic HNeRV low-level static release-surface wrapper.
# Delegates to the reviewed PR106 x-member exact-replay adapter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
fi
SOURCE_INFLATE="${REPO_ROOT}/experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh"
if [ ! -x "$SOURCE_INFLATE" ]; then
  echo "FATAL: delegated inflate.sh is missing or not executable: $SOURCE_INFLATE" >&2
  exit 66
fi
exec "$SOURCE_INFLATE" "$@"
