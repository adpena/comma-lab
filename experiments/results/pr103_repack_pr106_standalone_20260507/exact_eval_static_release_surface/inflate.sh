#!/usr/bin/env bash
# Static PR103-on-PR106 exact-eval packet wrapper.
# Delegates to the reviewed self-contained runtime in submissions/pr103_pr106_final_runtime.
# No score claim is made here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
fi
ADAPTER_INFLATE="${REPO_ROOT}/submissions/pr103_pr106_final_runtime/inflate.sh"
if [ ! -x "$ADAPTER_INFLATE" ]; then
  echo "FATAL: PR103-PR106 adapter inflate.sh missing or not executable: $ADAPTER_INFLATE" >&2
  exit 66
fi
exec "$ADAPTER_INFLATE" "$@"
