#!/usr/bin/env bash
# Static HDM3 exact-eval packet wrapper. Delegates to the reviewed
# PR106-R2 PR101-grammar runtime; no score claim is made here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
fi
ADAPTER_INFLATE="${REPO_ROOT}/submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh"
if [ ! -x "$ADAPTER_INFLATE" ]; then
  echo "FATAL: HDM3 adapter inflate.sh missing or not executable: $ADAPTER_INFLATE" >&2
  exit 66
fi
exec "$ADAPTER_INFLATE" "$@"
