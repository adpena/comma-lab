#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_BIN="$ROOT/.tooling/node_modules/.bin"
export PATH="$TOOLS_BIN:$PATH"
cd "$ROOT"
if ! command -v omx >/dev/null 2>&1; then
  echo "ERROR: omx not found. Run bash start.sh first or install local tooling." >&2
  exit 1
fi
printf 'Paste this prompt when OMX opens:\n  %s/PROMPT.md\n\n' "$ROOT"
exec omx --madmax --high "$@"
