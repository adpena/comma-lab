#!/usr/bin/env bash
set -euo pipefail

# Source-sized public replay shim for PR102. Do not install packages here.
# PACT_RUNTIME_DEPENDENCY_ROOT = experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local dir="$HERE"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/experiments/contest_auth_eval.py" ]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

if [ -n "${PACT_REPO_ROOT:-}" ]; then
  REPO_ROOT="$PACT_REPO_ROOT"
else
  REPO_ROOT="$(find_repo_root)" || {
    echo "ERROR: could not find repo root; set PACT_REPO_ROOT" >&2
    exit 1
  }
fi
PYTHON="${PACT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
PUBLIC_SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source"
RUNTIME_SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1"

if [ "$#" -ne 3 ]; then
  echo "ERROR: expected DATA_DIR OUTPUT_DIR FILE_LIST arguments" >&2
  exit 2
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python not executable: $PYTHON" >&2
  exit 1
fi
if [ ! -d "$PUBLIC_SOURCE_ROOT" ] || [ ! -d "$RUNTIME_SOURCE_ROOT" ]; then
  echo "ERROR: PR102 public source runtime missing" >&2
  echo "PUBLIC_SOURCE_ROOT=$PUBLIC_SOURCE_ROOT" >&2
  echo "RUNTIME_SOURCE_ROOT=$RUNTIME_SOURCE_ROOT" >&2
  exit 1
fi

for module in brotli numpy torch; do
  "$PYTHON" - "$module" <<'PY'
import importlib.util
import sys

module = sys.argv[1]
if importlib.util.find_spec(module) is None:
    raise SystemExit(f"ERROR: required PR102 runtime dependency missing: {module}")
PY
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PUBLIC_SOURCE_ROOT:$RUNTIME_SOURCE_ROOT:${PYTHONPATH:-}"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="$DATA_DIR/${BASE}.bin"
  DST="$OUTPUT_DIR/${BASE}.raw"
  if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found" >&2
    exit 1
  fi
  cd "$PUBLIC_SOURCE_ROOT"
  "$PYTHON" -m "submissions.hnerv_lc_v2_scale095_rplus1.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
