#!/usr/bin/env bash
# Exact-replay adapter for public PR106 `belt_and_suspenders` archives.
#
# The public runtime expects `<base>.bin`; our deterministic x-repack variants
# charge the same payload under member `x` to reduce ZIP overhead. This adapter
# preserves the public decoder and only adapts the member lookup contract.
# PACT_RUNTIME_DEPENDENCY_ROOT=experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source"
INFLATE_PY="$SOURCE_ROOT/submissions/belt_and_suspenders/inflate.py"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

if [ ! -f "$INFLATE_PY" ]; then
  echo "FATAL: PR106 public inflate.py missing: $INFLATE_PY" >&2
  exit 2
fi

PYBIN="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYBIN" ]; then
  PYBIN="$(command -v python3 || command -v python)"
fi

"$PYBIN" - <<'PY'
import brotli
print(f"[pr106-adapter] brotli={getattr(brotli, '__version__', 'unknown')}")
PY

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  if [ ! -f "$SRC" ] && [ -f "${DATA_DIR}/x" ]; then
    SRC="${DATA_DIR}/x"
  fi
  if [ ! -f "$SRC" ]; then
    echo "FATAL: neither ${DATA_DIR}/${BASE}.bin nor ${DATA_DIR}/x exists" >&2
    exit 3
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"
  echo "[pr106-adapter] inflating ${SRC} -> ${DST}"
  PYTHONPATH="$SOURCE_ROOT/submissions/belt_and_suspenders/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYBIN" "$INFLATE_PY" "$SRC" "$DST"
done < "$FILE_LIST"
