#!/usr/bin/env bash
# Forked PR101 inflate.sh for cross-paradigm ADMM-x-Op1-finalizer lane.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

mkdir -p "$OUTPUT_DIR"

INFLATE_BROTLI_SPEC="${INFLATE_BROTLI_SPEC:-brotli==1.2.0}"
INFLATE_TORCH_SPEC="${INFLATE_TORCH_SPEC:-torch==2.5.1+cu124}"
INFLATE_NUMPY_SPEC="${INFLATE_NUMPY_SPEC:-numpy==2.4.4}"
UV_BIN="${UV_BIN:-$(command -v uv || echo /usr/local/bin/uv)}"
if [ ! -x "$UV_BIN" ]; then
  echo "FATAL: uv not on PATH (UV_BIN=$UV_BIN); the canonical inflate-time" >&2
  echo "       env requires uv. Bootstrap with scripts/ensure_remote_uv.sh." >&2
  exit 1
fi

UV_WITH_INFLATE_DEPS=(
  --with "$INFLATE_BROTLI_SPEC"
  --with "$INFLATE_TORCH_SPEC"
  --with "$INFLATE_NUMPY_SPEC"
)

echo "[cross-paradigm-inflate] uv specs: brotli=$INFLATE_BROTLI_SPEC torch=$INFLATE_TORCH_SPEC numpy=$INFLATE_NUMPY_SPEC" >&2

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/x"
  if [ ! -f "$SRC" ]; then
    SRC="${DATA_DIR}/${BASE}.bin"
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "$UV_BIN" run --no-project "${UV_WITH_INFLATE_DEPS[@]}" python "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
