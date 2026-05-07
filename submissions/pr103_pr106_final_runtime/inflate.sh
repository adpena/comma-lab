#!/usr/bin/env bash
# PR103-on-PR106 final runtime adapter.
# Reads either <data_dir>/<base>.bin or x, writes <output_dir>/<base>.raw.
# NO_NVDEC_NEEDED - pure HNeRV tensor decode + bicubic upsample.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

PYBIN="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: managed Python is not executable: $PYBIN" >&2
  exit 4
fi

cd "$ROOT"
"$PYBIN" -m "submissions.${SUB_NAME}.inflate" --dependency-check

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  BASE_BIN="${DATA_DIR}/${BASE}.bin"
  X_MEMBER="${DATA_DIR}/x"

  if [ -f "$BASE_BIN" ] && [ -f "$X_MEMBER" ]; then
    echo "FATAL: ambiguous PR103-PR106 payload members; both ${BASE_BIN} and ${X_MEMBER} exist" >&2
    exit 5
  fi
  if [ -f "$BASE_BIN" ]; then
    SRC="$BASE_BIN"
  elif [ -f "$X_MEMBER" ]; then
    SRC="$X_MEMBER"
  else
    echo "FATAL: neither ${BASE_BIN} nor ${X_MEMBER} exists" >&2
    exit 3
  fi

  DST="${OUTPUT_DIR}/${BASE}.raw"
  echo "[pr103-pr106-final] inflating ${SRC} -> ${DST}"
  "$PYBIN" -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
