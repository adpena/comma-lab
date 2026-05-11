#!/usr/bin/env bash
# pr106_cool_chic_residual_sidecar inflate (research-only scaffold).
# Reads <data_dir>/<base>.bin (magic 0xFD + format_id 0x11 + PR106 + Cool-Chic residual),
# writes <output_dir>/<base>.raw uint8 RGB (N, H=874, W=1164, 3).
# NO_NVDEC_NEEDED — pure tensor + numpy decode.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$HERE/src:$HERE:${PYTHONPATH:-}"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  printf "Inflating %s ... " "$line"
  "$PYTHON_BIN" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
