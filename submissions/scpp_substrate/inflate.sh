#!/usr/bin/env bash
# scpp_substrate inflate: 88-94K-param SC++ self-compressed renderer.
# Reads <data_dir>/<base>.bin, writes <output_dir>/<base>.raw (uint8 RGB, (N,H,W,3)).
# NO_NVDEC_NEEDED — purely tensor-side decode + bicubic upsample, no DALI/NVDEC.
# Per CLAUDE.md "Forbidden in-place edits to public PR intake clones": this is
# an internal submission directory, not a public-PR intake clone.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "$PYTHON_BIN" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
