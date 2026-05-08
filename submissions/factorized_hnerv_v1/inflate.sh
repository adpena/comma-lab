#!/usr/bin/env bash
# factorized_hnerv_v1 inflate: SVD-factorized HNeRV decoder + PR-style latents.
# Reads <data_dir>/<base>.bin, writes <output_dir>/<base>.raw (uint8 RGB, (N,H,W,3)).
# NO_NVDEC_NEEDED — purely tensor-side decode + bicubic upsample, no DALI/NVDEC.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

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
  cd "$ROOT"
  python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
