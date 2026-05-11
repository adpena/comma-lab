#!/usr/bin/env bash
# pr106_latent_sidecar inflate: PR106 HNeRV decoder + per-pair latent sidecar.
# Reads <data_dir>/<base>.bin, writes <output_dir>/<base>.raw (uint8 RGB, (N,H,W,3)).
# NO_NVDEC_NEEDED — purely tensor-side decode + bicubic upsample, no DALI/NVDEC.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

if [ -f "$HERE/inflate.py" ] && [ -d "$HERE/src" ]; then
  export PYTHONPATH="$HERE/src:$HERE:${PYTHONPATH:-}"
  RUNNER=("$PYTHON_BIN" "$HERE/inflate.py")
else
  ROOT="$(cd "$HERE/../.." && pwd)"
  SUB_NAME="$(basename "$HERE")"
  cd "$ROOT"
  RUNNER=("$PYTHON_BIN" -m "submissions.${SUB_NAME}.inflate")
fi

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "${RUNNER[@]}" "$SRC" "$DST"
done < "$FILE_LIST"
