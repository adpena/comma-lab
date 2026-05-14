#!/usr/bin/env bash
# hdm8_film_grain_sidecar inflate: PR106/HDM8 HNeRV decode plus fixed
# deterministic postfilter from postfilter_config.json.
# Reads <data_dir>/<base>.bin or a single-member x archive payload, writes
# <output_dir>/<base>.raw (uint8 RGB, (N,H,W,3)).
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
  PRIMARY_SRC="${DATA_DIR}/${BASE}.bin"
  X_SRC="${DATA_DIR}/x"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  if [ -f "$PRIMARY_SRC" ] && [ -f "$X_SRC" ] && [ "$PRIMARY_SRC" != "$X_SRC" ]; then
    echo "ERROR: ambiguous archive payloads found: ${PRIMARY_SRC} and ${X_SRC}" >&2
    exit 1
  elif [ -f "$PRIMARY_SRC" ]; then
    SRC="$PRIMARY_SRC"
  elif [ -f "$X_SRC" ]; then
    SRC="$X_SRC"
  else
    echo "ERROR: neither ${PRIMARY_SRC} nor ${X_SRC} found" >&2
    exit 1
  fi

  printf "Inflating %s ... " "$line"
  "${RUNNER[@]}" "$SRC" "$DST"
done < "$FILE_LIST"
