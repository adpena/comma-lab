#!/usr/bin/env bash
# pr106_lapose_atom_sidecar inflate: PR106 r2 HNeRV decoder + LAPose
# inverse-dynamics motion-atom stream sidecar (per-atom typed corrections;
# pose-axis target; research_only=true).
# Reads <data_dir>/<base>.bin, writes <output_dir>/<base>.raw (uint8 RGB).
# NO_NVDEC_NEEDED — pure tensor + numpy decode + grid_sample atom application.
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
