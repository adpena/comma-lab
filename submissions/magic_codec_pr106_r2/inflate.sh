#!/usr/bin/env bash
# magic_codec_pr106_r2 inflate: PR106 HNeRV decoder wrapped in the magic-codec
# meta-grammar. Reads the magic-codec envelope, dispatches by primitive_id
# byte, runs the canonical inflate decoder.
#
# CLAUDE.md compliance:
#   * set -euo pipefail per check_shell_set_e_present
#   * positional args $1 archive_dir, $2 output_dir, $3 file_list
#   * NO scorer load (strict-scorer-rule)
#   * NO MPS / no /tmp paths
#   * NO_NVDEC_NEEDED — tensor-side decode only
#
# Substrate-engineering waiver: this runtime is the magic-codec dispatch
# wrapper around PR106's r2 inflate runtime. The magic-codec inflate.py is
# capped at ≤200 LOC per HNeRV parity discipline lesson 4 (substrate
# engineering waiver); the wrapped PR106 inflate is unchanged.
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
