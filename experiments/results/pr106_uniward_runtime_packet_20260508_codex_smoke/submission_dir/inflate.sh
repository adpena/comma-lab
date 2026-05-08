#!/usr/bin/env bash
# Self-contained PR106 inflate wrapper for experiments/results runtime packets.
# The upstream PR106 wrapper assumes submissions/<name>/ package layout; this
# packet keeps the PR106 Python decoder byte-identical but calls it by path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="${PR106_UNIWARD_PYTHON:-${PYTHON:-}}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in "$PWD/.venv/bin/python" "$HERE/../../../../.venv/bin/python" python python3; do
    if [[ "$candidate" == */* ]]; then
      if [[ -x "$candidate" ]]; then
        PYTHON_BIN="$candidate"
        break
      fi
    elif command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "FATAL: no Python interpreter found for PR106 UNIWARD inflate" >&2
  exit 127
fi

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [[ ! -f "$SRC" ]] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "$PYTHON_BIN" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
