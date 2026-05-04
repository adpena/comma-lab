#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"; OUTPUT_DIR="$2"; FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  printf "Inflating %s -> %s\n" "$BASE" "$DST"
  python "${HERE}/inflate.py" "$DATA_DIR" "$BASE" "$DST"
done < "$FILE_LIST"
