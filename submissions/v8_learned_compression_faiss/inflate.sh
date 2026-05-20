#!/usr/bin/env bash
# V8 learned-compression Faiss local inflate wrapper.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${1?missing archive_dir}"
OUTPUT_DIR="${2?missing output_dir}"
FILE_LIST="${3?missing file_list}"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r base || [[ -n "$base" ]]; do
  [[ -z "$base" ]] && continue
  if [[ "$base" = /* || "$base" = .* || "$base" = *"/"* || "$base" = *".."* || "$base" = *"\\"* ]]; then
    echo "unsafe V8 file_list entry: $base" >&2
    exit 2
  fi
  src="$ARCHIVE_DIR/$base"
  if [[ ! -f "$src" ]]; then
    if [[ -f "$ARCHIVE_DIR/${base%.*}.v8f" ]]; then
      src="$ARCHIVE_DIR/${base%.*}.v8f"
    else
      src="$ARCHIVE_DIR/${base%.*}.bin"
    fi
  fi
  if [[ ! -f "$src" ]]; then
    echo "missing V8 archive member for $base" >&2
    exit 2
  fi
  dst="$OUTPUT_DIR/${base%.*}.raw"
  "${PYTHON:-python3}" "$HERE/inflate.py" "$src" "$dst"
done < "$FILE_LIST"
