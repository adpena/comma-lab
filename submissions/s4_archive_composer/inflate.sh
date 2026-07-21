#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_DIR="${1:?archive directory required}"
INFLATED_DIR="${2:?inflated directory required}"
VIDEO_NAMES_FILE="${3:?video names file required}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$INFLATED_DIR"
while IFS= read -r video_name; do
  [ -z "$video_name" ] && continue
  stem="${video_name%.*}"
  "${PYTHON:-python3}" "$HERE/inflate.py" "$ARCHIVE_DIR/0.bin" "$INFLATED_DIR/$stem.raw"
done < "$VIDEO_NAMES_FILE"
