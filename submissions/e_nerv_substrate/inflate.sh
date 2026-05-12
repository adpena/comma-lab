#!/usr/bin/env bash
# E-NeRV substrate inflate wrapper. Per CLAUDE.md HNeRV parity discipline
# lesson 9: contest-hermetic runtime closure (torch + brotli only).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_DIR="${1?missing archive_dir}"
OUTPUT_DIR="${2?missing output_dir}"
FILE_LIST="${3?missing file_list}"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r base || [[ -n "$base" ]]; do
  [[ -z "$base" ]] && continue
  src="$ARCHIVE_DIR/$base"
  dst="$OUTPUT_DIR/$base.raw"
  exec_status=0
  python3 "$HERE/inflate.py" "$src" "$dst" || exec_status=$?
  if [[ "$exec_status" -ne 0 ]]; then
    echo "[inflate.sh] inflate.py failed for $base (rc=$exec_status)" >&2
    exit "$exec_status"
  fi
done < "$FILE_LIST"
