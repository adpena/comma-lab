#!/usr/bin/env bash
# NSCS03 end-to-end Ballé joint codec inflate.sh
# Catalog #146 contract: <inflate.sh archive_dir output_dir file_list>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCHIVE_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

"${PYTHON:-python3}" "$HERE/inflate.py" "$ARCHIVE_DIR" "$OUTPUT_DIR" "$FILE_LIST"
