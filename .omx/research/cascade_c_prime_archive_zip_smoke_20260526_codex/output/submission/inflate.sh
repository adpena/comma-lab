#!/bin/bash
# SPDX-License-Identifier: MIT
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
PYBIN="${PYBIN:-python}"
exec "$PYBIN" "$HERE/inflate.py" "$ARCHIVE_DIR" "$OUTPUT_DIR" "$FILE_LIST"
