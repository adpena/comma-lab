#!/usr/bin/env bash
# Carmack-Hotz strip-everything contest-compliant inflate (NSCS06 2026-05-15).
# Contract: $1=archive_dir $2=output_dir $3=file_list (Catalog #146).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
