#!/usr/bin/env bash
# sane_hnerv contest-compliant inflate (PR-95-parity packet; Wave N+45 BIND)
# Contract: $1=archive_dir $2=output_dir $3=file_list  (per Catalog #146)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
