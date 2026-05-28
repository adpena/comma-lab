#!/usr/bin/env bash
# pact_nerv_selector_v3 contest-compliant inflate (PACT-NERV-FULL-MAIN-WAVE 2026-05-27)
# Contract: $1=archive_dir $2=output_dir $3=file_list
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
