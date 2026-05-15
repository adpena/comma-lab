#!/usr/bin/env bash
# NSCS01 nullspace-split-renderer contest-compliant inflate wrapper.
# Per CLAUDE.md HNeRV parity discipline lesson 9: contest-hermetic runtime
# closure (torch + brotli only). Per Catalog #146: 3-positional-arg signature.
# Per Catalog #163: set -euo pipefail.
# Per CLAUDE.md "Strict scorer rule": no scorer at inflate time.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1?missing archive_dir}"
OUTPUT_DIR="${2?missing output_dir}"
FILE_LIST="${3?missing file_list}"
mkdir -p "$OUTPUT_DIR"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
