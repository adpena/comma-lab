#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <data_dir> <out_dir> <file_list>" >&2
  exit 2
fi
DATA_DIR="$1"; OUT_DIR="$2"; FILE_LIST="$3"
mkdir -p "$OUT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Ensure brotli is available (only dep beyond the upstream pyproject.toml).
"$PYTHON_BIN" -c "import brotli" 2>/dev/null || \
  "$PYTHON_BIN" -m pip install --quiet brotli 2>/dev/null || true

"$PYTHON_BIN" "$HERE/inflate.py" "$DATA_DIR" "$OUT_DIR" "$FILE_LIST"
