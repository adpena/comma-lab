#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Track-A torch-vehicle D2 HNeRV adapter.
#
# Consumes a monolithic archive.zip member (prefer <base>.bin, else 0.bin/x) whose
# payload is the driver-emitted HNeRV archive. Supports vendored, D2 variable-level
# decoder blobs, and the additive FiLM pose section.
# PACT_RUNTIME_DEPENDENCY_ROOT=experiments/public_runtime_adapters/torch_vehicle_d2_hnerv_adapter
set -euo pipefail

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${PYTHON:-python}"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  base="${line%.*}"
  if [ -f "$DATA_DIR/${base}.bin" ]; then
    src="$DATA_DIR/${base}.bin"
  elif [ -f "$DATA_DIR/0.bin" ]; then
    src="$DATA_DIR/0.bin"
  elif [ -f "$DATA_DIR/x" ]; then
    src="$DATA_DIR/x"
  else
    echo "FATAL: no D2 HNeRV payload for ${line}: tried ${base}.bin, 0.bin, x" >&2
    exit 3
  fi
  dst="$OUTPUT_DIR/${base}.raw"
  echo "[torch-vehicle-d2-adapter] inflating $src -> $dst" >&2
  "$PYBIN" "$HERE/inflate.py" "$src" "$dst"
done < "$FILE_LIST"
