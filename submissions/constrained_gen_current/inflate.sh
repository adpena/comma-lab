#!/usr/bin/env bash
set -euo pipefail
# Constrained generation inflate: runs gradient descent from masks + pose targets.
# The archive contains ~8KB of masks, pose targets, and noise seed.
# Inflate time is dominated by optimization (~27 min budget on T4 GPU).

ARCHIVE_DIR="${1:?archive dir required}"
INFLATED_DIR="${2:?inflated dir required}"
VIDEO_NAMES_FILE="${3:?video names file required}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-uv}"

mkdir -p "$INFLATED_DIR"

echo "[constrained-gen inflate] Archive: $ARCHIVE_DIR"
echo "[constrained-gen inflate] Output:  $INFLATED_DIR"

"$UV_BIN" run --with av --with torch --with numpy python "$SELF_DIR/inflate.py" \
    "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
