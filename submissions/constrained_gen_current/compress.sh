#!/usr/bin/env bash
set -euo pipefail
# Constrained generation compress: extract masks + pose targets from GT video.
#
# This produces a ~8KB archive containing:
#   - masks.bin: LZMA-compressed segmentation masks (~239 bytes)
#   - pose_targets.bin: float16 pose targets (~7KB)
#   - seed.bin: noise seed (8 bytes)
#   - meta.json: reproducibility metadata
#
# Compare to renderer archive (~150KB) or codec archive (~150KB).
# Rate reduction: 0.10 -> ~0.001 = 0.099 score points saved.
#
# Usage:
#   bash compress.sh [--upstream-root <path>] [--device <cuda|mps|cpu>]

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SELF_DIR/../../upstream" 2>/dev/null && pwd || true)"
UPSTREAM_ROOT="${COMMA_CHALLENGE_ROOT:-$DEFAULT_ROOT}"

if [ -z "${UPSTREAM_ROOT}" ] || [ ! -d "${UPSTREAM_ROOT}/models" ]; then
    echo "ERROR: Could not find upstream challenge root. Set COMMA_CHALLENGE_ROOT." >&2
    exit 1
fi

UV_BIN="${UV_BIN:-uv}"
DEVICE="${DEVICE:-cpu}"
NOISE_SEED="${NOISE_SEED:-42}"
VIDEO_NAMES_FILE="${VIDEO_NAMES_FILE:-$UPSTREAM_ROOT/public_test_video_names.txt}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upstream-root) UPSTREAM_ROOT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --seed) NOISE_SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "[constrained-gen compress] Upstream: $UPSTREAM_ROOT"
echo "[constrained-gen compress] Device: $DEVICE"
echo "[constrained-gen compress] Seed: $NOISE_SEED"

# Back up existing archive before overwriting
ARCHIVE_ZIP="$SELF_DIR/archive.zip"
if [ -f "$ARCHIVE_ZIP" ]; then
    BACKUP_NAME="$SELF_DIR/archive_$(date +%Y%m%dT%H%M%S).zip"
    cp "$ARCHIVE_ZIP" "$BACKUP_NAME"
    echo "[compress] Backed up existing archive to $BACKUP_NAME" >&2
fi

# Run the compression script (try uv first, fall back to python3)
if command -v "$UV_BIN" &>/dev/null; then
    "$UV_BIN" run --with av --with torch --with numpy python "$SELF_DIR/compress.py" \
        --upstream-root "$UPSTREAM_ROOT" \
        --device "$DEVICE" \
        --seed "$NOISE_SEED" \
        --video-names-file "$VIDEO_NAMES_FILE" \
        --output-dir "$SELF_DIR/archive_staging"
else
    python3 "$SELF_DIR/compress.py" \
        --upstream-root "$UPSTREAM_ROOT" \
        --device "$DEVICE" \
        --seed "$NOISE_SEED" \
        --video-names-file "$VIDEO_NAMES_FILE" \
        --output-dir "$SELF_DIR/archive_staging"
fi

# Package into archive.zip
echo "[compress] Packaging archive.zip..."
cd "$SELF_DIR/archive_staging"
zip -r "$ARCHIVE_ZIP" .
cd "$SELF_DIR"

ARCHIVE_SIZE=$(stat --format=%s "$ARCHIVE_ZIP" 2>/dev/null || stat -f%z "$ARCHIVE_ZIP" 2>/dev/null || echo "?")
echo "[compress] archive.zip: $ARCHIVE_SIZE bytes"
echo "[compress] Done."
