#!/bin/bash
# Build a contest-compliant archive and run the FULL e2e eval pipeline.
# This is the ONLY path that produces a valid score.
#
# Usage:
#   ./experiments/build_and_eval.sh \
#       --renderer submissions/robust_current/renderer.bin \
#       --masks submissions/robust_current/masks_crf50.mkv \
#       --poses submissions/robust_current/optimized_poses.pt \
#       --device mps
set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────
RENDERER=""
MASKS=""
POSES=""
DEVICE="mps"
WORK_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --renderer) RENDERER="$2"; shift 2;;
        --masks) MASKS="$2"; shift 2;;
        --poses) POSES="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --work-dir) WORK_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -z "$RENDERER" ] || [ -z "$MASKS" ] || [ -z "$POSES" ]; then
    echo "Usage: $0 --renderer <path> --masks <path> --poses <path> [--device mps|cuda]"
    exit 1
fi

# ── Validate inputs ──────────────────────────────────────────────────
for f in "$RENDERER" "$MASKS" "$POSES"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

RENDERER_SIZE=$(stat -f%z "$RENDERER" 2>/dev/null || stat -c%s "$RENDERER")
MASKS_SIZE=$(stat -f%z "$MASKS" 2>/dev/null || stat -c%s "$MASKS")
POSES_SIZE=$(stat -f%z "$POSES" 2>/dev/null || stat -c%s "$POSES")

echo "═══════════════════════════════════════════════════════════════════"
echo "BUILD & EVAL PIPELINE"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Renderer: $RENDERER ($RENDERER_SIZE bytes)"
echo "  Masks:    $MASKS ($MASKS_SIZE bytes)"
echo "  Poses:    $POSES ($POSES_SIZE bytes)"
echo "  Device:   $DEVICE"
echo ""

# ── Build archive ────────────────────────────────────────────────────
if [ -z "$WORK_DIR" ]; then
    WORK_DIR=$(mktemp -d)
fi
ARCHIVE_DIR="$WORK_DIR/archive"
INFLATED_DIR="$WORK_DIR/inflated"
mkdir -p "$ARCHIVE_DIR" "$INFLATED_DIR"

cp "$RENDERER" "$ARCHIVE_DIR/renderer.bin"
cp "$MASKS" "$ARCHIVE_DIR/masks.mkv"
cp "$POSES" "$ARCHIVE_DIR/optimized_poses.pt"

(cd "$ARCHIVE_DIR" && zip -9 -r "$WORK_DIR/archive.zip" .)
ARCHIVE_SIZE=$(stat -f%z "$WORK_DIR/archive.zip" 2>/dev/null || stat -c%s "$WORK_DIR/archive.zip")
RATE=$(python3 -c "print(f'{25 * $ARCHIVE_SIZE / 37545489:.4f}')")

echo "=== ARCHIVE ==="
echo "  Size: $ARCHIVE_SIZE bytes ($(echo "scale=1; $ARCHIVE_SIZE / 1024" | bc) KB)"
echo "  Rate term: $RATE"
unzip -l "$WORK_DIR/archive.zip" | grep -v "^Archive" | grep -v "^$" | sed 's/^/  /'
echo ""

# ── Inflate ──────────────────────────────────────────────────────────
echo "=== INFLATE ==="
echo "0.mkv" > "$WORK_DIR/video_names.txt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/upstream:$SCRIPT_DIR" \
    python3 -u "$SCRIPT_DIR/submissions/robust_current/inflate_renderer.py" \
    "$ARCHIVE_DIR" \
    "$INFLATED_DIR" \
    "$WORK_DIR/video_names.txt" 2>&1 | tail -5

# ── Verify output ────────────────────────────────────────────────────
RAW="$INFLATED_DIR/0.raw"
if [ ! -f "$RAW" ]; then
    echo "ERROR: 0.raw not generated!"
    ls -la "$INFLATED_DIR/"
    exit 1
fi
EXPECTED=$((1164 * 874 * 3 * 1200))
RAW_SIZE=$(stat -f%z "$RAW" 2>/dev/null || stat -c%s "$RAW")
if [ "$RAW_SIZE" != "$EXPECTED" ]; then
    echo "ERROR: Wrong frame count! $RAW_SIZE != $EXPECTED"
    exit 1
fi
echo "  Output: $RAW_SIZE bytes (correct)"
echo ""

# ── Evaluate via upstream scorer ─────────────────────────────────────
echo "=== UPSTREAM EVALUATE ==="
PYTHONPATH="$SCRIPT_DIR/upstream:$SCRIPT_DIR" \
    python3 -u "$SCRIPT_DIR/upstream/evaluate.py" \
    --submission-dir "$WORK_DIR" \
    --uncompressed-dir "$SCRIPT_DIR/upstream/videos/" \
    --video-names-file "$WORK_DIR/video_names.txt" \
    --device "$DEVICE" \
    --report "$WORK_DIR/report.txt" \
    --batch-size 4 2>&1 | grep -E "===|Average|Submission|Original|Compression|Final"

echo ""
echo "=== FULL REPORT ==="
cat "$WORK_DIR/report.txt" 2>/dev/null | tail -10
echo ""
echo "Work dir: $WORK_DIR"
echo "Archive: $WORK_DIR/archive.zip"
