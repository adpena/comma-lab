#!/bin/bash
set -euo pipefail
# Iterative encoder experiment queue
# Tests encoder variants WITHOUT post-filter, just raw scorer
# Zero GPU cost — only CPU (ffmpeg encode + scorer eval)

UPSTREAM="workspace/upstream/comma_video_compression_challenge"
SCORER_VENV="$UPSTREAM/.venv/bin/python"
GT_VIDEO="$UPSTREAM/videos/0.mkv"
RESULTS_DIR="reports/raw/encoder_sweep_$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

echo "=== Encoder Sweep — $(date) ===" | tee "$RESULTS_DIR/summary.txt"

# Define variants to test
declare -A VARIANTS
VARIANTS[current]="scale=524:394:flags=lanczos|-svtav1-params film-grain=22:keyint=180:sharpness=1|-pix_fmt yuv420p"
VARIANTS[keyint_inf]="scale=524:394:flags=lanczos|-svtav1-params film-grain=22:keyint=-1:sharpness=1|-pix_fmt yuv420p"
VARIANTS[10bit]="scale=524:394:flags=lanczos|-svtav1-params film-grain=22:keyint=180:sharpness=1|-pix_fmt yuv420p10le"
VARIANTS[fg30_denoise0]="scale=524:394:flags=lanczos|-svtav1-params film-grain=30:film-grain-denoise=0:keyint=180:sharpness=1|-pix_fmt yuv420p"
VARIANTS[full_stack]="scale=524:394:flags=lanczos|-svtav1-params film-grain=30:film-grain-denoise=0:keyint=-1:sharpness=1|-pix_fmt yuv420p10le"
VARIANTS[crf33]="scale=524:394:flags=lanczos|-svtav1-params film-grain=22:keyint=180:sharpness=1|-pix_fmt yuv420p|-crf 33"
VARIANTS[crf33_10bit]="scale=524:394:flags=lanczos|-svtav1-params film-grain=22:keyint=180:sharpness=1|-pix_fmt yuv420p10le|-crf 33"
VARIANTS[fg15]="scale=524:394:flags=lanczos|-svtav1-params film-grain=15:keyint=180:sharpness=1|-pix_fmt yuv420p"
VARIANTS[fg30]="scale=524:394:flags=lanczos|-svtav1-params film-grain=30:keyint=180:sharpness=1|-pix_fmt yuv420p"

for name in "${!VARIANTS[@]}"; do
    echo ""
    echo "=== Testing: $name ==="
    IFS='|' read -ra PARTS <<< "${VARIANTS[$name]}"

    TMPDIR=$(mktemp -d)
    mkdir -p "$TMPDIR/archive" "$TMPDIR/inflated"

    # Encode (preset 4 for speed — we're testing codec params, not quality at max preset)
    CRF_ARG="-crf 34"
    for part in "${PARTS[@]}"; do
        if [[ "$part" == -crf* ]]; then
            CRF_ARG="$part"
        fi
    done

    ffmpeg -y -i "$GT_VIDEO" \
        -vf "${PARTS[0]}" \
        -c:v libsvtav1 -preset 4 $CRF_ARG \
        ${PARTS[1]} \
        ${PARTS[2]} -an \
        "$TMPDIR/archive/0.mkv" 2>/dev/null

    SIZE=$(stat -f%z "$TMPDIR/archive/0.mkv" 2>/dev/null || stat -c%s "$TMPDIR/archive/0.mkv")
    echo "  Size: $SIZE bytes ($(echo "scale=1; $SIZE/1024" | bc) KB)"

    # Create archive.zip
    (cd "$TMPDIR" && zip -j archive.zip archive/0.mkv > /dev/null)

    echo "  Archive: $(stat -f%z "$TMPDIR/archive.zip" 2>/dev/null || stat -c%s "$TMPDIR/archive.zip") bytes"

    echo "$name: size=$SIZE" >> "$RESULTS_DIR/summary.txt"

    rm -rf "$TMPDIR"
done

echo ""
echo "=== Summary ===" | tee -a "$RESULTS_DIR/summary.txt"
echo "Results in $RESULTS_DIR/summary.txt"
echo "Next: run scorer on promising variants to measure distortion impact"
