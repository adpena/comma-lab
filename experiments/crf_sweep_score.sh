#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# CRF Sweep: Encode + Inflate + Score for each CRF value
#
# Strategy: Option A — local CPU proxy scoring via upstream evaluate.py.
# Gives relative ranking between CRF values (~5 min each).
# Once we identify the top 1-2 CRFs, run auth eval on Modal.
#
# Includes CRF 35 and 37 for fine-grained resolution in the
# steepest part of the rate-distortion curve.
#
# Encodes directly with ffmpeg using the EXACT production settings
# from config.env (codec, preset, resolution, film-grain, keyint,
# color metadata). Only the CRF value changes per iteration.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROBUST="$PROJECT/submissions/robust_current"
UPSTREAM="$PROJECT/workspace/upstream/comma_video_compression_challenge"
GT_VIDEO="$UPSTREAM/videos/0.mkv"
POSTFILTER="$ROBUST/postfilter_int8.pt"

# Load production config
source "$ROBUST/config.env"

# Output directory for results
RESULTS_DIR="$PROJECT/reports/raw/crf_sweep_$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

# CRF values to sweep — fine resolution in the 33-38 range
CRF_VALUES="${CRF_VALUES:-30 32 33 34 35 36 37 38 40}"

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"

echo "============================================"
echo "CRF Sweep Scoring Pipeline"
echo "============================================"
echo "GT video:    $GT_VIDEO"
echo "Postfilter:  $POSTFILTER"
echo "Codec:       ${VIDEO_CODEC:-libsvtav1}"
echo "Resolution:  ${SCALE_W}x${SCALE_H}"
echo "Preset:      ${SVT_AV1_PRESET}"
echo "Params:      ${SVT_AV1_PARAMS}"
echo "CRF values:  $CRF_VALUES"
echo "Results dir: $RESULTS_DIR"
echo "============================================"
echo ""

# Validate prerequisites
for prereq in "$GT_VIDEO" "$POSTFILTER" "$PYTHON"; do
  if [ ! -f "$prereq" ]; then
    echo "ERROR: Not found: $prereq" >&2
    exit 1
  fi
done

# Summary file
SUMMARY="$RESULTS_DIR/summary.jsonl"
> "$SUMMARY"

for CRF in $CRF_VALUES; do
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  CRF $CRF"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  WORK="/tmp/crf_sweep_work_${CRF}"
  MKV_PATH="$WORK/0.mkv"
  ARCHIVE_PATH="$WORK/archive.zip"
  RAW_PATH="$WORK/inflated.raw"
  SUBMISSION_DIR="$WORK/submission"
  ENCODE_TIME=0

  # ── Step 1: Encode at target CRF ────────────────────────────────
  # Matches production: downscale to SCALE_WxSCALE_H, libsvtav1,
  # preset 0, film-grain=22, keyint=-1, color metadata.
  if [ -f "$ARCHIVE_PATH" ] && [ -s "$ARCHIVE_PATH" ]; then
    echo "[step 1] Re-using cached archive: $ARCHIVE_PATH"
  else
    rm -rf "$WORK"
    mkdir -p "$WORK"

    echo "[step 1] Encoding at CRF $CRF (${VIDEO_CODEC}, ${SCALE_W}x${SCALE_H})..."
    ENCODE_START=$SECONDS

    # Build the scale filter matching production exactly
    VF="scale=${SCALE_W}:${SCALE_H}:flags=${DOWNSCALE_FLAGS:-lanczos}:in_range=${SOURCE_COLOR_RANGE}:out_range=${SOURCE_COLOR_RANGE}:in_color_matrix=${SOURCE_COLOR_MATRIX}:out_color_matrix=${SOURCE_COLOR_MATRIX}:in_primaries=${SOURCE_COLOR_PRIMARIES}:out_primaries=${SOURCE_COLOR_PRIMARIES}:in_transfer=${SOURCE_COLOR_TRC}:out_transfer=${SOURCE_COLOR_TRC},format=yuv420p"

    if [ "${VIDEO_CODEC:-libsvtav1}" = "libsvtav1" ]; then
      "$FFMPEG_BIN" -y -hide_banner -loglevel warning \
        -i "$GT_VIDEO" \
        -vf "$VF" \
        -an -c:v libsvtav1 \
        -preset "$SVT_AV1_PRESET" \
        -crf "$CRF" \
        -svtav1-params "$SVT_AV1_PARAMS" \
        -color_range "$SOURCE_COLOR_RANGE" \
        -colorspace "$SOURCE_COLOR_MATRIX" \
        -color_primaries "$SOURCE_COLOR_PRIMARIES" \
        -color_trc "$SOURCE_COLOR_TRC" \
        -map_metadata -1 \
        "$MKV_PATH"
    else
      "$FFMPEG_BIN" -y -hide_banner -loglevel warning \
        -i "$GT_VIDEO" \
        -vf "$VF" \
        -an -c:v libx265 \
        -preset medium \
        -crf "$CRF" \
        -x265-params "keyint=-1:min-keyint=-1:scenecut=0:bframes=4:ref=3" \
        -color_range "$SOURCE_COLOR_RANGE" \
        -colorspace "$SOURCE_COLOR_MATRIX" \
        -color_primaries "$SOURCE_COLOR_PRIMARIES" \
        -color_trc "$SOURCE_COLOR_TRC" \
        "$MKV_PATH"
    fi

    ENCODE_TIME=$((SECONDS - ENCODE_START))
    MKV_BYTES=$(stat -f%z "$MKV_PATH")
    echo "[step 1] Encoded: $MKV_BYTES bytes (${ENCODE_TIME}s)"

    # ── Step 1b: Package archive.zip ─────────────────────────────
    echo "[step 1b] Packaging archive.zip..."
    ARCHIVE_TMP="$WORK/archive_tmp"
    mkdir -p "$ARCHIVE_TMP"
    cp "$MKV_PATH" "$ARCHIVE_TMP/0.mkv"
    cp "$POSTFILTER" "$ARCHIVE_TMP/postfilter_int8.pt"
    (cd "$ARCHIVE_TMP" && zip -0 -q "$ARCHIVE_PATH" 0.mkv postfilter_int8.pt)
    rm -rf "$ARCHIVE_TMP"
    echo "[step 1b] Archive: $(stat -f%z "$ARCHIVE_PATH") bytes"
  fi

  # ── Step 2: Inflate (decode + postfilter) ────────────────────────
  if [ -f "$RAW_PATH" ] && [ -s "$RAW_PATH" ]; then
    echo "[step 2] Re-using cached inflate: $RAW_PATH"
  else
    rm -f "$RAW_PATH"  # Remove stale zero-byte file if present
    echo "[step 2] Inflating (decode + postfilter)..."
    $PYTHON -c "
import sys
sys.path.insert(0, '$ROBUST')
sys.path.insert(0, '$PROJECT/src')
from inflate_postfilter import inflate_with_postfilter, load_postfilter_int8

model = load_postfilter_int8('$POSTFILTER', device='cpu')
n = inflate_with_postfilter(
    '$MKV_PATH', '$RAW_PATH', model,
    target_w=1164, target_h=874, device='cpu',
)
print(f'Inflated {n} frames')
"
  fi

  # ── Step 3: Score (upstream evaluate.py on CPU) ──────────────────
  echo "[step 3] Scoring with upstream evaluate.py..."
  rm -rf "$SUBMISSION_DIR"
  mkdir -p "$SUBMISSION_DIR/inflated"
  cp "$ARCHIVE_PATH" "$SUBMISSION_DIR/archive.zip"
  cp "$RAW_PATH" "$SUBMISSION_DIR/inflated/0.raw"

  REPORT="$SUBMISSION_DIR/report.txt"
  SCORE_START=$SECONDS
  $PYTHON "$UPSTREAM/evaluate.py" \
    --submission-dir "$SUBMISSION_DIR" \
    --uncompressed-dir "$UPSTREAM/videos" \
    --report "$REPORT" \
    --video-names-file "$UPSTREAM/public_test_video_names.txt" \
    --device cpu
  SCORE_TIME=$((SECONDS - SCORE_START))

  # ── Step 4: Parse and record ─────────────────────────────────────
  cp "$REPORT" "$RESULTS_DIR/report_crf${CRF}.txt"

  $PYTHON -c "
import json, re, sys

report = open('$REPORT').read()
patterns = {
    'pose_distortion': r'Average PoseNet Distortion:\s*([0-9.]+)',
    'seg_distortion': r'Average SegNet Distortion:\s*([0-9.]+)',
    'archive_bytes': r'Submission file size:\s*([0-9,]+) bytes',
    'original_bytes': r'Original uncompressed size:\s*([0-9,]+) bytes',
    'rate': r'Compression Rate:\s*([0-9.]+)',
    'score': r'Final score: .* =\s*([0-9.]+)',
}
result = {'crf': $CRF, 'encode_time': $ENCODE_TIME, 'score_time': $SCORE_TIME}
for key, pat in patterns.items():
    m = re.search(pat, report)
    if m:
        raw = m.group(1).replace(',', '')
        result[key] = int(raw) if key.endswith('bytes') else float(raw)
    else:
        print(f'WARNING: could not parse {key}', file=sys.stderr)
        result[key] = None

print(json.dumps(result))
with open('$SUMMARY', 'a') as f:
    f.write(json.dumps(result) + '\n')
"

  echo "  Done: CRF $CRF (score: ${SCORE_TIME}s)"
done

echo ""
echo "============================================"
echo "  SWEEP COMPLETE"
echo "============================================"
echo ""

# Print summary table
$PYTHON -c "
import json, sys

results = []
with open(sys.argv[1]) as f:
    for line in f:
        results.append(json.loads(line))

results.sort(key=lambda r: r['crf'])

print(f'{\"CRF\":>5} {\"Archive\":>12} {\"Rate\":>10} {\"PoseNet\":>10} {\"SegNet\":>10} {\"Score\":>10}')
print('-' * 63)
for r in results:
    print(f'{r[\"crf\"]:>5} {r.get(\"archive_bytes\",0):>12,} {r.get(\"rate\",0):>10.6f} {r.get(\"pose_distortion\",0):>10.6f} {r.get(\"seg_distortion\",0):>10.6f} {r.get(\"score\",0):>10.4f}')

best = min(results, key=lambda r: r.get('score', 999))
print()
print(f'Best CRF: {best[\"crf\"]} with score {best.get(\"score\",\"?\"):.4f}')

prod = next((r for r in results if r['crf'] == 34), None)
if prod and best['crf'] != 34:
    delta = best['score'] - prod['score']
    print(f'Delta vs CRF 34: {delta:+.4f}')

print()
print('NOTE: CPU/PyAV proxy scores (not authoritative).')
print('Relative ranking is valid. Run auth eval on top 1-2 candidates.')
" "$SUMMARY"

echo ""
echo "Full results: $SUMMARY"
echo "Reports: $RESULTS_DIR/report_crf*.txt"
