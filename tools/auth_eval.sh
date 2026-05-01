#!/usr/bin/env bash
# auth_eval.sh — Single-command authoritative evaluation
#
# Handles the FULL pipeline: clean state → build archive → inflate → score
# No stale data possible. No manual steps.
#
# Usage:
#   ./tools/auth_eval.sh <checkpoint.pt> [--crf 34] [--tag myrun]
#
# Examples:
#   ./tools/auth_eval.sh .backups/postfilter_standard_h64_long2500_v3_best_int8.pt
#   ./tools/auth_eval.sh .backups/postfilter_standard_h64_long2500_v3_best_int8.pt --crf 35
#   ./tools/auth_eval.sh experiments/postfilter_weights/postfilter_psd_standard_h64_best_int8.pt --tag psd_ep809
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$REPO_ROOT/workspace/upstream/comma_video_compression_challenge"
SUBMISSION="$REPO_ROOT/submissions/robust_current"
CONFIG_ENV="$SUBMISSION/config.env"

# Parse args
CHECKPOINT="${1:?Usage: auth_eval.sh <checkpoint.pt> [--crf N] [--tag NAME]}"
CRF=""
TAG="auth_eval_$(date +%Y%m%dT%H%M%S)"

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --crf) CRF="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Validate inputs
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [ ! -f "$UPSTREAM/evaluate.py" ]; then
    echo "ERROR: Upstream evaluate.py not found at $UPSTREAM" >&2
    exit 1
fi

CHECKPOINT_SIZE=$(stat -f%z "$CHECKPOINT" 2>/dev/null || stat -c%s "$CHECKPOINT" 2>/dev/null)
if [ -z "$CHECKPOINT_SIZE" ]; then
    echo "ERROR: Cannot determine size of $CHECKPOINT (stat failed)" >&2
    exit 1
fi
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
echo "=== AUTH EVAL: $TAG ==="
echo "  Checkpoint: $CHECKPOINT ($CHECKPOINT_SIZE bytes)"
echo "  CRF: ${CRF:-default from config.env}"

# Create isolated workspace — no stale state possible
WORK="$(mktemp -d "${TMPDIR:-/tmp}/auth_eval.XXXXXX")"
trap "rm -rf \"$WORK\"" EXIT
echo "  Workspace: $WORK"

ARCHIVE_DIR="$WORK/archive"
INFLATED_DIR="$WORK/inflated"
REPORT="$WORK/report.txt"
mkdir -p "$ARCHIVE_DIR" "$INFLATED_DIR"

# Step 1: Get the video
if [ -n "$CRF" ] && [ -f "/tmp/crf_sweep/crf${CRF}.mkv" ]; then
    # Use pre-encoded CRF sweep video
    cp "/tmp/crf_sweep/crf${CRF}.mkv" "$ARCHIVE_DIR/0.mkv"
    echo "  Video: CRF $CRF from sweep cache"
elif [ -n "$CRF" ]; then
    echo "ERROR: CRF $CRF video not found at /tmp/crf_sweep/crf${CRF}.mkv" >&2
    echo "Run the CRF sweep first or use the default CRF." >&2
    exit 1
else
    # Use existing archive video (CRF from config.env)
    if [ -f "$SUBMISSION/archive.zip" ]; then
        unzip -o "$SUBMISSION/archive.zip" "0.mkv" -d "$ARCHIVE_DIR" > /dev/null 2>&1 || true
    fi
    if [ ! -f "$ARCHIVE_DIR/0.mkv" ]; then
        echo "ERROR: No video source. Provide --crf or ensure archive.zip exists." >&2
        exit 1
    fi
    echo "  Video: from existing archive"
fi

VIDEO_SIZE=$(stat -f%z "$ARCHIVE_DIR/0.mkv" 2>/dev/null || stat -c%s "$ARCHIVE_DIR/0.mkv")

# Step 2: Bundle checkpoint into archive (use python zipfile per
# feedback_zip_dep_bootstrap_trap — `zip` shell binary not present on PyTorch container images)
cp "$CHECKPOINT" "$ARCHIVE_DIR/postfilter_int8.pt"
python3 -c "
import os, zipfile
src = '$ARCHIVE_DIR'
out = '$WORK/archive.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, _, files in os.walk(src):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, src)
            z.write(full, arcname=rel)
"
ARCHIVE_SIZE=$(stat -f%z "$WORK/archive.zip" 2>/dev/null || stat -c%s "$WORK/archive.zip")
echo "  Archive: $ARCHIVE_SIZE bytes (video=$VIDEO_SIZE + model=$CHECKPOINT_SIZE)"

# Step 3: Copy checkpoint to submission dir (inflate.sh reads from there too)
cp "$CHECKPOINT" "$SUBMISSION/postfilter_int8.pt"

# Step 4: Inflate
echo "  Inflating..."
INFLATE_START=$(date +%s)
bash "$SUBMISSION/inflate.sh" \
    "$ARCHIVE_DIR" \
    "$INFLATED_DIR" \
    "$UPSTREAM/public_test_video_names.txt" \
    2>&1 | grep -v "^$" | tail -3
INFLATE_END=$(date +%s)
INFLATE_SEC=$((INFLATE_END - INFLATE_START))
echo "  Inflate: ${INFLATE_SEC}s"

# Verify inflate produced output
if [ ! -f "$INFLATED_DIR/0.raw" ]; then
    echo "ERROR: Inflate failed — no output file" >&2
    exit 1
fi
INFLATED_SIZE=$(stat -f%z "$INFLATED_DIR/0.raw" 2>/dev/null || stat -c%s "$INFLATED_DIR/0.raw")
EXPECTED=3662409600
if [ "$INFLATED_SIZE" -lt "$((EXPECTED - 1000))" ]; then
    echo "ERROR: Inflate incomplete — $INFLATED_SIZE bytes (expected ~$EXPECTED)" >&2
    exit 1
fi

# Step 5: Swap into submission dir for evaluate.py
# (evaluate.py reads submission_dir/inflated/ and submission_dir/archive.zip)
LOCKFILE="$SUBMISSION/.auth_eval.lock"
if [ -f "$LOCKFILE" ]; then
    echo "ERROR: Another auth_eval is modifying $SUBMISSION (lockfile exists)" >&2
    exit 1
fi
trap "rm -f \"$LOCKFILE\"; rm -rf \"$WORK\"" EXIT
touch "$LOCKFILE"
rm -rf "$SUBMISSION/inflated"
mv "$INFLATED_DIR" "$SUBMISSION/inflated"
cp "$WORK/archive.zip" "$SUBMISSION/archive.zip"

# Step 6: Score
echo "  Scoring..."
SCORE_START=$(date +%s)
cd "$UPSTREAM"
if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: Upstream venv not found at $UPSTREAM/.venv/bin/python" >&2
    exit 1
fi
.venv/bin/python evaluate.py \
    --submission-dir "$SUBMISSION" \
    --uncompressed-dir videos \
    --video-names-file public_test_video_names.txt \
    --report "$REPORT" \
    2>&1 | grep -v "it \[" | tail -10
SCORE_END=$(date +%s)
SCORE_SEC=$((SCORE_END - SCORE_START))
cd "$REPO_ROOT"

echo "  Score: ${SCORE_SEC}s"
echo ""
echo "=== RESULT ==="
cat "$REPORT" 2>/dev/null || echo "(no report generated)"
echo ""

# Step 7: Save report with tag
REPORTS_DIR="$REPO_ROOT/reports/raw/auth_eval"
mkdir -p "$REPORTS_DIR"
cp "$REPORT" "$REPORTS_DIR/${TAG}_report.txt" 2>/dev/null
echo "Report saved: reports/raw/auth_eval/${TAG}_report.txt"
echo "Total time: $((INFLATE_SEC + SCORE_SEC))s"
