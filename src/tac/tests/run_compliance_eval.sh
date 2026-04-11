#!/usr/bin/env bash
# Run the EXACT official evaluation pipeline locally.
# This is what comma.ai will run on their machine.
# If this passes, the submission WILL work.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${REPO}/workspace/upstream/comma_video_compression_challenge"
SUBMISSION="${REPO}/submissions/robust_current"

echo "=== COMPLIANCE EVALUATION ==="
echo "Submission: ${SUBMISSION}"
echo "Upstream:   ${UPSTREAM}"
echo ""

# Pre-flight checks
echo "[1/6] Pre-flight checks..."
for f in "${SUBMISSION}/archive.zip" "${SUBMISSION}/inflate.sh" "${SUBMISSION}/config.env"; do
    if [ ! -f "$f" ]; then
        echo "FAIL: Missing $f" >&2
        exit 1
    fi
done
echo "  All required files present."

# Check archive size
ARCHIVE_SIZE=$(stat -f%z "${SUBMISSION}/archive.zip" 2>/dev/null || stat -c%s "${SUBMISSION}/archive.zip")
echo "  Archive size: ${ARCHIVE_SIZE} bytes"
if [ "$ARCHIVE_SIZE" -gt 1048576 ]; then
    echo "  WARNING: Archive > 1MB (${ARCHIVE_SIZE} bytes)"
fi

# Check inflate.sh is executable
if [ ! -x "${SUBMISSION}/inflate.sh" ]; then
    chmod +x "${SUBMISSION}/inflate.sh"
    echo "  Fixed: inflate.sh was not executable"
fi

# Check postfilter exists
if [ -f "${SUBMISSION}/postfilter_int8.pt" ]; then
    PF_SIZE=$(stat -f%z "${SUBMISSION}/postfilter_int8.pt" 2>/dev/null || stat -c%s "${SUBMISSION}/postfilter_int8.pt")
    echo "  Post-filter: ${PF_SIZE} bytes"
fi

# Clean previous eval artifacts
echo ""
echo "[2/6] Cleaning previous artifacts..."
rm -rf "${SUBMISSION}/archive" "${SUBMISSION}/inflated" "${SUBMISSION}/report.txt"

# Run official evaluate.sh
echo ""
echo "[3/6] Running official evaluate.sh..."
cd "${UPSTREAM}"
PYTHONUNBUFFERED=1 bash evaluate.sh \
    --submission-dir "${SUBMISSION}" \
    --device "${1:-mps}"

# Check report was generated
echo ""
echo "[4/6] Checking report..."
if [ ! -f "${SUBMISSION}/report.txt" ]; then
    echo "FAIL: report.txt not generated" >&2
    exit 1
fi

echo ""
echo "[5/6] Report contents:"
cat "${SUBMISSION}/report.txt"

# Extract score
SCORE=$(grep "Final score" "${SUBMISSION}/report.txt" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
echo ""
echo "[6/6] RESULT"
echo "============================================"
echo "  OFFICIAL SCORE: ${SCORE}"
echo "============================================"

# Compare to known baselines
echo ""
echo "  vs leaderboard #2 (neural_inflate): 1.89"
echo "  vs our canonical proxy:             ~1.49"
