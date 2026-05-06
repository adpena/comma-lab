#!/usr/bin/env bash
# Download Lightning checkpoint → build archive → eval locally.
# One command: bash download_and_eval.sh [run_name]
#
# Usage:
#   bash submissions/robust_current/download_and_eval.sh lightning_full
#   bash submissions/robust_current/download_and_eval.sh lightning_full --skip-score

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/../.." && pwd)"
LIGHTNING_TARGET="${LIGHTNING_TARGET:?Set LIGHTNING_TARGET to an SSH config alias or user-qualified remote target}"
LIGHTNING_SSH_KEY="${LIGHTNING_SSH_KEY:-}"
REMOTE_RESULTS="${LIGHTNING_REMOTE_RESULTS:-\${LIGHTNING_REMOTE_RESULTS}}"
if [ -z "$REMOTE_RESULTS" ] || [ "$REMOTE_RESULTS" = '${LIGHTNING_REMOTE_RESULTS}' ]; then
    echo "Set LIGHTNING_REMOTE_RESULTS to the remote results directory"
    exit 2
fi
SCP_ARGS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [ -n "$LIGHTNING_SSH_KEY" ]; then
    SCP_ARGS=(-i "$LIGHTNING_SSH_KEY" "${SCP_ARGS[@]}")
fi

RUN_NAME="${1:-lightning_full}"
SKIP_SCORE="${2:-}"

echo "=== Download & Eval Pipeline ==="
echo "  Run: ${RUN_NAME}"
echo "  Remote: ${LIGHTNING_TARGET}"
echo ""

# 1. Create local directory for this checkpoint
CKPT_DIR="${REPO_ROOT}/experiments/results/${RUN_NAME}"
mkdir -p "$CKPT_DIR"

# 2. Download checkpoint + meta from Lightning
echo "[1/5] Downloading checkpoint from Lightning..."
scp "${SCP_ARGS[@]}" \
    "${LIGHTNING_TARGET}:${REMOTE_RESULTS}/postfilter_${RUN_NAME}_best_int8.pt" \
    "$CKPT_DIR/postfilter_int8.pt" 2>/dev/null

scp "${SCP_ARGS[@]}" \
    "${LIGHTNING_TARGET}:${REMOTE_RESULTS}/postfilter_${RUN_NAME}_best_meta.json" \
    "$CKPT_DIR/meta.json" 2>/dev/null

echo "  Downloaded: $(ls -lh "$CKPT_DIR/postfilter_int8.pt" | awk '{print $5}')"
echo "  Meta: $(cat "$CKPT_DIR/meta.json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'ep={d[\"epoch\"]} scorer={d[\"scorer\"]:.4f}')")"

# 3. Compute checkpoint hash
CKPT_MD5=$(md5 -q "$CKPT_DIR/postfilter_int8.pt")
echo "  MD5: $CKPT_MD5"

# 4. Promote to submissions/ and rebuild archive
echo ""
echo "[2/5] Promoting checkpoint..."
cp "$CKPT_DIR/postfilter_int8.pt" "$SELF_DIR/postfilter_int8.pt"
echo "  Copied to submissions/robust_current/postfilter_int8.pt"

echo ""
echo "[3/5] Rebuilding archive (compress.sh)..."
COMMA_CHALLENGE_ROOT="${REPO_ROOT}/workspace/upstream/comma_video_compression_challenge" \
    bash "$SELF_DIR/compress.sh" 2>&1 | tail -3

ARCHIVE_SIZE=$(stat -f%z "$SELF_DIR/archive.zip" 2>/dev/null || stat --printf="%s" "$SELF_DIR/archive.zip")
echo "  Archive: ${ARCHIVE_SIZE} bytes"

# 5. Verify archive contains postfilter
echo ""
echo "[4/5] Verifying archive compliance..."
# Per CLAUDE.md non-negotiable (feedback_pipefail_grep_q_trap): under
# `set -euo pipefail`, `unzip -l ... | grep -q PATTERN` triggers SIGPIPE on
# unzip when grep stops reading the first match — pipefail then aborts the
# script with no useful error. Capture-first idiom avoids the trap.
ARCHIVE_LIST=$(unzip -l "$SELF_DIR/archive.zip" 2>&1)
if echo "$ARCHIVE_LIST" | grep -q postfilter_int8.pt; then
    echo "  postfilter_int8.pt: BUNDLED ✓"
else
    echo "  ERROR: postfilter_int8.pt NOT in archive!"
    exit 1
fi

# 6. Run eval (unless --skip-score)
if [ "$SKIP_SCORE" = "--skip-score" ]; then
    echo ""
    echo "[5/5] Skipping score (--skip-score)"
    echo ""
    echo "To score manually:"
    echo "  python submissions/robust_current/runner.py evaluate \\"
    echo "    --upstream-dir workspace/upstream/comma_video_compression_challenge \\"
    echo "    --run-name ${RUN_NAME}_eval --skip-compress --device cuda"
else
    echo ""
    echo "[5/5] Running eval via runner.py..."
    EVAL_DEVICE="${AUTH_EVAL_DEVICE:-cuda}"
    case "$EVAL_DEVICE" in
      cuda) ;;
      *)
        echo "FATAL: scoring from this helper is CUDA-only; use --skip-score for local package checks" >&2
        exit 64
        ;;
    esac
    python3 "$SELF_DIR/runner.py" evaluate \
        --upstream-dir "${REPO_ROOT}/workspace/upstream/comma_video_compression_challenge" \
        --run-name "${RUN_NAME}_eval" \
        --skip-compress \
        --device "$EVAL_DEVICE"
fi

echo ""
echo "=== Done ==="
echo "  Checkpoint: $CKPT_DIR/postfilter_int8.pt (md5 $CKPT_MD5)"
echo "  Archive: $SELF_DIR/archive.zip (${ARCHIVE_SIZE} bytes)"
echo "  Run dir: $SELF_DIR/eval_runs/${RUN_NAME}_eval/"
