#!/bin/bash
# Authoritative renderer evaluation on Lightning T4.
#
# Mirrors the Modal auth_eval function from modal_asymmetric_warp_deploy.py
# but runs on Lightning AI infrastructure. Use when Lightning credits reset.
#
# Pre-requisites on Lightning:
#   - Upstream scorer files at /home/zeus/content/upstream (includes videos/, models/)
#   - SSH key at ~/.ssh/lightning_rsa
#   - LIGHTNING_USER env var set
#
# Usage:
#   bash scripts/lightning_auth_eval_renderer.sh
#   bash scripts/lightning_auth_eval_renderer.sh --checkpoint renderer_epoch02000.pt
#   bash scripts/lightning_auth_eval_renderer.sh --checkpoint /local/path/to/renderer.bin
#   bash scripts/lightning_auth_eval_renderer.sh --checkpoint renderer_best.pt --archive-size 180000
#
# The script:
#   1. Syncs latest source + checkpoint to Lightning
#   2. Runs experiments/auth_eval_renderer.py on Lightning T4
#   3. Downloads results JSON

set -euo pipefail

# ── Configuration ──
LIGHTNING_USER="${LIGHTNING_USER:?Set LIGHTNING_USER env var (e.g. s_XXXXX)}"
LIGHTNING_HOST="${LIGHTNING_USER}@ssh.lightning.ai"
SSH_KEY="${HOME}/.ssh/lightning_rsa"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
SCP="scp -i $SSH_KEY -o StrictHostKeyChecking=no"
RSYNC="rsync -avz -e 'ssh -i $SSH_KEY -o StrictHostKeyChecking=no'"

REMOTE_ROOT="/home/zeus/content/pact"
UPSTREAM="/home/zeus/content/upstream"
EVAL_DIR="/tmp/auth_eval_renderer_$(date +%Y%m%dT%H%M%S)"

# ── Parse arguments ──
CHECKPOINT="experiments/results/fridrich_renderer/renderer_best.pt"
ARCHIVE_SIZE_ARG=""
BATCH_SIZE="16"
LOCAL_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --archive-size)
            ARCHIVE_SIZE_ARG="--archive-size-bytes $2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--checkpoint <path>] [--archive-size <bytes>] [--batch-size <n>]"
            exit 1
            ;;
    esac
done

# Determine if checkpoint is a local file that needs uploading
# vs a path that already exists on Lightning
if [[ -f "$CHECKPOINT" ]]; then
    LOCAL_CHECKPOINT="$CHECKPOINT"
    CKPT_BASENAME="$(basename "$CHECKPOINT")"
    REMOTE_CHECKPOINT="$EVAL_DIR/$CKPT_BASENAME"
    CKPT_SIZE=$(stat -f%z "$LOCAL_CHECKPOINT" 2>/dev/null || stat --printf="%s" "$LOCAL_CHECKPOINT")
    CKPT_MD5=$(md5 -q "$LOCAL_CHECKPOINT" 2>/dev/null || md5sum "$LOCAL_CHECKPOINT" | awk '{print $1}')
elif [[ "$CHECKPOINT" == /* ]]; then
    # Absolute path on Lightning (e.g. /home/zeus/content/...)
    REMOTE_CHECKPOINT="$CHECKPOINT"
    CKPT_BASENAME="$(basename "$CHECKPOINT")"
    CKPT_SIZE="(remote)"
    CKPT_MD5="(remote)"
else
    # Relative path — assume it's under REMOTE_ROOT
    REMOTE_CHECKPOINT="$REMOTE_ROOT/$CHECKPOINT"
    CKPT_BASENAME="$(basename "$CHECKPOINT")"
    CKPT_SIZE="(remote)"
    CKPT_MD5="(remote)"
fi

echo "============================================"
echo "  Lightning T4: Renderer Auth Eval"
echo "  $(date)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Size: $CKPT_SIZE bytes"
echo "  MD5: $CKPT_MD5"
echo "============================================"
echo ""

# ── Step 1: Sync source + checkpoint ──
echo "[1/3] Syncing source tree + checkpoint to Lightning..."

# Sync src/tac/ (model definitions, export utils)
eval $RSYNC --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    --exclude='*.egg-info' --exclude='results/' --exclude='precomputed_local/' \
    src/ "$LIGHTNING_HOST:$REMOTE_ROOT/src/"

# Sync the auth eval script
eval $RSYNC --exclude='__pycache__' --exclude='*.pyc' --exclude='results/' \
    experiments/auth_eval_renderer.py \
    "$LIGHTNING_HOST:$REMOTE_ROOT/experiments/"

# Create eval dir on remote
$SSH $LIGHTNING_HOST "mkdir -p $EVAL_DIR"

# Upload local checkpoint if needed
if [[ -n "$LOCAL_CHECKPOINT" ]]; then
    echo "  Uploading checkpoint: $LOCAL_CHECKPOINT -> $REMOTE_CHECKPOINT"
    $SCP "$LOCAL_CHECKPOINT" "$LIGHTNING_HOST:$REMOTE_CHECKPOINT"
fi

echo "  Source tree synced."
echo ""

# ── Step 2: Run auth eval on Lightning T4 ──
echo "[2/3] Running auth eval on Lightning T4..."
echo "  Remote checkpoint: $REMOTE_CHECKPOINT"
echo ""

$SSH $LIGHTNING_HOST "
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate cloudspace 2>/dev/null

cd $REMOTE_ROOT
export PYTHONPATH=src:$UPSTREAM

# Verify GPU
python -c 'import torch; print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}\")' 2>&1

# Run auth eval
python experiments/auth_eval_renderer.py \
    --checkpoint $REMOTE_CHECKPOINT \
    --upstream-dir $UPSTREAM \
    --device cuda \
    --batch-size $BATCH_SIZE \
    --output-dir $EVAL_DIR \
    $ARCHIVE_SIZE_ARG \
    2>&1
" 2>&1 | tee /tmp/lightning_auth_eval_renderer.log

echo ""

# ── Step 3: Download results ──
echo "[3/3] Downloading results..."

# Find and download the results JSON
CKPT_STEM="${CKPT_BASENAME%.*}"
RESULT_JSON="/tmp/auth_eval_renderer_${CKPT_STEM}.json"

$SCP "$LIGHTNING_HOST:$EVAL_DIR/auth_eval_*.json" "$RESULT_JSON" 2>/dev/null || true

if [[ -f "$RESULT_JSON" ]]; then
    echo ""
    echo "============================================"
    echo "  RESULTS"
    echo "============================================"
    # Pretty-print key fields from JSON
    python3 -c "
import json, sys
with open('$RESULT_JSON') as f:
    r = json.load(f)
print(f'  PoseNet dist:     {r[\"avg_posenet_dist\"]:.8f}')
print(f'  SegNet dist:      {r[\"avg_segnet_dist\"]:.8f}')
print(f'  Archive size:     {r[\"archive_size_bytes\"]:,} bytes')
print(f'  Rate:             {r[\"rate\"]:.8f}')
print()
print(f'  Score breakdown:')
print(f'    100*seg       = {r[\"score_seg\"]:.4f}')
print(f'    sqrt(10*pose) = {r[\"score_pose\"]:.4f}')
print(f'    25*rate       = {r[\"score_rate\"]:.4f}')
print(f'  FINAL SCORE:      {r[\"final_score\"]:.4f}')
print(f'  Time:             {r[\"elapsed_seconds\"]:.0f}s')
" 2>/dev/null || cat "$RESULT_JSON"
    echo "============================================"
    echo ""
    echo "  Results saved: $RESULT_JSON"
    echo "  Full log: /tmp/lightning_auth_eval_renderer.log"
else
    echo "  No results JSON downloaded -- check log for errors:"
    tail -20 /tmp/lightning_auth_eval_renderer.log
fi

# ── Cleanup remote eval dir ──
$SSH $LIGHTNING_HOST "rm -rf $EVAL_DIR" 2>/dev/null || true

echo ""
echo "============================================"
echo "  DONE — $(date)"
echo "============================================"
