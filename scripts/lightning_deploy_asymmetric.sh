#!/bin/bash
# Lightning T4 deployment for asymmetric warp renderer training.
#
# Pre-requisites on Lightning:
#   - DALI installed
#   - Upstream scorer files at /home/zeus/content/upstream (includes videos/, models/)
#   - SSH working
#   NOTE: No precomputed frames needed — decode on target with DALI (council decision)
#
# Run from local machine:
#   bash scripts/lightning_deploy_asymmetric.sh
#
# This script:
#   1. Syncs the updated source tree to Lightning
#   2. Runs DALI mask validation (P0 blocker #4)
#   3. Runs smoke test (P3 Karpathy protocol)
#   4. Launches asymmetric warp training (48h budget)

set -euo pipefail

LIGHTNING_USER="${LIGHTNING_USER:?Set LIGHTNING_USER env var (e.g. s_XXXXX)}"
LIGHTNING_HOST="${LIGHTNING_USER}@ssh.lightning.ai"
SSH_KEY="${HOME}/.ssh/lightning_rsa"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
SCP="scp -i $SSH_KEY -o StrictHostKeyChecking=no"
RSYNC="rsync -avz -e 'ssh -i $SSH_KEY -o StrictHostKeyChecking=no'"

REMOTE_ROOT="/home/zeus/content/pact"
UPSTREAM="/home/zeus/content/upstream"
# NOTE: No PRECOMPUTED variable — decode on target with DALI (council decision)

echo "============================================"
echo "  Lightning T4: Asymmetric Warp Deploy"
echo "  $(date)"
echo "============================================"
echo ""

# ---------------------------------------------------------------
# Step 1: Sync source tree
# ---------------------------------------------------------------
echo "[1/4] Syncing source tree to Lightning..."

# Sync src/ and experiments/ (the code that changed in 27 commits)
eval $RSYNC --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    --exclude='*.egg-info' --exclude='results/' --exclude='precomputed_local/' \
    src/ "$LIGHTNING_HOST:$REMOTE_ROOT/src/"

eval $RSYNC --exclude='__pycache__' --exclude='*.pyc' --exclude='results/' \
    --exclude='precomputed_local/' \
    experiments/train_renderer_fridrich.py \
    experiments/validate_dali_masks.py \
    "$LIGHTNING_HOST:$REMOTE_ROOT/experiments/"

echo "  Source tree synced."
echo ""

# ---------------------------------------------------------------
# Step 2: DALI mask validation (P0 blocker #4)
# ---------------------------------------------------------------
echo "[2/4] DALI mask validation (P0 blocker #4)..."
echo "  Comparing PyAV vs DALI SegNet argmax masks..."
echo ""

$SSH $LIGHTNING_HOST "
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate cloudspace 2>/dev/null

cd $REMOTE_ROOT
export PYTHONPATH=src:$UPSTREAM

python experiments/validate_dali_masks.py \
    --video-dir $UPSTREAM/videos \
    --video-names $UPSTREAM/public_test_video_names.txt \
    --weights-dir $UPSTREAM/models \
    --max-frames 100 \
    --max-videos 3 \
    --device cuda \
    --threshold 0.01
" 2>&1 | tee /tmp/lightning_dali_validation.log

# Check for explicit PASS string (not absence-of-FAIL, which could mask silent errors)
if grep -q "^PASS:" /tmp/lightning_dali_validation.log; then
    echo ""
    echo "  DALI validation passed."
elif grep -q "^FAIL:" /tmp/lightning_dali_validation.log; then
    echo ""
    echo "ABORT: DALI mask validation failed. Do not proceed."
    echo "See /tmp/lightning_dali_validation.log for details."
    exit 1
else
    echo ""
    echo "ABORT: DALI validation produced no PASS/FAIL verdict. Possible crash."
    echo "See /tmp/lightning_dali_validation.log for details."
    exit 1
fi
echo ""

# ---------------------------------------------------------------
# Step 3: Smoke test (P3 Karpathy protocol)
# ---------------------------------------------------------------
echo "[3/4] Smoke test (P3 Karpathy protocol)..."
echo ""

$SSH $LIGHTNING_HOST "
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate cloudspace 2>/dev/null

cd $REMOTE_ROOT
export PYTHONPATH=src:$UPSTREAM

python experiments/train_renderer_fridrich.py \
    --validate-smoke \
    --pair-mode asymmetric
" 2>&1 | tee /tmp/lightning_smoke_test.log

if grep -q "ALL 5 CHECKS PASSED" /tmp/lightning_smoke_test.log; then
    echo ""
    echo "  Smoke test passed."
else
    echo ""
    echo "ABORT: Smoke test failed. Check /tmp/lightning_smoke_test.log"
    exit 1
fi

echo ""

# ---------------------------------------------------------------
# Step 4: Launch asymmetric warp training
# ---------------------------------------------------------------
echo "[4/4] Launching asymmetric warp training..."
echo "  pair_mode=asymmetric, 48h budget, T4 GPU"
echo ""

# Use nohup + screen so training survives SSH disconnect
$SSH $LIGHTNING_HOST "
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate cloudspace 2>/dev/null

cd $REMOTE_ROOT
export PYTHONPATH=src:$UPSTREAM

# Create results dir
mkdir -p experiments/results/fridrich_renderer

# Launch in screen session (survives disconnect)
screen -dmS asymmetric_train bash -c '
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate cloudspace 2>/dev/null
cd $REMOTE_ROOT
export PYTHONPATH=src:$UPSTREAM

python experiments/train_renderer_fridrich.py \
    --pair-mode asymmetric \
    --epochs 10000 \
    --batch-size 4 \
    --lr 2e-4 \
    --embed-dim 6 \
    --base-ch 36 \
    --mid-ch 60 \
    --motion-hidden 32 \
    --max-flow-px 20.0 \
    --max-residual 20.0 \
    --seg-boundary 0.005 \
    --pose-boundary 0.02 \
    --rho-init 10.0 \
    --rho-growth 1.02 \
    --tv-weight 0.05 \
    --flow-weight 0.0 \
    --rate-weight 0.01 \
    --target-bytes 200000 \
    --gate-reg-weight 0.1 \
    --even-pairs-only \
    --device cuda \
    --seed 42 \
    --checkpoint-every 500 \
    --eval-every 200 \
    --log-every 25 \
    --max-hours 48.0 \
    --phase2-mse-weight 0.1 \
    2>&1 | tee experiments/results/fridrich_renderer/train_asymmetric.log
'

echo 'Training launched in screen session: asymmetric_train'
echo 'Monitor: ssh ... screen -r asymmetric_train'
echo 'Log: experiments/results/fridrich_renderer/train_asymmetric.log'
"

echo ""
echo "============================================"
echo "  DEPLOYMENT COMPLETE"
echo "  $(date)"
echo ""
echo "  Monitor training:"
echo "    ssh -i $SSH_KEY $LIGHTNING_HOST 'screen -r asymmetric_train'"
echo ""
echo "  Tail log:"
echo "    ssh -i $SSH_KEY $LIGHTNING_HOST 'tail -f $REMOTE_ROOT/experiments/results/fridrich_renderer/train_asymmetric.log'"
echo ""
echo "  Fetch checkpoints:"
echo "    scp -i $SSH_KEY $LIGHTNING_HOST:$REMOTE_ROOT/experiments/results/fridrich_renderer/best_*.pt ."
echo "============================================"
