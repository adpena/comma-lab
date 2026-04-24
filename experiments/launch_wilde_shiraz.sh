#!/bin/bash
# Launch WILDE and SHIRAZ on two A100 instances (Vast.ai)
# Estimated: ~21h each, ~$12.60 each, ~$25.20 total
set -euo pipefail

echo "=== WILDE + SHIRAZ A/B TEST DEPLOYMENT ==="
echo "Date: $(date)"
echo ""

# Common training args (shared between WILDE and SHIRAZ)
COMMON_ARGS="
    --tto-frames experiments/results/tto_v7_hinge_500/tto_frames.pt
    --checkpoint /nonexistent
    --gt-poses experiments/results/gt_poses.pt
    --upstream upstream/
    --device cuda
    --base-ch 32 --mid-ch 48 --motion-hidden 24 --depth 1
    --pose-dim 6 --use-dsconv --use-dilation
    --padding-mode replicate
    --eval-roundtrip
    --ema-decay 0.997
    --use-per-class-weights
    --use-swa
    --use-texture-loss --texture-loss-weight 0.5
    --use-linf-penalty --linf-weight 0.01
    --use-markov-loss --markov-weight 0.1
    --checkpoint-every 100 --eval-every 50 --log-every 25
    --seed 42
"

# WILDE-specific args
WILDE_ARGS="
    --output-dir experiments/results/wilde
    --segnet-loss-mode hinge --hinge-margin 1.0
    --error-boost 9.0 --error-boost-phase3 49.0
    --freeze-motion-phase2 --freeze-renderer-phase3
    --pose-weight 10.0 --seg-weight 100.0 --pixel-weight 0.1
    --phase1-epochs 600 --phase2-epochs 880 --phase3-epochs 200
    --phase1-lr 1e-3 --phase2-lr 3e-4 --phase3-lr 1e-4
    --phase1-batch-size 16 --phase2-batch-size 8 --phase3-batch-size 8
"

# SHIRAZ-specific args
SHIRAZ_ARGS="
    --output-dir experiments/results/shiraz
    --loss-mode focal_ste --focal-gamma 2.0
    --segnet-loss-mode hinge --hinge-margin 1.0
    --error-boost 1.0
    --hard-frame-ratio 0.3 --error-replay-every 100
    --pose-weight 10.0 --seg-weight 100.0 --pixel-weight 0.1
    --phase1-epochs 400 --phase2-epochs 1080 --phase3-epochs 200
    --phase1-lr 1e-3 --phase2-lr 3e-4 --phase3-lr 1e-4
    --phase1-batch-size 16 --phase2-batch-size 8 --phase3-batch-size 8
"

echo "WILDE command:"
echo "  PYTHONPATH=src:upstream python3 -u experiments/train_distill.py $COMMON_ARGS $WILDE_ARGS"
echo ""
echo "SHIRAZ command:"
echo "  PYTHONPATH=src:upstream python3 -u experiments/train_distill.py $COMMON_ARGS $SHIRAZ_ARGS"
echo ""
echo "Deploy: create 2x A100 instances on Vast.ai, rsync code, launch each."
