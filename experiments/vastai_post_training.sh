#!/bin/bash
# Post-training automation: runs on Vast.ai 4090 after float training completes.
# Chains: Phase 2 (reduced) → QAT → Pose TTO → FP4 export → download
#
# Usage: ssh into Vast.ai, then:
#   cd /workspace/pact && bash experiments/vastai_post_training.sh
#
# Or launch remotely:
#   ssh -p PORT root@HOST 'cd /workspace/pact && nohup bash experiments/vastai_post_training.sh > post_training.log 2>&1 &'
set -euo pipefail

export PYTHONPATH="src:upstream:$PWD"
RESULTS="/workspace/pact/experiments/results"
OVERNIGHT="$RESULTS/overnight_small_renderer"

echo "═══════════════════════════════════════════════════════════════════"
echo "POST-TRAINING PIPELINE"
echo "Started: $(date)"
echo "═══════════════════════════════════════════════════════════════════"

# ── Step 0: Check Phase 1 checkpoint exists ───────────────────────────
if [ ! -f "$OVERNIGHT/distill_latest.pt" ]; then
    echo "ERROR: No checkpoint found at $OVERNIGHT/distill_latest.pt"
    echo "Wait for Phase 1 to complete first."
    exit 1
fi
echo "Checkpoint: $(ls -la $OVERNIGHT/distill_latest.pt)"

# ── Step 1: Kill any running training ─────────────────────────────────
echo ""
echo "── Step 1: Kill existing training ──"
pkill -f "train_distill.py" 2>/dev/null && echo "  Killed existing training" || echo "  No training running"
sleep 2

# ── Step 2: Restart Phase 2 with reduced epochs + Fridrich losses ─────
echo ""
echo "── Step 2: Phase 2 (1000 epochs, scorer-guided + Fridrich) ──"
python3 -u experiments/train_distill.py \
    --tto-frames experiments/results/tto_v7_hinge_500/tto_frames.pt \
    --checkpoint /nonexistent \
    --gt-poses experiments/results/gt_poses.pt \
    --upstream upstream/ \
    --output-dir "$OVERNIGHT" \
    --base-ch 24 --mid-ch 32 --depth 1 --pose-dim 6 --use-dsconv \
    --device cuda \
    --eval-roundtrip \
    --segnet-loss-mode hinge --hinge-margin 0.5 \
    --pose-weight 10.0 --seg-weight 100.0 --pixel-weight 0.1 \
    --phase1-epochs 0 --phase2-epochs 1000 --phase3-epochs 500 \
    --phase2-lr 3e-4 --phase3-lr 1e-4 \
    --phase2-batch-size 4 --phase3-batch-size 4 \
    --use-texture-loss --texture-loss-weight 0.5 \
    --use-linf-penalty --linf-weight 0.01 \
    --checkpoint-every 200 --eval-every 100 --log-every 25 \
    --skip-phase1 \
    --resume "$OVERNIGHT/distill_latest.pt" \
    --seed 42 \
    2>&1 | tee "$RESULTS/phase2_reduced.log"

echo ""
echo "Phase 2+3 complete: $(date)"

# ── Step 3: QAT Fine-Tune ────────────────────────────────────────────
echo ""
echo "── Step 3: QAT Fine-Tune (FP4-robust) ──"
BEST_PT="$OVERNIGHT/distill_best.pt"
if [ ! -f "$BEST_PT" ]; then
    BEST_PT="$OVERNIGHT/distill_latest.pt"
fi

python3 -u experiments/qat_finetune.py \
    --checkpoint "$BEST_PT" \
    --upstream upstream/ \
    --output-dir "$RESULTS/qat_fp4" \
    --device cuda \
    --base-ch 24 --mid-ch 32 --pose-dim 6 --use-dsconv \
    --skip-int8-warmup \
    --fp4-epochs 200 \
    --lr 3e-5 \
    --batch-size 4 \
    2>&1 | tee "$RESULTS/qat_finetune.log"

echo ""
echo "QAT complete: $(date)"

# ── Step 4: Pose TTO on QAT model ────────────────────────────────────
echo ""
echo "── Step 4: Pose TTO (1000 steps on QAT renderer) ──"
QAT_BEST="$RESULTS/qat_fp4/qat_best_float.pt"
if [ ! -f "$QAT_BEST" ]; then
    QAT_BEST="$RESULTS/qat_fp4/renderer_fp4.bin"
fi

python3 -u experiments/optimize_poses.py \
    --checkpoint "$QAT_BEST" \
    --device cuda \
    --steps 1000 --lr 0.01 --batch-pairs 50 \
    --eval-roundtrip \
    --upstream upstream/ \
    --video upstream/videos/0.mkv \
    --output-dir "$RESULTS/pose_tto_final" \
    2>&1 | tee "$RESULTS/pose_tto_final.log"

echo ""
echo "Pose TTO complete: $(date)"

# ── Step 5: Summary ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "POST-TRAINING COMPLETE: $(date)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Artifacts ready for download:"
echo "  Float best:  $BEST_PT"
echo "  QAT float:   $RESULTS/qat_fp4/qat_best_float.pt"
echo "  FP4 binary:  $RESULTS/qat_fp4/renderer_fp4.bin"
echo "  Poses:       $RESULTS/pose_tto_final/optimized_poses.pt"
echo "  QAT results: $RESULTS/qat_fp4/qat_results.json"
echo ""
ls -la "$RESULTS/qat_fp4/renderer_fp4.bin" "$RESULTS/pose_tto_final/optimized_poses.pt" 2>/dev/null
echo ""
echo "Download with:"
echo "  scp -P PORT root@HOST:/workspace/pact/experiments/results/qat_fp4/renderer_fp4.bin ."
echo "  scp -P PORT root@HOST:/workspace/pact/experiments/results/pose_tto_final/optimized_poses.pt ."
echo ""
echo "Then destroy the instance immediately to save money."
