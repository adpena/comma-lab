#!/bin/bash
# Overnight training: small DSConv renderer with ALL fixes
# Launch on Vast.ai 4090 after rsync-ing the repo
set -euo pipefail

echo "=== OVERNIGHT TRAINING: QUANTIZR KILLER ==="
echo "Architecture: base_ch=24, mid_ch=32, DSConv, FiLM pose_dim=6"
echo "Training: eval_roundtrip=True, noise_std=0.5, hinge loss, pose_weight=10"
echo "Phases: 1000 (pixel) + 3000 (scorer) + 1000 (hard-pair)"
echo ""

# Ensure dependencies
pip install safetensors einops segmentation-models-pytorch av click 2>/dev/null || true

export PYTHONPATH="src:upstream:$PWD"

# Phase 1+2+3: Train small renderer
python3 -u experiments/train_distill.py \
  --tto-frames experiments/results/tto_v7_hinge_500/tto_frames.pt \
  --checkpoint "" \
  --gt-poses experiments/results/gt_poses.pt \
  --upstream upstream/ \
  --output-dir experiments/results/overnight_small_renderer \
  --base-ch 24 --mid-ch 32 --depth 1 --pose-dim 6 --use-dsconv \
  --device cuda \
  --eval-roundtrip \
  --segnet-loss-mode hinge --hinge-margin 0.5 \
  --pose-weight 10.0 --seg-weight 100.0 --pixel-weight 0.1 \
  --phase1-epochs 1000 --phase1-lr 1e-3 --phase1-batch-size 8 \
  --phase2-epochs 3000 --phase2-lr 3e-4 --phase2-batch-size 4 \
  --phase3-epochs 1000 --phase3-lr 1e-4 --phase3-batch-size 4 \
  --checkpoint-every 200 --eval-every 100 --log-every 25 \
  --seed 42 \
  2>&1 | tee experiments/results/overnight_train.log

echo ""
echo "=== TRAINING COMPLETE ==="

# Export FP4
BEST_CKPT="experiments/results/overnight_small_renderer/distill_best.pt"
if [ -f "$BEST_CKPT" ]; then
  python3 -u -c "
from tac.renderer_export import export_asymmetric_checkpoint_fp4, load_asymmetric_checkpoint
from tac.renderer import AsymmetricPairGenerator
import torch
from pathlib import Path

# Load the trained model
ckpt = torch.load('$BEST_CKPT', map_location='cpu', weights_only=True)
model = AsymmetricPairGenerator(
    num_classes=5, embed_dim=6, base_ch=24, mid_ch=32,
    depth=1, pose_dim=6, use_dsconv=True,
)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)

out = Path('experiments/results/overnight_small_renderer/renderer_fp4.bin')
export_asymmetric_checkpoint_fp4(model, out)
print(f'FP4 export: {out.stat().st_size:,} bytes')
"
  echo "FP4 export complete"
fi

# Run pose TTO
echo ""
echo "=== POSE TTO ==="
python3 -u experiments/optimize_poses.py \
  --checkpoint experiments/results/overnight_small_renderer/distill_best.pt \
  --device cuda \
  --steps 1000 --lr 0.01 --batch-pairs 50 \
  --eval-roundtrip \
  --upstream upstream/ \
  --video upstream/videos/0.mkv \
  --output-dir experiments/results/overnight_pose_tto \
  2>&1 | tee experiments/results/overnight_pose_tto.log

echo ""
echo "=== ALL DONE ==="
echo "Results in experiments/results/overnight_small_renderer/"
echo "Pose TTO in experiments/results/overnight_pose_tto/"
ls -la experiments/results/overnight_small_renderer/*.pt experiments/results/overnight_small_renderer/*.bin 2>/dev/null
ls -la experiments/results/overnight_pose_tto/*.pt 2>/dev/null
