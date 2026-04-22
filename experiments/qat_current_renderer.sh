#!/bin/bash
# QAT the CURRENT renderer (288K) on a second 4090.
# Gives us a submission candidate while the small renderer trains.
set -euo pipefail
export PYTHONPATH="src:upstream:$PWD"

echo "=== QAT: CURRENT RENDERER (288K) ==="
echo "Started: $(date)"

# Install deps
pip install safetensors einops segmentation-models-pytorch av click 2>/dev/null

# Run QAT — skip INT8 warmup (go direct FP4, the parametrize backward
# crash only affects MPS, CUDA handles it fine)
python3 -u experiments/qat_finetune.py \
    --checkpoint submissions/robust_current/renderer.bin \
    --upstream upstream/ \
    --output-dir experiments/results/qat_current_renderer \
    --device cuda \
    --skip-int8-warmup \
    --fp4-epochs 200 \
    --lr 3e-5 \
    --batch-size 4 \
    2>&1 | tee experiments/results/qat_current.log

# After QAT: run pose TTO on the QAT model
echo ""
echo "=== POSE TTO ON QAT MODEL ==="
QAT_CKPT="experiments/results/qat_current_renderer/qat_best_float.pt"
if [ ! -f "$QAT_CKPT" ]; then
    QAT_CKPT="experiments/results/qat_current_renderer/renderer_fp4.bin"
fi

python3 -u experiments/optimize_poses.py \
    --checkpoint "$QAT_CKPT" \
    --device cuda \
    --steps 1000 --lr 0.01 --batch-pairs 50 \
    --eval-roundtrip \
    --upstream upstream/ \
    --video upstream/videos/0.mkv \
    --output-dir experiments/results/pose_tto_qat_current \
    2>&1 | tee experiments/results/pose_tto_qat_current.log

echo ""
echo "=== DONE: $(date) ==="
echo "Artifacts:"
ls -la experiments/results/qat_current_renderer/renderer_fp4.bin 2>/dev/null
ls -la experiments/results/pose_tto_qat_current/optimized_poses.pt 2>/dev/null
