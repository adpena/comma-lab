#!/bin/bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:$(pwd)/upstream:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

TAG="${TAG:-cloud_h64_standard}"
VARIANT="${VARIANT:-dilated}"
HIDDEN="${HIDDEN:-64}"
EPOCHS="${EPOCHS:-2500}"
LOSS="${LOSS:-standard}"
TEMP_START="${TEMP_START:-1.0}"
TEMP_END="${TEMP_END:-1.0}"
ALPHA="${ALPHA:-20}"
DUAL_SAL="${DUAL_SAL:-}"
ALPHA_SEG="${ALPHA_SEG:-200}"
USE_STE="${USE_STE:-}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-1.0}"
RESUME="${RESUME:-}"

EXTRA_ARGS=""
[ -n "$DUAL_SAL" ] && EXTRA_ARGS="$EXTRA_ARGS --use-dual-saliency --alpha-seg $ALPHA_SEG"
[ -n "$USE_STE" ] && EXTRA_ARGS="$EXTRA_ARGS --use-ste --boundary-weight $BOUNDARY_WEIGHT"
[ -n "$RESUME" ] && EXTRA_ARGS="$EXTRA_ARGS --resume-from $RESUME"

echo "=== Training: tag=$TAG variant=$VARIANT h=$HIDDEN epochs=$EPOCHS loss=$LOSS ==="
[ -n "$DUAL_SAL" ] && echo "=== Dual saliency: alpha_seg=$ALPHA_SEG ==="
[ -n "$USE_STE" ] && echo "=== STE: boundary_weight=$BOUNDARY_WEIGHT ==="

python3 train_tac.py \
    --tag "$TAG" \
    --variant "$VARIANT" \
    --hidden "$HIDDEN" \
    --epochs "$EPOCHS" \
    --loss-mode "$LOSS" \
    --temperature-start "$TEMP_START" \
    --temperature-end "$TEMP_END" \
    --alpha "$ALPHA" \
    --sal-lambda 1.0 \
    --subsample 4 \
    --output-dir ./weights \
    --archive ./archive.zip \
    --gt-video ./upstream/videos/0.mkv \
    --saliency ./saliency.npy \
    --models-dir ./upstream/models \
    --upstream-dir ./upstream \
    $EXTRA_ARGS
