#!/bin/bash
set -euo pipefail

# Find the right Python: conda cloudspace (Lightning), or system python3
if [ -x /home/zeus/miniconda3/envs/cloudspace/bin/python ]; then
    PYTHON=/home/zeus/miniconda3/envs/cloudspace/bin/python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

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
HARD_FRAME="${HARD_FRAME:-0.0}"
EVAL_EVERY="${EVAL_EVERY:-5}"
PRECOMPUTED="${PRECOMPUTED:-}"
WALL_CLOCK_TIMEOUT="${WALL_CLOCK_TIMEOUT:-0}"

EXTRA_ARGS=""
[ -n "${PROFILE:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --profile $PROFILE"
[ -n "$DUAL_SAL" ] && EXTRA_ARGS="$EXTRA_ARGS --use-dual-saliency --alpha-seg $ALPHA_SEG"
[ -n "$USE_STE" ] && EXTRA_ARGS="$EXTRA_ARGS --use-ste --boundary-weight $BOUNDARY_WEIGHT"
[ -n "$RESUME" ] && EXTRA_ARGS="$EXTRA_ARGS --resume-from $RESUME"
[ "$HARD_FRAME" != "0.0" ] && EXTRA_ARGS="$EXTRA_ARGS --hard-frame-ratio $HARD_FRAME"
[ -n "$PRECOMPUTED" ] && EXTRA_ARGS="$EXTRA_ARGS --precomputed $PRECOMPUTED"
[ "$WALL_CLOCK_TIMEOUT" != "0" ] && EXTRA_ARGS="$EXTRA_ARGS --wall-clock-timeout $WALL_CLOCK_TIMEOUT"

echo "=== Training: tag=$TAG variant=$VARIANT h=$HIDDEN epochs=$EPOCHS loss=$LOSS ==="
[ -n "$DUAL_SAL" ] && echo "=== Dual saliency: alpha_seg=$ALPHA_SEG ==="
[ -n "$USE_STE" ] && echo "=== STE: boundary_weight=$BOUNDARY_WEIGHT ==="

$PYTHON train_tac.py \
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
    --eval-every "$EVAL_EVERY" \
    --output-dir ./weights \
    --archive ./archive.zip \
    --gt-video ./upstream/videos/0.mkv \
    --saliency ./saliency.npy \
    --models-dir ./upstream/models \
    --upstream-dir ./upstream \
    $EXTRA_ARGS
