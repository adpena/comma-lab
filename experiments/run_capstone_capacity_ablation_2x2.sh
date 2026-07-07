#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# 2x2 capacity-confirm ablation (capstone spec section 6 step 6): the decisive
# capacity-vs-data read BEFORE the multi-day 600-pair bet.
#
#   {base_ch=20, base_ch=24} x {48 pairs, 192 pairs}
#   CE-only (--curriculum none), stored_latent carrier, int8 export,
#   --scorer-backend mlx_gpu (FP32-exact via MLX_METAL_GPU_ARCH=applegpu_g15,
#   set by the campaign CLI when mlx_gpu is selected), equal epochs-per-pair,
#   warmup-EMA eval (the landed EMA-warmup fix -> shadow tracks live on short runs).
#
# The single decisive number:
#   sign( plateau_d_seg(base_ch=20 @ 192) - plateau_d_seg(base_ch=20 @ 48) )
#     negative + large -> DATA-LIMITED (base_ch=20 @ 600 could reach the floor)
#     >= 0             -> CAPACITY-LIMITED (base_ch=24 is the right scale)
# ALSO: does base_ch=24 reach LOWER d_seg than base_ch=20 at the same pairs?
#
# Marker-on-exit per arm + a final DONE.marker so a successor can resume / read
# the verdict without a live session (the durable-daemon discipline).
set -uo pipefail

EPOCHS="${ABLATION_EPOCHS:-120}"
EVAL_EVERY="${ABLATION_EVAL_EVERY:-10}"
OUT_ROOT="${ABLATION_OUT_ROOT:-experiments/results/capstone_capacity_ablation_2x2_20260611}"
TARGETS_CACHE="experiments/results/capstone_gt_targets_cache"
PY="${PY:-.venv/bin/python}"
mkdir -p "$OUT_ROOT"

run_arm () {
  local base_ch="$1" pairs="$2"
  local arm="bc${base_ch}_p${pairs}"
  local out="$OUT_ROOT/$arm"
  mkdir -p "$out"
  echo "[ablation] START arm=$arm base_ch=$base_ch pairs=$pairs epochs=$EPOCHS $(date -u +%FT%TZ)"
  OMP_NUM_THREADS=6 "$PY" experiments/run_capstone_campaign.py \
      --max-pairs "$pairs" \
      --base-channels "$base_ch" \
      --carrier stored_latent \
      --decoder-dtype int8 \
      --curriculum none \
      --epochs "$EPOCHS" \
      --eval-every "$EVAL_EVERY" \
      --seg-weight 100.0 --pose-weight 1.0 \
      --muon-lr 3e-2 --grad-clip 50 --grad-clip-muon 50 \
      --scorer-backend mlx_gpu \
      --authority-recheck-every 50 \
      --device cpu \
      --targets-cache "$TARGETS_CACHE" \
      --out-dir "$out" > "$out/run.log" 2>&1
  local rc=$?
  echo "EXIT=$rc base_ch=$base_ch pairs=$pairs epochs=$EPOCHS at=$(date -u +%FT%TZ)" > "$out/ARM_DONE.marker"
  echo "[ablation] DONE arm=$arm rc=$rc $(date -u +%FT%TZ)"
}

# Order: the two base_ch=20 arms FIRST (they answer the decisive sign question),
# then base_ch=24 (the does-bigger-win comparison). 48-pair arms reuse the existing
# n=48 cache; the first 192-pair arm builds the n=192 GT cache once (shared).
run_arm 20 48
run_arm 20 192
run_arm 24 48
run_arm 24 192

echo "ALL_ARMS_DONE at=$(date -u +%FT%TZ) epochs=$EPOCHS" > "$OUT_ROOT/DONE.marker"
echo "[ablation] ALL ARMS DONE -> $OUT_ROOT/DONE.marker"
