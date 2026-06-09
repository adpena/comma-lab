#!/bin/bash
# B1 229K CLEAN + STABILIZED PR95-faithful baseline (detached).
# [macOS-MLX research-signal] — NOT a contest score. Score = B2 bridge exact eval.
#
# Supersedes b1_229k_pilot_20260609T055851Z, which DIVERGED within stage-1 (CE loss
# 18 -> sustained ~350-400) because it was OFF-SPEC: ~10 equal-weight loss terms +
# --pr95-stage-source-weight-amplification + NO grad-clip — a kitchen-sink that violates
# the operator's binding "B1 = clean PR95 baseline, zero novelty" directive.
#
# This run is the ACTUALLY-clean PR95 baseline:
#   KEEP  : the 8-stage faithful curriculum + the CORE score-aware distillation
#           (--distillation-weight 1.0 = SegNet-KL gradient-through + boundary-argmax-hinge;
#            --pose-distillation-weight 1.0 = PoseNet-MSE) + QAT + EMA(0.997) + eval-roundtrip.
#   ADD   : STABILIZER — grad-clip (Wave-N+11 canonical max_norm=1.0) + warmup + weight-decay
#           + cosine LR decay over the run (the diverging run had none -> instability).
#   DROP  : the kitchen-sink (all 0.0-default flags the diverging run forced to 1.0):
#            --pr95-stage-source-weight-amplification
#            --segnet-direct-live-distillation-weight / --segnet-direct-live-class-histogram-weight
#            --pose-direct-live-distillation-weight
#            --scorer-space-step-guard / --scorer-input-distribution-guard-weight
#            --scorer-input-contrast-floor-weight / --scorer-input-shape-tether-weight
#            --posenet-yuv6-geometry-tether-weight / --posenet-temporal-signal-floor-weight
#
# Run detached:  nohup bash scripts/launch_b1_clean_stabilized_pr95.sh </dev/null >/dev/null 2>&1 & disown
set -uo pipefail
cd /Users/adpena/Projects/pact
RUN_ID="b1_229k_clean_$(date -u +%Y%m%dT%H%M%SZ)"
SSD_RUN="/Volumes/VertigoDataTier/pact/${RUN_ID}"
mkdir -p "${SSD_RUN}/checkpoints"
HEARTBEAT=".omx/tmp/heartbeat_b1_${RUN_ID}.log"
mkdir -p .omx/tmp
( while true; do
    printf '%s pid=%s run=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "${RUN_ID}" >> "${HEARTBEAT}"
    sleep 60
  done ) &
HB_PID=$!
trap "kill ${HB_PID} 2>/dev/null" EXIT
echo "CLEAN_B1_RUN_ID=${RUN_ID}" >> "${SSD_RUN}/run_id.txt"
.venv/bin/python experiments/train_substrate_hi_nerv_mlx_local.py \
  --full \
  --allow-direct-research-full-launch \
  --research-curriculum-total-epochs 3000 \
  --pr95-faithful-curriculum \
  --pr95-muon-policy faithful_stage8_only \
  --decoder-channels 36,30,23,17,14,11,8 \
  --latent-dim-coarse 16 --latent-dim-mid 20 --latent-dim-fine 24 --embed-dim 64 \
  --num-pairs 600 --batch-pairs 16 \
  --ema-decay 0.997 --ema-archive-selection \
  --diagnostics-every-n-steps 50 \
  --distillation-weight 1.0 \
  --segnet-distillation-objective boundary_argmax_hinge \
  --pose-distillation-weight 1.0 \
  --eval-roundtrip-ste \
  --coder-qat --coder-qat-c1a-entropy-weight 1.0e-4 \
  --hard-byte-ceiling 300000 \
  --grad-clip-max-norm 1.0 \
  --warmup-epochs 10 \
  --weight-decay 1.0e-4 \
  --cosine-decay --cosine-decay-total-epochs 3000 --cosine-decay-min-lr-ratio 1.0e-2 \
  --checkpoint-selection-metric-key total --checkpoint-selection-metric-required \
  --post-export-receiver-cache-quality-gate \
  --upstream-dir upstream \
  --checkpoint-dir "${SSD_RUN}/checkpoints" \
  --checkpoint-interval-epochs 250 \
  --output-dir "${SSD_RUN}"
RC=$?
printf '%s TRAIN_EXIT rc=%s run=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RC}" "${RUN_ID}" >> "${HEARTBEAT}"
exit ${RC}
