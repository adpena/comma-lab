#!/usr/bin/env bash
# B1-R3 DESCENT-PROOF SMOKE (detached). [macOS-MLX research-signal] — NOT a contest score.
#
# Council T4 verdict (feedback_grand_council_symposium_all_results_roadmap_20260609): build the
# atlas-weighted dense carrier + run a CHEAP descent-proof smoke FIRST; gate the long B1-R3 on it.
#
# Strict-scrutiny refined diagnosis (this session):
#   R1 = recon source-weight-amplification ON  + NO grad-clip  -> DIVERGED (stage-1 CE blew up).
#   R2 = grad-clip ON + amplification DROPPED ("kitchen-sink") -> recon under-weighted -> per-pair
#        latents went DEAD -> renderer emitted 2 fixed frames -> d_seg=0.50 FLAT over 3000ep.
#   R3 (this) = amplification ON  +  grad-clip ON  = the untested synthesis. The trainer ALREADY has
#   the source-RGB recon term (adapter.py:5733 recon = mean((rgb-gt)^2)); the bug was its WEIGHT, not
#   its absence. Amplification (line 5400 = part of the PR95-faithful recipe) drives the latents;
#   grad-clip keeps it stable. Cheap (600 epochs, not 3000) so the descent is provable in ~15 min.
#
# Gate: after ep250, the harvester exact-evals the backend-only archive (B2 bridge, 600-pair CPU).
# DESCEND (d_seg << 0.50) -> launch the full staged B1-R3. FLAT -> fork to PR95-faithful / SNeRV carrier.
#
# Run detached: nohup bash scripts/launch_b1_r3_descent_smoke.sh </dev/null >/dev/null 2>&1 & disown
set -uo pipefail
cd /Users/adpena/Projects/pact
RUN_ID="b1_r3_descent_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
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
echo "B1_R3_RUN_ID=${RUN_ID}" >> "${SSD_RUN}/run_id.txt"

.venv/bin/python experiments/train_substrate_hi_nerv_mlx_local.py \
  --full \
  --allow-direct-research-full-launch \
  --research-curriculum-total-epochs 600 \
  --pr95-faithful-curriculum \
  --pr95-muon-policy faithful_stage8_only \
  --pr95-stage-source-weight-amplification \
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
  --cosine-decay --cosine-decay-total-epochs 600 --cosine-decay-min-lr-ratio 1.0e-2 \
  --checkpoint-selection-metric-key total --checkpoint-selection-metric-required \
  --post-export-receiver-cache-quality-gate \
  --upstream-dir upstream \
  --checkpoint-dir "${SSD_RUN}/checkpoints" \
  --checkpoint-interval-epochs 250 \
  --output-dir "${SSD_RUN}"
RC=$?
printf '%s TRAIN_EXIT rc=%s run=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RC}" "${RUN_ID}" >> "${HEARTBEAT}"
exit ${RC}
