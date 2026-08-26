#!/usr/bin/env bash
# B1 CLEAN PR95-FAITHFUL BASELINE (zero novelty). [macOS-MLX research-signal].
#
# THE REFRAME (2026-06-09): R1/R2/R3 were NOT the PR95 recipe — they were OFF-SPEC
# pilots. R1 enabled `--pr95-stage-source-weight-amplification` (a kitchen-sink addition
# NOT in PR95's canonical 8-stage curriculum) and diverged in stage 1 (CE 18 -> ~400);
# R2 dropped amplification but kept other deviations and got dead latents; R3 re-added
# amplification + clip and diverged after ep201. The `_pr95_full_control_contract`
# docstring states the operator binding directive verbatim: "B1 = clean PR95 baseline,
# zero novelty" — and `_append_clean_baseline_relaxable` exists precisely to DROP the
# off-spec additions (amplification, scorer-input guards/tethers/floors, class-escape).
#
# THE NON-ARBITRARY FIX: run the CLEAN PR95-faithful curriculum we have NEVER actually
# run (the leaderboard-proven 0.193 recipe), with ZERO novelty:
#   - --pr95-faithful-curriculum            : the canonical 8-stage CE-first curriculum.
#   - --pr95-muon-policy faithful_stage8_only: Muon FINAL stage only (AdamW formation;
#                                              PR95-faithful; avoids Muon-from-start dead-row risk).
#   - NO --pr95-stage-source-weight-amplification : the off-spec addition that diverged R1/R3.
#   - NO scorer-input guards/tethers/floors / class-escape : the kitchen-sink (relaxed/dropped).
#   - the canonical controls: eval-roundtrip-STE, pr95_yuv6, coder-QAT/C1a, EMA + EMA-archive
#     selection, strict checkpoint selection, hard-byte-ceiling, dual-ascent.
#
# WHY THIS SHOULD WORK where the off-spec pilots failed: the contract-free capacity probe
# (run_hi_nerv_recon_fit_capacity) proved the SAME 229K carrier FITS the full video under
# pure recon (N=1 -> 21.3 dB; N=600 ep0 -> 18.96 dB). The carrier is capable; the off-spec
# additions destabilized it. The clean canonical recipe is the proven, non-arbitrary path.
#
# Staged-scorer (recon-first) + Aurora/comp-Muon-inspired optimizer stacks are
# research-informed FUTURE iterations (Vehicle-1 variants), NOT the B1 clean baseline.
#
# Epoch budget: PR95 canonical is 29,650 (research-relaxed here). 8000 is a pragmatic
# MVP-first floor (>> the 600 the off-spec pilots used); extend toward 29,650 if it descends.
#
# Run detached: nohup bash scripts/launch_b1_clean_pr95_baseline.sh </dev/null >/dev/null 2>&1 & disown
set -euo pipefail
cd /Users/adpena/Projects/pact
EPOCHS="${B1_CLEAN_EPOCHS:-8000}"
NUM_PAIRS="${B1_CLEAN_NUM_PAIRS:-600}"
RUN_ID="b1_clean_pr95_baseline_$(date -u +%Y%m%dT%H%M%SZ)"
SSD_RUN="/Volumes/VertigoDataTier/pact/${RUN_ID}"
mkdir -p "${SSD_RUN}/checkpoints"
HEARTBEAT=".omx/tmp/heartbeat_${RUN_ID}.log"
mkdir -p .omx/tmp
( while true; do
    printf '%s pid=%s run=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "${RUN_ID}" >> "${HEARTBEAT}"
    sleep 60
  done ) &
HB_PID=$!
trap "kill ${HB_PID} 2>/dev/null || true" EXIT
echo "B1_CLEAN_RUN_ID=${RUN_ID}" >> "${SSD_RUN}/run_id.txt"

RC=0
.venv/bin/python experiments/train_substrate_hi_nerv_mlx_local.py \
  --full \
  --allow-direct-research-full-launch \
  --research-curriculum-total-epochs "${EPOCHS}" \
  --pr95-faithful-curriculum \
  --pr95-muon-policy faithful_stage8_only \
  --decoder-channels 36,30,23,17,14,11,8 \
  --latent-dim-coarse 16 --latent-dim-mid 20 --latent-dim-fine 24 --embed-dim 64 \
  --num-pairs "${NUM_PAIRS}" --batch-pairs 16 \
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
  --cosine-decay --cosine-decay-total-epochs "${EPOCHS}" --cosine-decay-min-lr-ratio 1.0e-2 \
  --checkpoint-selection-metric-key total --checkpoint-selection-metric-required \
  --post-export-receiver-cache-quality-gate \
  --upstream-dir upstream \
  --checkpoint-dir "${SSD_RUN}/checkpoints" \
  --checkpoint-interval-epochs 1000 \
  --output-dir "${SSD_RUN}" || RC=$?
printf '%s TRAIN_EXIT rc=%s run=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RC}" "${RUN_ID}" >> "${HEARTBEAT}"
exit ${RC}
