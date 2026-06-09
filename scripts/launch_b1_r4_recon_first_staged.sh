#!/bin/bash
# B1-R4 RECON-FIRST STAGED CURRICULUM (the DERIVED production fix). [macOS-MLX research-signal].
#
# THE DISCOVERY (2026-06-09): the recon-first annealed curriculum ALREADY EXISTS as
# `--staged-scorer-curriculum`, but R1/R2/R3 ALL used `--pr95-faithful-curriculum`
# (CE-Seg from stage 1, NO pure-recon formation). The staged curriculum is exactly the
# fix the capacity probe pointed to:
#   stage 0  hi_nerv_receiver_fit_recon_scaffold   (0 -> recon_frac):  PURE RECON (distill=0,pose=0)
#   stage 1  hi_nerv_segnet_last_frame_admission   (-> segnet_end):    recon=1.0 HELD + SegNet admitted
#   stage 2  hi_nerv_joint_scorer_waterfill        (-> end):           recon=FINAL_RECON_WEIGHT + pose
#
# WHY THIS IS THE FIX (derived, not arbitrary):
#  - Arm A capacity probe: pure recon fits the full video (N=1 -> 21.3 dB; N=600 ep0 -> 18.96 dB).
#    So the carrier IS capable; the production 4.5 dB failure was the OBJECTIVE/CURRICULUM
#    applying score-aware from epoch 0 (R3) with no held recon anchor (R2 dropped it -> mean-field).
#  - The staged curriculum establishes the RGB fit FIRST (stage 0), then anneals seg -> pose
#    while HOLDING the recon anchor, so the renderer cannot abandon the frame manifold.
#
# DERIVED KNOBS (each justified, none arbitrary; provenance per the adversarial review):
#  - --staged-scorer-curriculum                : the recon-first mode (D-MEAS: Arm A says recon fits).
#  - --staged-scorer-recon-fraction 0.5        : 50% pure-recon formation (D-CTRL: Arm A fit fast;
#                                                600 shared pairs need more than N=1's 60 ep, so 50%).
#  - --staged-scorer-final-recon-weight 0.5    : HELD-STRONG anchor (D-MEAS: R2's recon-drop to ~0
#                                                caused mean-field; the 0.25 default is too weak.
#                                                0.5 keeps the renderer on the manifold while
#                                                seg/pose refine). THE critical control.
#  - --pr95-muon-policy faithful_stage8_only   : AdamW formation, Muon FINAL stage only (D-CTRL:
#                                                Arm A fit under AdamW; PR95 uses Muon last-stage-only;
#                                                Muon-from-start risks rectangular-projection dead rows
#                                                = the Aurora hypothesis. Keep formation on AdamW).
#  - --grad-clip-max-norm 1.0                  : stability (R1 diverged with no clip at the score-aware
#                                                stages; Arm A fit no-clip on PURE recon, but the
#                                                score-aware stages need the clip).
#
# Gate: this is the production fix, GATED on the Arm A full trajectory confirming the carrier fits
# (>> 4.5 dB). The smoke below validates the staged launch + that the recon-scaffold stage fits.
#
# Run detached: nohup bash scripts/launch_b1_r4_recon_first_staged.sh </dev/null >/dev/null 2>&1 & disown
set -uo pipefail
cd /Users/adpena/Projects/pact
EPOCHS="${B1_R4_EPOCHS:-3000}"
NUM_PAIRS="${B1_R4_NUM_PAIRS:-600}"
RUN_ID="b1_r4_recon_first_staged_$(date -u +%Y%m%dT%H%M%SZ)"
SSD_RUN="/Volumes/VertigoDataTier/pact/${RUN_ID}"
mkdir -p "${SSD_RUN}/checkpoints"
HEARTBEAT=".omx/tmp/heartbeat_${RUN_ID}.log"
mkdir -p .omx/tmp
( while true; do
    printf '%s pid=%s run=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "${RUN_ID}" >> "${HEARTBEAT}"
    sleep 60
  done ) &
HB_PID=$!
trap "kill ${HB_PID} 2>/dev/null" EXIT
echo "B1_R4_RUN_ID=${RUN_ID}" >> "${SSD_RUN}/run_id.txt"

.venv/bin/python experiments/train_substrate_hi_nerv_mlx_local.py \
  --full \
  --allow-direct-research-full-launch \
  --research-curriculum-total-epochs "${EPOCHS}" \
  --staged-scorer-curriculum \
  --staged-scorer-recon-fraction 0.5 \
  --staged-scorer-segnet-fraction 0.2 \
  --staged-scorer-final-recon-weight 0.5 \
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
  --checkpoint-interval-epochs 500 \
  --output-dir "${SSD_RUN}"
RC=$?
printf '%s TRAIN_EXIT rc=%s run=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RC}" "${RUN_ID}" >> "${HEARTBEAT}"
exit ${RC}
