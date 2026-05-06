#!/bin/bash
# Wave-Ω-2 NeRV mask codec FULL CUDA training dispatcher (PRE-STAGED, DO NOT
# EXECUTE until Wave-1 anchor lands). Wraps scripts/remote_lane_nerv.sh with
# full-CUDA-training params (60,000 SGD steps target per Phase F empirical
# plan) and ensures the Council-mandated infrastructure rules are enforced:
#
#   * Stage 0 self-bootstrap (uv + ffmpeg + cu124 driver-pin) per
#     scripts/remote_archive_only_eval.sh canonical pattern (CLAUDE.md
#     forbidden re-implementing remote bootstrap inline).
#   * Stage 5 contest-CUDA auth eval after training (CLAUDE.md "Auth eval
#     EVERYWHERE" non-negotiable). RUN_AUTH_EVAL=1 forced.
#   * Heartbeat loop (preflight Check 41) inherited from
#     remote_lane_nerv.sh (which already starts heartbeat in its own
#     script via probe_nvdec.sh + provenance writer).
#   * Heartbeat backstop here in case the wrapped script's HB never starts
#     (covers Stage 0 bootstrap window before remote_lane_nerv.sh begins).
#
# Predicted [contest-CUDA] band (header for operator):
#   STANDALONE                      [low=0.95, mid=1.05, high=1.30]
#   STACKED-WITH-WAVE-1 (orth ext)  [low=0.27, mid=0.30, high=0.34]
#
# Stacking is conditional on the Wave-1 anchor (Q-FAITHFUL or owv3_0120)
# landing first AND its renderer.bin + poses being slot-aware-compatible
# with the masks.nrv replacement that Stage 2 of remote_lane_nerv.sh
# performs (BASE_ARCHIVE → replace masks.* with masks.nrv).
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS — every flag passed to wrapped
# scripts was verified by argparse-grep:
#   train_nerv_mask.py: --profile --device --upstream --gt-masks-source
#                       --decoded-baseline-path --decoded-baseline-member
#                       --alpha-primitive-contract --output-dir --num-frames
#                       --mask-height --mask-width --steps --eval-every
#                       --weight-dtype --allow-forensic-segnet-target
#   contest_auth_eval.py: --archive --inflate-sh --upstream-dir --device
#                         --keep-work-dir --work-dir
#   (auth eval invoked transitively by remote_lane_nerv.sh Stage 5)

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
cd "$WORKSPACE"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"

LABEL="${LABEL:-wave_omega_2_nerv_full_cuda}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/${LABEL}_results}"
mkdir -p "$LOG_DIR"
HEARTBEAT_BACKSTOP="$LOG_DIR/heartbeat_wave_omega_2.log"
RUN_LOG="$LOG_DIR/run.log"

log() { echo "[wave-omega-2] $(date -u +%FT%TZ) $*" | tee -a "$RUN_LOG"; }

# ─── Stage 0: self-bootstrap (delegates to canonical wrapper) ────────────
# CLAUDE.md non-negotiable: NEVER write `curl ... | sh` or `apt-get ffmpeg`
# inline. We source the canonical bootstrap from remote_archive_only_eval.sh
# by extracting just its bootstrap_runtime_deps + require_uv_and_ffmpeg_contract
# functions if it's available on the remote, else we fail loud.
CANON_BOOTSTRAP="$WORKSPACE/scripts/remote_archive_only_eval.sh"
if [ ! -f "$CANON_BOOTSTRAP" ]; then
    log "FATAL: canonical bootstrap script missing: $CANON_BOOTSTRAP"
    log "  cannot self-bootstrap uv/ffmpeg/cu124 — abort"
    exit 7
fi
# Source bootstrap functions only — guard against running the body of the
# canonical script (which would invoke an archive eval).
log "=== Stage 0: source canonical bootstrap functions from remote_archive_only_eval.sh ==="
# Extract just the function definitions via awk; eval them in our shell.
# This isolates the bootstrap_runtime_deps + require_uv_and_ffmpeg_contract
# definitions from the canonical script's main body.
BOOTSTRAP_TMP="$LOG_DIR/_canonical_bootstrap_funcs.sh"
awk '
    /^bootstrap_runtime_deps\(\)/    { in_fn=1 }
    /^require_uv_and_ffmpeg_contract\(\)/  { in_fn=1 }
    in_fn { print }
    in_fn && /^\}$/  { in_fn=0 }
' "$CANON_BOOTSTRAP" > "$BOOTSTRAP_TMP"
# shellcheck disable=SC1090
source "$BOOTSTRAP_TMP"
require_uv_and_ffmpeg_contract

# Heartbeat backstop (preflight Check 41) — covers the Stage 0+1 window
# before remote_lane_nerv.sh starts its own heartbeat.
( while true; do echo "$(date -u +%FT%TZ) wave_omega_2_backstop pid=$$" >> "$HEARTBEAT_BACKSTOP"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# ─── Stage 1: pre-flight gates ───────────────────────────────────────────
log "=== Stage 1: pre-flight gates ==="
# Wave-1 anchor presence (the renderer.bin + poses we'll mask-stack onto).
WAVE1_ANCHOR_ARCHIVE="${WAVE1_ANCHOR_ARCHIVE:-$WORKSPACE/experiments/results/lane_q_faithful_retrain_20260501/archive_lane_q_faithful.zip}"
WAVE1_FALLBACK_ARCHIVE="${WAVE1_FALLBACK_ARCHIVE:-$WORKSPACE/experiments/results/lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501/owv3_0120_stack_archive.zip}"
if [ -f "$WAVE1_ANCHOR_ARCHIVE" ]; then
    BASE_ARCHIVE="$WAVE1_ANCHOR_ARCHIVE"
    log "  Wave-1 anchor: $BASE_ARCHIVE (Q-FAITHFUL trained)"
elif [ -f "$WAVE1_FALLBACK_ARCHIVE" ]; then
    BASE_ARCHIVE="$WAVE1_FALLBACK_ARCHIVE"
    log "  Wave-1 anchor MISSING — falling back to deploy champion: $BASE_ARCHIVE"
else
    log "FATAL: neither Wave-1 Q-FAITHFUL nor 0.9974 owv3_0120 fallback archive present"
    exit 8
fi
export BASE_ARCHIVE

# Council-mandated decoded-baseline + alpha-primitive contract.
DECODED_BASELINE_PATH="${DECODED_BASELINE_PATH:-$BASE_ARCHIVE}"
DECODED_BASELINE_MEMBER="${DECODED_BASELINE_MEMBER:-masks.mkv}"
GT_MASKS_SOURCE="${GT_MASKS_SOURCE:-decoded-baseline}"

# Phase F mandate: 60,000 SGD steps to drive argmax disagreement < 0.1%.
NERV_STEPS="${NERV_STEPS:-60000}"
NERV_EVAL_EVERY="${NERV_EVAL_EVERY:-1000}"
NERV_PROFILE="${NERV_PROFILE:-nerv_mask_lane_g_v3}"
NERV_WEIGHT_DTYPE="${NERV_WEIGHT_DTYPE:-fp16}"

# Auth eval requirements (CLAUDE.md non-negotiable, see remote_lane_nerv.sh
# Stage 1.5 RUN_AUTH_EVAL=1 gate).
RUN_AUTH_EVAL="${RUN_AUTH_EVAL:-1}"
POSE_REGEN_PROVENANCE="${POSE_REGEN_PROVENANCE:-$WORKSPACE/.omx/state/wave_omega_2_pose_regen_provenance.json}"
ALPHA_GEO_PROVENANCE="${ALPHA_GEO_PROVENANCE:-$WORKSPACE/.omx/state/wave_omega_2_alpha_geo_provenance.json}"
ALPHA_PRIMITIVE_CONTRACT="${ALPHA_PRIMITIVE_CONTRACT:-$WORKSPACE/experiments/results/alpha_geo_contract.json}"
L2_CLEARANCE_PATH="${L2_CLEARANCE_PATH:-$WORKSPACE/.omx/state/lane12_nerv_l2_clearance.json}"
BASELINE_SCORE="${BASELINE_SCORE:-0.9974}"  # owv3_0120 deploy champion

if [ "$RUN_AUTH_EVAL" = "1" ]; then
    for required in "$POSE_REGEN_PROVENANCE" "$ALPHA_GEO_PROVENANCE" "$ALPHA_PRIMITIVE_CONTRACT"; do
        if [ ! -f "$required" ]; then
            log "FATAL: RUN_AUTH_EVAL=1 requires $required to exist (per remote_lane_nerv.sh Stage 1.5 gates)"
            log "  to bypass for SMOKE only: RUN_AUTH_EVAL=0 (no contest-CUDA score will be produced)"
            exit 9
        fi
    done
fi

# Provenance for THIS wrapper (separate from inner script's provenance).
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
cat > "$LOG_DIR/wave_omega_2_provenance.json" <<JSON
{
  "schema_version": 1,
  "started_at_utc": "$(date -u +%FT%TZ)",
  "wrapper_script": "scripts/wave_omega_2_nerv_full_cuda.sh",
  "wrapped_script": "scripts/remote_lane_nerv.sh",
  "label": "$LABEL",
  "git_hash": "$GIT_HASH",
  "gpu_name": "$GPU_NAME",
  "driver_version": "$DRIVER_VER",
  "base_archive": "$BASE_ARCHIVE",
  "wave1_anchor_used": "$([ "$BASE_ARCHIVE" = "$WAVE1_ANCHOR_ARCHIVE" ] && echo "q_faithful" || echo "owv3_0120_fallback")",
  "nerv_steps": $NERV_STEPS,
  "nerv_eval_every": $NERV_EVAL_EVERY,
  "nerv_profile": "$NERV_PROFILE",
  "nerv_weight_dtype": "$NERV_WEIGHT_DTYPE",
  "run_auth_eval": $RUN_AUTH_EVAL,
  "baseline_score": "$BASELINE_SCORE",
  "predicted_band_standalone": [0.95, 1.30],
  "predicted_band_stacked_with_wave1": [0.27, 0.34]
}
JSON

# ─── Stage 2: dispatch wrapped remote_lane_nerv.sh ───────────────────────
log "=== Stage 2: dispatch remote_lane_nerv.sh with full-CUDA params ==="
log "  NERV_STEPS=$NERV_STEPS NERV_EVAL_EVERY=$NERV_EVAL_EVERY NERV_PROFILE=$NERV_PROFILE"
log "  RUN_AUTH_EVAL=$RUN_AUTH_EVAL BASE_ARCHIVE=$BASE_ARCHIVE"

# Pass step/eval-every/weight-dtype overrides through environment to
# remote_lane_nerv.sh. The inner script forwards these into
# experiments/train_nerv_mask.py, so the Phase F full-CUDA target does not
# depend on the shorter profile default.
if ! grep -q 'NERV_TRAINING_ARGS=.*' "$WORKSPACE/scripts/remote_lane_nerv.sh"; then
    log "PRE-STAGE BLOCKER: remote_lane_nerv.sh is missing NERV_TRAINING_ARGS forwarding."
    log "  Refusing to launch because NERV_STEPS/NERV_EVAL_EVERY/NERV_WEIGHT_DTYPE would be ignored."
    exit 10
fi

export BASE_ARCHIVE LOG_DIR DECODED_BASELINE_PATH DECODED_BASELINE_MEMBER GT_MASKS_SOURCE
export NERV_PROFILE PROFILE="$NERV_PROFILE"
export NERV_STEPS NERV_EVAL_EVERY NERV_WEIGHT_DTYPE
export RUN_AUTH_EVAL POSE_REGEN_PROVENANCE ALPHA_GEO_PROVENANCE ALPHA_PRIMITIVE_CONTRACT L2_CLEARANCE_PATH BASELINE_SCORE

# Override the inner script's LOG_DIR so artifacts land in our wave-Ω-2
# tagged directory rather than the default lane_12_nerv_results.
INNER_LOG_DIR="$LOG_DIR/lane_12_nerv_inner"
mkdir -p "$INNER_LOG_DIR"
LOG_DIR="$INNER_LOG_DIR" bash "$WORKSPACE/scripts/remote_lane_nerv.sh"
INNER_RC=$?
if [ "$INNER_RC" -ne 0 ]; then
    log "FATAL: wrapped remote_lane_nerv.sh exited rc=$INNER_RC"
    exit "$INNER_RC"
fi

# ─── Stage 3: surface result + summary ───────────────────────────────────
RESULT_JSON="$INNER_LOG_DIR/contest_auth_eval.json"
if [ -f "$RESULT_JSON" ]; then
    SCORE=$(.venv/bin/python -c "import json; print(json.load(open('$RESULT_JSON')).get('total_score','unknown'))" 2>/dev/null || echo "parse_failed")
    log "=== WAVE_OMEGA_2_DONE [contest-CUDA] ==="
    log "  score: $SCORE"
    log "  baseline (owv3_0120): $BASELINE_SCORE"
    log "  predicted band standalone: [0.95, 1.30]"
    log "  predicted band stacked: [0.27, 0.34]"
    log "  artifact: $RESULT_JSON"
else
    log "FATAL: contest_auth_eval JSON missing at $RESULT_JSON — wrapped script did not complete Stage 5"
    exit 11
fi
