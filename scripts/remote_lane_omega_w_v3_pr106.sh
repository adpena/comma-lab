#!/bin/bash
# NO_NVDEC_NEEDED — pure tensor-side codec + scorer-forward; no DALI/NVDEC video pipeline.
# Lane Ω-W-V3 — Water-fill v2 → PR106 HNeRV decoder repack
#
# Anchor: revival_plan_01_water_filling_codec_v2_pr106_decoder + revival_plan_08_sensitivity_map_pr106_producer
# Council 8/10 GO. Predicted band [0.194, 0.204] [contest-CUDA].
#
# Pipeline (4 stages, all on a single Vast.ai 4090 ~$0.30/hr × 1 hour ≈ $0.30,
# or T4 final auth eval ~$0.22/hr × 30 min ≈ $0.11):
#
#   Stage 1 (CPU): Extract PR106 HNeRV decoder + latents from archive.zip
#   Stage 2 (CUDA): Build per-channel β-Fisher sensitivity_map.pt over 600 contest pairs
#   Stage 3 (CPU): Repack via water_filling_codec_v2 → apogee_v2_archive.zip
#   Stage 4 (CUDA-T4): contest_auth_eval — score must be < 0.20945 (PR106 baseline) to ship
#
# Stub-mode preview (CPU-only sensitivity all-ones) already empirically verified:
#   PR106 archive: 186,239 bytes
#   Apogee-v2 archive: 164,087 bytes (-22,152 / -11.9%)
#   Rate-component score Δ: -0.01475 (within audit prediction band)
#
# Strict-scorer-rule: scorer is loaded ONLY at Stage 2 (compress-time sensitivity build).
# Never loaded at inflate time per CLAUDE.md feedback_strict_scorer_rule.
#
# E2E_SMOKE_OPT_OUT: parser round-trip verified locally on real apogee_v2 0.bin
# (commit c7f237eb) — 28 tensors / 228,958 params decoded byte-faithfully.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_omega_w_v3_pr106"
LOG_DIR="$WORKSPACE/experiments/results/${LANE_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
log() { echo "[lane-owv3-pr106] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Heartbeat (per CLAUDE.md remote-code-parity rule)
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do echo "$(date -u +%FT%TZ) lane-owv3-pr106 alive" >> "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# ── Stage 0: Provenance + CUDA preflight ──────────────────────────────────
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
"$PYBIN" -c "
import json, time, sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: --device cuda required per CLAUDE.md MPS-auth-eval-is-NOISE')
prov = {
    'lane_id': '$LANE_ID',
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
}
with open('$LOG_DIR/provenance.json', 'w') as f:
    json.dump(prov, f, indent=2)
print('[stage-0] provenance written; CUDA available')
"

# ── Stage 1: Extract PR106 decoder (CPU) ──────────────────────────────────
log "=== Stage 1: extract PR106 HNeRV decoder ==="
PR106_ARCHIVE="${PR106_ARCHIVE:-experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip}"
"$PYBIN" experiments/extract_pr106_decoder.py \
    --archive "$PR106_ARCHIVE" \
    --out-dir "$LOG_DIR/" 2>&1 | tee -a "$LOG_DIR/stage1_extract.log"
[ ! -f "$LOG_DIR/state_dict.pt" ] && { log "FATAL: stage 1 did not produce state_dict.pt"; exit 1; }
log "Stage 1 OK: state_dict.pt + latents.pt + metadata.json"

# ── Stage 2: Build β-Fisher sensitivity map (CUDA) ────────────────────────
log "=== Stage 2: build β-Fisher sensitivity map (CUDA-required) ==="
"$PYBIN" experiments/build_sensitivity_map_pr106.py \
    --state-dict "$LOG_DIR/state_dict.pt" \
    --latents "$LOG_DIR/latents.pt" \
    --upstream-dir "${UPSTREAM_DIR:-upstream}" \
    --out "$LOG_DIR/sensitivity_map.pt" \
    --device cuda 2>&1 | tee -a "$LOG_DIR/stage2_sensitivity.log"
[ ! -f "$LOG_DIR/sensitivity_map.pt" ] && { log "FATAL: stage 2 did not produce sensitivity_map.pt"; exit 1; }
log "Stage 2 OK: sensitivity_map.pt with [contest-CUDA] tag"

# ── Stage 3: Repack via water_filling_codec_v2 (CPU) ──────────────────────
log "=== Stage 3: repack PR106 → apogee_v2 via water-filling ==="
TARGET_BYTES="${TARGET_BYTES:-145000}"
"$PYBIN" experiments/repack_pr106_with_water_filling.py \
    --state-dict "$LOG_DIR/state_dict.pt" \
    --sensitivity "$LOG_DIR/sensitivity_map.pt" \
    --pr106-archive "$PR106_ARCHIVE" \
    --target-bytes "$TARGET_BYTES" \
    --out-dir "$LOG_DIR/" 2>&1 | tee -a "$LOG_DIR/stage3_repack.log"
APOGEE_V2_ARCHIVE="$LOG_DIR/apogee_v2_archive.zip"
[ ! -f "$APOGEE_V2_ARCHIVE" ] && { log "FATAL: stage 3 did not produce apogee_v2_archive.zip"; exit 1; }
APOGEE_V2_BYTES=$(stat -c%s "$APOGEE_V2_ARCHIVE" 2>/dev/null || stat -f%z "$APOGEE_V2_ARCHIVE")
log "Stage 3 OK: apogee_v2_archive.zip ($APOGEE_V2_BYTES bytes)"

# ── Stage 4: contest_auth_eval (CUDA-T4 ideal; 4090 acceptable) ───────────
log "=== Stage 4: contest_auth_eval on apogee_v2_archive.zip ==="
"$PYBIN" experiments/contest_auth_eval.py \
    --archive "$APOGEE_V2_ARCHIVE" \
    --inflate-sh submissions/apogee_v2/inflate.sh \
    --upstream-dir "${UPSTREAM_DIR:-upstream}" \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee -a "$LOG_DIR/stage4_auth_eval.log"

# ── Final: harvest + report ────────────────────────────────────────────────
RESULT_JSON="$LOG_DIR/eval_work/contest_auth_eval.adjudicated.json"
[ ! -f "$RESULT_JSON" ] && { log "FATAL: stage 4 did not produce contest_auth_eval result"; exit 1; }
SCORE=$("$PYBIN" -c "import json; d=json.load(open('$RESULT_JSON')); print(d.get('score_recomputed_from_components', d.get('score', 'NaN')))")
log "============================================================"
log "Lane Ω-W-V3 RESULT: contest-CUDA score = $SCORE"
log "Apogee-v2 archive bytes: $APOGEE_V2_BYTES"
log "PR106 baseline score: 0.20945673"
log "Sub-0.20 ship-gate: pass if $SCORE < 0.20945"
log "============================================================"
log "Artifacts: $LOG_DIR"
