#!/bin/bash
# Remote lane script: substrate nscs06_v8_chroma_lut L0 SCAFFOLD dispatch.
#
# Trainer: experiments/train_substrate_nscs06_v8_chroma_lut.py
# Lane: lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521
#
# Per CASCADE COMPRESSION symposium commit d125af6c3 PRIORITY 3 + HONEST
# CASCADE-MORTALITY ASSESSMENT commit d884dd6aa Rank 2 + canonical
# equation #26 IN-DOMAIN context nscs06_v8_chroma_lut.
#
# L0 SCAFFOLD: --smoke ONLY until per-substrate symposium per Catalog #325
# PROCEED verdict (window 2026-05-21 -> 2026-06-04). Full mode raises
# NotImplementedError in the trainer per Catalog #240 recipe-vs-trainer-state
# consistency.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# MUST be set before sourcing the bootstrap to prevent the sourced main flow
# (which expects a pre-built archive.zip) from running.
#
# Score-tagging: NONE (L0 SCAFFOLD; --smoke produces non-promotable archive
# tagged [prediction] per Catalog #287 + #323).
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# === Catalog #244 / D1 incident anchor (commit 611495f26): canonical Modal/CUDA env hygiene ===
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing directive
# 2026-05-15 ("all possible should be pulled into the decorator or similar reusable
# and shareable tools and helpers and such"). Sister substrates D1/D4/Z3/Z4/Z5/v7
# carry this block; v8 inherits per canonical pattern.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521"
TAG="${TAG:-substrate_nscs06_v8_chroma_lut}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_nscs06_v8_chroma_lut_results}"

# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${NSCS06_V8_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$NSCS06_V8_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
NSCS06_V8_VIDEO_PATH="${NSCS06_V8_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
NSCS06_V8_OUTPUT_DIR="${NSCS06_V8_OUTPUT_DIR:-$OUTPUT_DIR}"
NSCS06_V8_EPOCHS="${NSCS06_V8_EPOCHS:-1}"
NSCS06_V8_UPSTREAM_DIR="${NSCS06_V8_UPSTREAM_DIR:-$WORKSPACE/upstream}"
# OVERNIGHT-BBB FIX 2026-05-21 (cron 5e07de6e verdict MEDIUM diagnosis at
# fc-01KS5XN8WF9JF15KVX3GPCFAE7 rc=1 elapsed=6.9s):
# Default flipped cpu -> cuda per OVERNIGHT-V Phase 2 BUILD atomic recipe flip.
# Pre-fix: stale L0 SCAFFOLD --smoke-only carryover defaulted to cpu, but
# OVERNIGHT-V flipped NSCS06_V8_TRAINER_MODE smoke->full without atomic device
# flip; trainer canonical _device_or_die(device=cpu, smoke=False) REFUSED per
# CLAUDE.md "MPS auth eval is NOISE" + "EMA — non-negotiable" + full-training-
# needs-CUDA convention; trainer.log: "--device cpu is permitted only with
# --smoke ... Use --device cuda for promotion-grade training."
# Sister-comparator: ATW_v2 / Balle / D1 / DPP / Z3 / Z3_G1 / Z4 / Z5 all
# default _DEVICE to cuda; NSCS06 v8 was the only outlier. Per Catalog #240
# recipe-vs-trainer-state consistency + Catalog #326 driver-mode-mismatch trap
# sister class extended to device-mode consistency.
NSCS06_V8_DEVICE="${NSCS06_V8_DEVICE:-cuda}"  # full-mode atomic flip per OVERNIGHT-BBB; smoke-mode operators may override via env

# Catalog #326 driver-mode env var: explicit precedence over SMOKE_ONLY.
# Recipe-side `env_overrides` block declares NSCS06_V8_TRAINER_MODE explicitly;
# bare `SMOKE_ONLY` is also honored for backwards-compat. Per CLAUDE.md
# "Forbidden ... driver hardcoding smoke=1 regardless of dispatch env vars"
# non-negotiable: BOTH keys are consulted; recipe override beats default.
NSCS06_V8_TRAINER_MODE="${NSCS06_V8_TRAINER_MODE:-${SMOKE_ONLY:+smoke}}"
NSCS06_V8_TRAINER_MODE="${NSCS06_V8_TRAINER_MODE:-smoke}"

DISPATCH_INSTANCE_JOB_ID="${NSCS06_V8_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${NSCS06_V8_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"

log() { echo "[lane-nscs06-v8-chroma-lut] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv 2>/dev/null || true

# Stage 0: validate trainer mode. Per OVERNIGHT-V Phase 2 BUILD landing
# (2026-05-21): the trainer's `_full_main` is now IMPLEMENTED (~440 LOC at
# experiments/train_substrate_nscs06_v8_chroma_lut.py:565-1003) and the recipe
# was atomically flipped to `dispatch_enabled: true / NSCS06_V8_TRAINER_MODE:
# "full"`. The prior L0 SCAFFOLD smoke-only guard was a stale-state-divergence
# bug class (Catalog #240 sister + Catalog #326 driver-mode anti-pattern); fixed
# by OVERNIGHT-RR (2026-05-21) per cron 2b6527f6 verdict LOW + rc=22 recurrence
# diagnosis at QQ call_id fc-01KS5QRXWNVYC54E2Y9Z8KZ4W2 stdout
# `FATAL: NSCS06_V8_TRAINER_MODE=full; only 'smoke' is supported in L0 SCAFFOLD`.
# The accepted modes are now {smoke, full}; anything else is FATAL.
if [ "$NSCS06_V8_TRAINER_MODE" != "smoke" ] && [ "$NSCS06_V8_TRAINER_MODE" != "full" ]; then
    log "FATAL: NSCS06_V8_TRAINER_MODE=$NSCS06_V8_TRAINER_MODE; only 'smoke' or 'full' accepted"
    log "FATAL: per OVERNIGHT-V Phase 2 BUILD landing + OVERNIGHT-RR driver atomic-flip 2026-05-21"
    exit 22
fi
log "NSCS06_V8_TRAINER_MODE=$NSCS06_V8_TRAINER_MODE accepted"

# Stage 0b: dispatch claim verification (optional in L0 SCAFFOLD; skipped
# when the recipe is research_only AND no DISPATCH_INSTANCE_JOB_ID is set,
# i.e. local engineering smoke).
if [ -n "$DISPATCH_INSTANCE_JOB_ID" ]; then
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py"
        exit 21
    fi
    log "claim helper available; lane=$LANE_ID instance=$DISPATCH_INSTANCE_JOB_ID"
fi

# Stage 1: resolve PYBIN.
if [ -z "$PYBIN" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    PYBIN="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$PYBIN" ]; then
    PYBIN="python3"
fi
log "PYBIN=$PYBIN"

# Stage 2: source bootstrap dependencies if not already setup.
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "sourcing canonical bootstrap (Catalog #163 sentinel honored)"
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
    if command -v bootstrap_runtime_deps >/dev/null 2>&1; then
        bootstrap_runtime_deps || log "bootstrap_runtime_deps returned non-zero; continuing for L0 SCAFFOLD"
    fi
fi

# Stage 3: run trainer with mode-conditional --smoke flag. Per OVERNIGHT-V
# Phase 2 BUILD landing + OVERNIGHT-RR driver atomic-flip 2026-05-21:
# NSCS06_V8_TRAINER_MODE=full -> omit --smoke -> trainer enters _full_main;
# NSCS06_V8_TRAINER_MODE=smoke -> pass --smoke -> trainer enters _smoke_main.
# Per CLAUDE.md "Forbidden substrate driver hardcoding smoke=1 / --smoke
# regardless of dispatch env vars (the driver-mode-mismatch trap)" + Catalog
# #326 sister discipline + Z6-v2 Wave 2 empirical anchor.
SMOKE_FLAG=""
if [ "$NSCS06_V8_TRAINER_MODE" = "smoke" ]; then
    SMOKE_FLAG="--smoke"
fi
log "running v8 chroma-LUT trainer mode=$NSCS06_V8_TRAINER_MODE smoke_flag='$SMOKE_FLAG'"
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

set +e
$PYBIN "$WORKSPACE/experiments/train_substrate_nscs06_v8_chroma_lut.py" \
    --video-path "$NSCS06_V8_VIDEO_PATH" \
    --output-dir "$NSCS06_V8_OUTPUT_DIR" \
    --upstream-dir "$NSCS06_V8_UPSTREAM_DIR" \
    --device "$NSCS06_V8_DEVICE" \
    --epochs "$NSCS06_V8_EPOCHS" \
    $SMOKE_FLAG \
    2>&1 | tee -a "$LOG_DIR/trainer.log"
TRAINER_RC=${PIPESTATUS[0]}
set -e

log "trainer exit code: $TRAINER_RC"

# Stage 4: write provenance + completion marker.
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer_path": "experiments/train_substrate_nscs06_v8_chroma_lut.py",
  "recipe": "substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml",
  "mode": "$NSCS06_V8_TRAINER_MODE",
  "device": "$NSCS06_V8_DEVICE",
  "output_dir": "$NSCS06_V8_OUTPUT_DIR",
  "trainer_rc": $TRAINER_RC,
  "dispatch_platform": "$DISPATCH_PLATFORM",
  "canonical_equation_in_domain_context": "nscs06_v8_chroma_lut",
  "evidence_grade": "predicted",
  "axis_tag": "[prediction]",
  "score_claim_valid": false,
  "promotable": false,
  "research_only": true,
  "per_substrate_symposium_pending": true,
  "completed_at_utc": "$(date -u +%FT%TZ)"
}
EOF

if [ "$TRAINER_RC" -eq 0 ]; then
    log "LANE_NSCS06_V8_CHROMA_LUT_DONE [prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]"
else
    log "LANE_NSCS06_V8_CHROMA_LUT_FAILED rc=$TRAINER_RC"
fi

exit $TRAINER_RC
