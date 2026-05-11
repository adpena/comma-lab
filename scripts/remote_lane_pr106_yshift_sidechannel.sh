#!/bin/bash
# CUDA scorer/eval lane. Non-zero modes require CUDA and DALI/NVDEC because the
# score-table producer and contest auth eval both use the official video path.
# Lane #pr106_yshift_sidechannel — PR106 + per-frame Y-shift sidechannel
# (codex_metric_yshift SC01 mode-7 pattern, score_aware_sidechannel paradigm variant #3)
#
# DISPATCH GATE: only run AFTER lane_pr106_latent_sidecar lands an empirical
# contest-CUDA score < 0.20800. Per docs/INDEX_score_aware_sidechannel_thread_20260504.md
# decision pipeline TICK 2 — sequential validation prevents wasting GPU spend on
# stacking lanes that interact unexpectedly.
#
# Operator picks search mode via env var:
#   PR106_YSHIFT_MODE=zero        → CPU smoke (no real correction; wire format only)
#   PR106_YSHIFT_MODE=score_table → consume or build precomputed CUDA candidate table
#   PR106_YSHIFT_MODE=gradient    → CUDA, ~$0.20, ~5min
#                                   1 backward pass ∂score/∂{Y_off, dy, dx} per frame, quantize
#   PR106_YSHIFT_MODE=brute_force → CUDA, ~$0.40, ~10min
#                                   7×7×7 grid per frame; pick min(distortion)
#
# Pipeline (3 stages, single Vast.ai 4090 ~$0.30/hr × 30min ≈ $0.30):
#
#   Stage 1 (CPU OR CUDA): Build pr106_yshift archive on top of PR106 anchor
#                          (uses experiments/build_pr106_yshift_sidechannel.py)
#   Stage 2 (CPU): Local parser-roundtrip verification (cheap sanity check)
#   Stage 3 (CUDA-T4 or 4090): contest_auth_eval — score must be < lane_pr106_latent_sidecar
#                              landed score to ship as a stack-on improvement
#
# Predicted (per docs/codex_metric_yshift_audit_20260504.md):
#   gradient mode: -0.0005 to -0.0015 score Δ standalone
#   brute_force mode: -0.001 to -0.002 score Δ standalone
#   STACKED on lane_pr106_latent_sidecar: ~-0.003 score Δ total (orthogonal)
#
# Strict-scorer-rule: scorer is loaded ONLY at Stage 1 (gradient/brute_force search)
# AND Stage 3 (contest auth eval). NEVER at inflate time. Per-frame deltas are
# precomputed and frozen into the SC01 sidechannel payload before Stage 3.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
PR106_YSHIFT_MODE="${PR106_YSHIFT_MODE:-zero}"
SCORE_STEP="${PR106_YSHIFT_SCORE_STEP:-1.0}"
PR106_YSHIFT_CANDIDATE_RADIUS="${PR106_YSHIFT_CANDIDATE_RADIUS:-3}"
PR106_YSHIFT_N_PAIRS="${PR106_YSHIFT_N_PAIRS:-600}"
PR106_YSHIFT_SCORE_TABLE_NPY="${PR106_YSHIFT_SCORE_TABLE_NPY:-}"
PR106_YSHIFT_SCORE_TABLE_MANIFEST="${PR106_YSHIFT_SCORE_TABLE_MANIFEST:-}"
PR106_YSHIFT_SCORE_TABLE_LANE_ID="${PR106_YSHIFT_SCORE_TABLE_LANE_ID:-lane_pr106_yshift_score_table}"
PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID="${PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID:-}"
PR106_YSHIFT_SCORE_TABLE_BATCH_PAIRS="${PR106_YSHIFT_SCORE_TABLE_BATCH_PAIRS:-8}"
PR106_YSHIFT_SCORE_TABLE_CANDIDATE_BATCH_SIZE="${PR106_YSHIFT_SCORE_TABLE_CANDIDATE_BATCH_SIZE:-32}"
PR106_YSHIFT_SCORE_TABLE_RESUME="${PR106_YSHIFT_SCORE_TABLE_RESUME:-1}"
PR106_ARCHIVE="${PR106_ARCHIVE:-experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip}"

if ! [[ "$PR106_YSHIFT_MODE" =~ ^(zero|score_table|gradient|brute_force)$ ]]; then
    echo "FATAL: PR106_YSHIFT_MODE must be one of {zero, score_table, gradient, brute_force}; got: $PR106_YSHIFT_MODE" >&2
    exit 2
fi
if [ "$PR106_YSHIFT_MODE" = "score_table" ] && [ -n "$PR106_YSHIFT_SCORE_TABLE_NPY" ] && [ ! -f "$PR106_YSHIFT_SCORE_TABLE_NPY" ]; then
    echo "FATAL: supplied PR106_YSHIFT_SCORE_TABLE_NPY does not exist: $PR106_YSHIFT_SCORE_TABLE_NPY" >&2
    exit 2
fi

LANE_ID="lane_pr106_yshift_sidechannel_${PR106_YSHIFT_MODE}"
LOG_DIR="${PR106_YSHIFT_LOG_DIR:-$WORKSPACE/experiments/results/${LANE_ID}_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
log() { echo "[lane-pr106-yshift-${PR106_YSHIFT_MODE}] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }
if [ "$PR106_YSHIFT_MODE" = "score_table" ] && [ -z "$PR106_YSHIFT_SCORE_TABLE_NPY" ] && [ -z "$PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID" ]; then
    log "FATAL: score_table generation requires PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID matching an active lane claim"
    exit 2
fi
if [ -z "$PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID" ]; then
    PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID="$(basename "$LOG_DIR")"
fi

[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"

# Stage 0: NVDEC probe — required for non-zero modes before any GPU-work
# marker, including bare `nvidia-smi`.
if [ "$PR106_YSHIFT_MODE" != "zero" ] && [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] && [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        log "FATAL: NVDEC/DALI probe failed; exact CUDA eval is not trustworthy on this host."
        exit 2
    }
fi

cd "$WORKSPACE"

# Heartbeat (per CLAUDE.md remote-code-parity rule)
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do echo "$(date -u +%FT%TZ) lane-pr106-yshift alive" >> "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# ── Stage 0: Provenance + CUDA preflight (only enforce CUDA for non-zero modes) ──
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
"$PYBIN" -c "
import json, time, sys, os, torch
mode = '$PR106_YSHIFT_MODE'
if mode != 'zero' and not torch.cuda.is_available():
    sys.exit(f'FATAL: --device cuda required for mode={mode} per CLAUDE.md MPS-auth-eval-is-NOISE')
prov = {
    'lane_id': '$LANE_ID',
    'predicted_band': [0.2065, 0.208],
    'mode': mode,
    'score_step': float('$SCORE_STEP'),
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'pr106_archive': '$PR106_ARCHIVE',
    'cuda_available': bool(torch.cuda.is_available()),
}
with open('$LOG_DIR/provenance.json', 'w') as f:
    json.dump(prov, f, indent=2)
print(f'[stage-0] provenance written; CUDA={torch.cuda.is_available()}; mode={mode}; step={prov[\"score_step\"]}')
"

# ── Stage 1: Build pr106_yshift archive ───────────────────────────────────
log "=== Stage 1: build pr106_yshift sidechannel (mode=$PR106_YSHIFT_MODE, step=$SCORE_STEP) ==="
BUILD_DIR="$LOG_DIR/build"
mkdir -p "$BUILD_DIR"
if [ "$PR106_YSHIFT_MODE" = "score_table" ] && [ -z "$PR106_YSHIFT_SCORE_TABLE_NPY" ]; then
    log "=== Stage 1a: generate CUDA yshift score table ==="
    SCORE_TABLE_DIR="$LOG_DIR/score_table"
    mkdir -p "$SCORE_TABLE_DIR"
    PR106_YSHIFT_SCORE_TABLE_NPY="$SCORE_TABLE_DIR/score_table.npy"
    PR106_YSHIFT_SCORE_TABLE_MANIFEST="$SCORE_TABLE_DIR/score_table_manifest.json"
    if [ "$PR106_YSHIFT_SCORE_TABLE_RESUME" = "1" ] && [ -f "$PR106_YSHIFT_SCORE_TABLE_NPY" ] && [ -f "$PR106_YSHIFT_SCORE_TABLE_MANIFEST" ]; then
        log "Stage 1a RESUME: validating completed score table at $PR106_YSHIFT_SCORE_TABLE_NPY"
    fi
    SCORE_TABLE_ARGS=(
        experiments/build_pr106_yshift_score_table.py
        --pr106-archive "$PR106_ARCHIVE"
        --out-dir "$SCORE_TABLE_DIR"
        --candidate-radius "$PR106_YSHIFT_CANDIDATE_RADIUS"
        --score-step "$SCORE_STEP"
        --n-pairs "$PR106_YSHIFT_N_PAIRS"
        --batch-pairs "$PR106_YSHIFT_SCORE_TABLE_BATCH_PAIRS"
        --candidate-batch-size "$PR106_YSHIFT_SCORE_TABLE_CANDIDATE_BATCH_SIZE"
        --lane-id "$PR106_YSHIFT_SCORE_TABLE_LANE_ID"
        --instance-job-id "$PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID"
    )
    if [ "$PR106_YSHIFT_SCORE_TABLE_RESUME" = "1" ]; then
        SCORE_TABLE_ARGS+=(--resume-checkpoint)
    fi
    "$PYBIN" -u "${SCORE_TABLE_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
    if [ ! -f "$PR106_YSHIFT_SCORE_TABLE_NPY" ] || [ ! -f "$PR106_YSHIFT_SCORE_TABLE_MANIFEST" ]; then
        log "FATAL: score-table generation did not produce table+manifest"
        exit 3
    fi
fi
BUILD_ARGS=(
    experiments/build_pr106_yshift_sidechannel.py
    --pr106-archive "$PR106_ARCHIVE"
    --search-mode "$PR106_YSHIFT_MODE"
    --score-step "$SCORE_STEP"
    --candidate-radius "$PR106_YSHIFT_CANDIDATE_RADIUS"
    --n-pairs "$PR106_YSHIFT_N_PAIRS"
    --out-dir "$BUILD_DIR"
)
if [ "$PR106_YSHIFT_MODE" = "score_table" ]; then
    BUILD_ARGS+=(--score-table-npy "$PR106_YSHIFT_SCORE_TABLE_NPY")
    if [ -n "$PR106_YSHIFT_SCORE_TABLE_MANIFEST" ]; then
        BUILD_ARGS+=(--score-table-manifest "$PR106_YSHIFT_SCORE_TABLE_MANIFEST")
    fi
fi
"$PYBIN" -u "${BUILD_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
YSHIFT_ARCHIVE="$BUILD_DIR/pr106_yshift_sidechannel_archive.zip"
if [ ! -f "$YSHIFT_ARCHIVE" ]; then
    log "FATAL: stage 1 did not produce $YSHIFT_ARCHIVE"
    exit 3
fi
ARCHIVE_BYTES=$(stat -c '%s' "$YSHIFT_ARCHIVE" 2>/dev/null || stat -f '%z' "$YSHIFT_ARCHIVE")
log "stage 1 OK: archive bytes=$ARCHIVE_BYTES"

# ── Stage 2: Local parser-roundtrip sanity ────────────────────────────────
log "=== Stage 2: parser-roundtrip sanity (no GPU forward) ==="
"$PYBIN" -c "
import sys, zipfile
sys.path.insert(0, '$WORKSPACE/submissions/pr106_yshift_sidechannel')
sys.path.insert(0, '$WORKSPACE/submissions/apogee_intN/src')
from inflate import parse_yshift_archive
with zipfile.ZipFile('$YSHIFT_ARCHIVE') as z:
    bin_bytes = z.read('0.bin')
sd, lat, meta, sc = parse_yshift_archive(bin_bytes)
sc_status = 'present' if sc is not None else 'ABSENT'
sc_n = sc['raw'].shape[0] if sc is not None else 0
print(f'parse OK: {len(sd)} tensors, latents shape={tuple(lat.shape)}, sidechannel={sc_status} ({sc_n} frames)')
assert len(sd) == 28, f'expected 28 PR106 tensors, got {len(sd)}'
assert tuple(lat.shape) == (600, 28), f'expected (600, 28) latents, got {tuple(lat.shape)}'
assert sc is not None, 'sidechannel missing — build script bug'
assert sc['raw'].shape == (1200, 3), f'expected (1200, 3) sidechannel, got {sc[\"raw\"].shape}'
" 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Stage 3: Contest auth eval (CUDA T4 or 4090) ──────────────────────────
if [ "$PR106_YSHIFT_MODE" = "zero" ]; then
    log "=== Stage 3 SKIPPED: mode=zero produces no real distortion change. CPU smoke complete. ==="
    log "DONE: lane=$LANE_ID mode=$PR106_YSHIFT_MODE archive_bytes=$ARCHIVE_BYTES (no contest score; CPU wire-format proof; not contest-CUDA evidence)"
    exit 0
fi

log "=== Stage 3: contest auth eval (CUDA) ==="
INFLATE_SH="$WORKSPACE/submissions/pr106_yshift_sidechannel/inflate.sh"
EVAL_DIR="$LOG_DIR/eval"
mkdir -p "$EVAL_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$YSHIFT_ARCHIVE" \
    --inflate-sh "$INFLATE_SH" \
    --work-dir "$EVAL_DIR" \
    --keep-work-dir \
    --device cuda 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Final summary ─────────────────────────────────────────────────────────
SCORE_JSON="$EVAL_DIR/contest_auth_eval.json"
if [ -f "$SCORE_JSON" ]; then
    AUTH_SUMMARY=$("$PYBIN" -m tac.auth_eval_schema completion-summary "$SCORE_JSON")
    log "DONE: lane=$LANE_ID mode=$PR106_YSHIFT_MODE archive_bytes=$ARCHIVE_BYTES auth_eval_summary=$AUTH_SUMMARY"
    log "  beats lane_pr106_latent_sidecar landed score? Operator must compare via tools/score_dashboard.py."
    log "  cross-ref docs/INDEX_score_aware_sidechannel_thread_20260504.md TICK 2 gate."
else
    log "FATAL: stage 3 did not produce $SCORE_JSON"
    exit 4
fi
