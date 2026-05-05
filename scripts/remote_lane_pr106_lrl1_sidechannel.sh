#!/bin/bash
# NO_NVDEC_NEEDED — pure tensor-side codec + scorer-forward; no DALI/NVDEC video pipeline.
# Lane #pr106_lrl1_sidechannel — PR106 + per-frame LRL1 luma low-rank correction
# (codex_metric LRL1 mode-8 pattern, score_aware_sidechannel paradigm variant #6)
#
# DISPATCH GATE: only run AFTER BOTH lane_pr106_latent_sidecar AND
# lane_pr106_yshift_sidechannel land empirical contest-CUDA scores < 0.20650
# (per docs/INDEX_score_aware_sidechannel_thread_20260504.md decision pipeline
# TICK 3 — sequential validation). LRL1 is the natural 3rd stack-on after
# variants #1 (latent sidecar) and #3 (yshift) are empirically validated.
#
# Operator picks search mode via env var:
#   PR106_LRL1_MODE=zero        → CPU smoke (no real correction; wire format only)
#   PR106_LRL1_MODE=gradient    → CUDA, ~$0.30, ~7min
#                                 1 backward pass ∂score/∂(luma residual) per frame,
#                                 project onto top-K Lanczos eigenvectors → basis,
#                                 quantize per-frame coefs to int8.
#   PR106_LRL1_MODE=brute_force → CUDA, ~$0.50, ~12min
#                                 PCA on residual for basis; coordinate descent on
#                                 K-dim coef simplex per frame.
#
# Pipeline (3 stages, single Vast.ai 4090 ~$0.30/hr × 30min ≈ $0.30):
#
#   Stage 1 (CPU OR CUDA): Build pr106_lrl1 archive on top of PR106 anchor
#                          (uses experiments/build_pr106_lrl1_sidechannel.py)
#   Stage 2 (CPU): Local parser-roundtrip verification (cheap sanity check)
#   Stage 3 (CUDA-T4 or 4090): contest_auth_eval — score must be <
#                              min(latent_sidecar, yshift) landed score to ship
#                              as a stack-on improvement
#
# Predicted (per docs/codex_metric_lrl1_audit_20260504.md):
#   gradient mode: -0.001 to -0.002 score Δ standalone (K=2-4)
#   brute_force mode: -0.002 to -0.003 score Δ standalone
#   STACKED on lanes #1 + #3: ~-0.001 to -0.002 score Δ marginal
#   (orthogonal — luma low-rank operates at different basis than #1 latent or
#    #3 per-frame Y/translate)
#
# Strict-scorer-rule: scorer is loaded ONLY at Stage 1 (gradient/brute_force search)
# AND Stage 3 (contest auth eval). NEVER at inflate time. Per-frame deltas are
# precomputed and frozen into the LR01 sidechannel payload before Stage 3.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
PR106_LRL1_MODE="${PR106_LRL1_MODE:-zero}"
LRL1_K="${PR106_LRL1_K:-4}"
LRL1_LOW_H="${PR106_LRL1_LOW_H:-48}"
LRL1_LOW_W="${PR106_LRL1_LOW_W:-64}"
COEFF_STEP="${PR106_LRL1_COEFF_STEP:-1.0}"
BASIS_STEP="${PR106_LRL1_BASIS_STEP:-1.0}"
PR106_ARCHIVE="${PR106_ARCHIVE:-experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip}"

[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

if ! [[ "$PR106_LRL1_MODE" =~ ^(zero|gradient|brute_force)$ ]]; then
    echo "FATAL: PR106_LRL1_MODE must be one of {zero, gradient, brute_force}; got: $PR106_LRL1_MODE" >&2
    exit 2
fi

LANE_ID="lane_pr106_lrl1_sidechannel_${PR106_LRL1_MODE}"
LOG_DIR="$WORKSPACE/experiments/results/${LANE_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
log() { echo "[lane-pr106-lrl1-${PR106_LRL1_MODE}] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Heartbeat (per CLAUDE.md remote-code-parity rule)
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do echo "$(date -u +%FT%TZ) lane-pr106-lrl1 alive" >> "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# ── Stage 0: Provenance + CUDA preflight (only enforce CUDA for non-zero modes) ──
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
"$PYBIN" -c "
import json, time, sys, os, torch
mode = '$PR106_LRL1_MODE'
if mode != 'zero' and not torch.cuda.is_available():
    sys.exit(f'FATAL: --device cuda required for mode={mode} per CLAUDE.md MPS-auth-eval-is-NOISE')
prov = {
    'lane_id': '$LANE_ID',
    'mode': mode,
    'K': int('$LRL1_K'),
    'low_h': int('$LRL1_LOW_H'),
    'low_w': int('$LRL1_LOW_W'),
    'coeff_step': float('$COEFF_STEP'),
    'basis_step': float('$BASIS_STEP'),
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
print(f'[stage-0] provenance written; CUDA={torch.cuda.is_available()}; mode={mode}; K={prov[\"K\"]}; basis={prov[\"low_h\"]}x{prov[\"low_w\"]}')
"

# ── Stage 1: Build pr106_lrl1 archive ─────────────────────────────────────
log "=== Stage 1: build pr106_lrl1 sidechannel (mode=$PR106_LRL1_MODE, K=$LRL1_K, basis=${LRL1_LOW_H}x${LRL1_LOW_W}) ==="
BUILD_DIR="$LOG_DIR/build"
mkdir -p "$BUILD_DIR"
"$PYBIN" -u experiments/build_pr106_lrl1_sidechannel.py \
    --pr106-archive "$PR106_ARCHIVE" \
    --search-mode "$PR106_LRL1_MODE" \
    --K "$LRL1_K" \
    --low-h "$LRL1_LOW_H" \
    --low-w "$LRL1_LOW_W" \
    --coeff-step "$COEFF_STEP" \
    --basis-step "$BASIS_STEP" \
    --out-dir "$BUILD_DIR" 2>&1 | tee -a "$LOG_DIR/run.log"
LRL1_ARCHIVE="$BUILD_DIR/pr106_lrl1_sidechannel_archive.zip"
if [ ! -f "$LRL1_ARCHIVE" ]; then
    log "FATAL: stage 1 did not produce $LRL1_ARCHIVE"
    exit 3
fi
ARCHIVE_BYTES=$(stat -c '%s' "$LRL1_ARCHIVE" 2>/dev/null || stat -f '%z' "$LRL1_ARCHIVE")
log "stage 1 OK: archive bytes=$ARCHIVE_BYTES"

# ── Stage 2: Local parser-roundtrip sanity ────────────────────────────────
log "=== Stage 2: parser-roundtrip sanity (no GPU forward) ==="
"$PYBIN" -c "
import sys, zipfile
sys.path.insert(0, '$WORKSPACE/submissions/pr106_lrl1_sidechannel')
sys.path.insert(0, '$WORKSPACE/submissions/apogee_intN/src')
from inflate import parse_lrl1_archive
with zipfile.ZipFile('$LRL1_ARCHIVE') as z:
    bin_bytes = z.read('0.bin')
sd, lat, meta, sc = parse_lrl1_archive(bin_bytes)
sc_status = 'present' if sc is not None else 'ABSENT'
sc_K = sc['K'] if sc is not None else 0
sc_basis = sc['basis'].shape if sc is not None else None
sc_coeffs = sc['coeffs'].shape if sc is not None else None
print(f'parse OK: {len(sd)} tensors, latents shape={tuple(lat.shape)}, sidechannel={sc_status} (K={sc_K}, basis={sc_basis}, coeffs={sc_coeffs})')
assert len(sd) == 28, f'expected 28 PR106 tensors, got {len(sd)}'
assert tuple(lat.shape) == (600, 28), f'expected (600, 28) latents, got {tuple(lat.shape)}'
assert sc is not None, 'sidechannel missing — build script bug'
assert sc['K'] == int('$LRL1_K'), f'K mismatch: got {sc[\"K\"]}, expected $LRL1_K'
assert sc['low_h'] == int('$LRL1_LOW_H'), f'low_h mismatch'
assert sc['low_w'] == int('$LRL1_LOW_W'), f'low_w mismatch'
assert sc['coeffs'].shape == (1200, int('$LRL1_K')), f'expected (1200, $LRL1_K) coeffs, got {sc[\"coeffs\"].shape}'
" 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Stage 3: Contest auth eval (CUDA T4 or 4090) ──────────────────────────
if [ "$PR106_LRL1_MODE" = "zero" ]; then
    log "=== Stage 3 SKIPPED: mode=zero produces no real distortion change. CPU smoke complete. ==="
    log "DONE: lane=$LANE_ID mode=$PR106_LRL1_MODE archive_bytes=$ARCHIVE_BYTES (no contest score; CPU wire-format proof)"
    exit 0
fi

log "=== Stage 3: contest auth eval (CUDA) ==="
INFLATE_SH="$WORKSPACE/submissions/pr106_lrl1_sidechannel/inflate.sh"
EVAL_DIR="$LOG_DIR/eval"
mkdir -p "$EVAL_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$LRL1_ARCHIVE" \
    --inflate-sh "$INFLATE_SH" \
    --work-dir "$EVAL_DIR" \
    --device cuda 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Final summary ─────────────────────────────────────────────────────────
SCORE_JSON="$EVAL_DIR/contest_auth_eval.json"
if [ -f "$SCORE_JSON" ]; then
    SCORE=$("$PYBIN" -c "import json; print(json.load(open('$SCORE_JSON'))['final_score'])" 2>/dev/null || echo "PARSE_FAIL")
    log "DONE: lane=$LANE_ID mode=$PR106_LRL1_MODE archive_bytes=$ARCHIVE_BYTES contest_cuda_score=$SCORE"
    log "  beats min(lane_pr106_latent_sidecar, lane_pr106_yshift_sidechannel) landed score?"
    log "  Operator must compare via tools/score_dashboard.py."
    log "  cross-ref docs/INDEX_score_aware_sidechannel_thread_20260504.md TICK 3 gate."
else
    log "FATAL: stage 3 did not produce $SCORE_JSON"
    exit 4
fi
